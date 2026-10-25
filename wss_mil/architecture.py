"""WSS-MIL architecture source (no pretrained parameters).

The default class below exposes optional ablation flags; the paper's
reference configuration uses use_ap=False and pooling_mode="max".
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from . import config

class CaryTree(nn.Module):

    def __init__(self, C=config.C, T=config.WINDOW_SIZE):
        super().__init__()
        self.C = C
        self.T = T
        self.num_levels = math.ceil(math.log(T, C)) + 1

    def forward(self, leaf_features):
        B, T, D = leaf_features.shape
        levels = [leaf_features]
        current = leaf_features
        for level in range(self.num_levels - 1):
            N = current.size(1)
            pad_size = (self.C - N % self.C) % self.C
            if pad_size > 0:
                padding = torch.zeros(B, pad_size, D, device=current.device)
                current = torch.cat([current, padding], dim=1)
            N_padded = current.size(1)
            current = current.view(B, N_padded // self.C, self.C, D)
            current = current.max(dim=2).values
            levels.append(current)
        return levels


class SelfAttentionBlock(nn.Module):
    """Pre-norm full multi-head self-attention within one tree level."""

    def __init__(self, d_model=config.D, n_heads=config.N_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x


class LevelWiseSelfAttentionEncoder(nn.Module):
    """Reuse the same self-attention blocks independently at each tree level.

    No graph adjacency or graph-attention operation is used.
    """

    def __init__(self, num_blocks=config.NUM_TRANSFORMER_BLOCKS, d_model=config.D):
        super().__init__()
        self.blocks = nn.ModuleList([SelfAttentionBlock(d_model) for _ in range(num_blocks)])

    def forward(self, levels):
        processed_levels = []
        for level_features in levels:
            h = level_features
            for block in self.blocks:
                h = block(h)
            processed_levels.append(h)
        return processed_levels


class VectorQuantization(nn.Module):

    def __init__(self, num_levels, K=config.K, D=config.D, decay=0.99):
        super().__init__()
        self.num_levels = num_levels
        self.K = K
        self.D = D
        self.decay = decay
        for s in range(num_levels):
            self.register_buffer(f'codebook_{s}', torch.randn(K, D))
            self.register_buffer(f'ema_count_{s}', torch.ones(K))

    def get_codebook(self, level_idx):
        return getattr(self, f'codebook_{level_idx}')

    def quantize_level(self, features, level_idx):
        B, N, D = features.shape
        codebook = self.get_codebook(level_idx)
        flat = features.reshape(-1, D)
        flat_detached = flat.detach()
        distances = torch.cdist(flat_detached, codebook)
        indices = distances.argmin(dim=1)
        quantized = codebook[indices].reshape(B, N, D)
        if self.training:
            with torch.no_grad():
                ema_count = getattr(self, f'ema_count_{level_idx}')
                one_hot = F.one_hot(indices, self.K).float()
                new_count = one_hot.sum(0)
                ema_count.mul_(self.decay).add_(new_count, alpha=1 - self.decay)
                dw = one_hot.T @ flat_detached
                n = ema_count.unsqueeze(1).clamp(min=1.0)
                codebook.mul_(self.decay).add_(dw / n, alpha=1 - self.decay)
        quantized_st = features + (quantized - features).detach()
        commit_loss = F.mse_loss(features, quantized.detach())
        distances_for_entropy = torch.cdist(flat_detached, codebook.detach()).reshape(B, N, self.K)
        probs = F.softmax(-distances_for_entropy, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1) / math.log(self.K)
        return (quantized_st, commit_loss, entropy)

    def forward(self, levels):
        quantized_levels = []
        total_commit_loss = 0.0
        all_entropies = []
        for s, features in enumerate(levels):
            quantized, commit_loss, entropy = self.quantize_level(features, s)
            quantized_levels.append(quantized)
            total_commit_loss += commit_loss
            all_entropies.append(entropy)
        total_commit_loss /= len(levels)
        return (quantized_levels, total_commit_loss, all_entropies)


class AttentionPooling(nn.Module):

    def __init__(self, d_model=config.D):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))

    def forward(self, features):
        scores = self.score_net(features).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return (pooled, weights)


class OriginalAttentionPooling(nn.Module):

    def __init__(self, d_model=config.D):
        super().__init__()
        self.score_net = nn.Linear(d_model, 1)

    def forward(self, features):
        scores = self.score_net(features)
        weighted = features * scores
        pooled = weighted.max(dim=1).values
        return (pooled, scores.squeeze(-1))


class ConfigurableTemporalEmbedding(nn.Module):

    def __init__(self, input_channels=1, d_model=config.D, kernel_size=config.CONV_KERNEL, use_pos_embedding=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, d_model, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, config.WINDOW_SIZE, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.relu(self.conv(x))
        h = h.transpose(1, 2)
        if self.use_pos_embedding:
            h = h + self.pos_embedding[:, :h.size(1), :]
        h = self.norm(h)
        return h


class PlainMaxPooling(nn.Module):

    def forward(self, features):
        pooled = features.max(dim=1).values
        B, N, _ = features.shape
        weights = torch.full((B, N), 1.0 / N, device=features.device)
        return (pooled, weights)


class ComponentAblationTreeMIL(nn.Module):
    """Hierarchical MIL with level-wise full self-attention and optional ablations."""

    def __init__(self, use_pos_embedding=True, use_tree=True, use_attention=True, use_vq=True, use_ap=True, use_entropy_loss=True, use_commit_loss=True, pooling_mode='max'):
        super().__init__()
        T = config.WINDOW_SIZE
        D = config.D
        self.use_pos_embedding = use_pos_embedding
        self.use_tree = use_tree
        self.use_attention = use_attention
        self.use_vq = use_vq
        self.use_ap = use_ap
        self.use_entropy_loss = use_entropy_loss and use_vq
        self.use_commit_loss = use_commit_loss and use_vq
        num_levels = math.ceil(math.log(T, config.C)) + 1 if use_tree else 1
        self.num_levels = num_levels
        self.temporal_embedding = ConfigurableTemporalEmbedding(use_pos_embedding=use_pos_embedding)
        self.tree = CaryTree() if use_tree else None
        self.self_attention = LevelWiseSelfAttentionEncoder() if use_attention else None
        self.vq = VectorQuantization(num_levels) if use_vq else None
        if not use_ap:
            self.attention_pooling = PlainMaxPooling()
        elif pooling_mode == 'attention_mlp':
            self.attention_pooling = AttentionPooling()
        else:
            self.attention_pooling = OriginalAttentionPooling()
        self.classifier = nn.Linear(D, 1)
        self.point_scorer = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1))

    def _zero_entropies(self, levels):
        return [torch.zeros(lv.size(0), lv.size(1), device=lv.device, dtype=lv.dtype) for lv in levels]

    def forward(self, x, return_point_scores=False, return_pooling_weights=False):
        T = x.size(2)
        leaf_features = self.temporal_embedding(x)
        if self.use_tree:
            levels = self.tree(leaf_features)
        else:
            levels = [leaf_features]
        if self.use_attention:
            levels = self.self_attention(levels)
        if self.use_vq:
            quantized_levels, commitment_loss, entropies = self.vq(levels)
        else:
            quantized_levels = levels
            commitment_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            entropies = self._zero_entropies(levels)
        all_instances = torch.cat(quantized_levels, dim=1)
        pooled, attn_weights = self.attention_pooling(all_instances)
        bag_logit = self.classifier(pooled).squeeze(-1)
        bag_pred = torch.sigmoid(bag_logit)
        if self.use_entropy_loss:
            entropy_loss = sum((e.mean() for e in entropies)) / len(entropies)
        else:
            entropy_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if not self.use_commit_loss:
            commitment_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        output = {'bag_pred': bag_pred, 'bag_logit': bag_logit, 'commitment_loss': commitment_loss, 'entropy_loss': entropy_loss}
        if return_point_scores:
            output['point_scores'] = self._compute_point_scores(quantized_levels, entropies, T)
        if return_pooling_weights:
            output['pooling_weights'] = attn_weights
        return output

    def _compute_point_scores(self, quantized_levels, entropies, T):
        B = quantized_levels[0].size(0)
        leaf_quantized = quantized_levels[0][:, :T, :]
        id_logits = self.point_scorer(leaf_quantized).squeeze(-1)
        id_scores = torch.sigmoid(id_logits)
        leaf_entropy = entropies[0][:, :T]
        ood_scores = leaf_entropy.clone()
        count = 1
        for s in range(1, len(entropies)):
            level_ent = entropies[s]
            N_s = level_ent.size(1)
            if N_s == 0:
                continue
            coverage = max(1, T // N_s)
            expanded = level_ent.unsqueeze(-1).expand(-1, -1, coverage)
            expanded = expanded.reshape(B, -1)
            if expanded.size(1) >= T:
                ood_scores = ood_scores + expanded[:, :T]
                count += 1
        ood_scores = ood_scores / count
        eta = config.ETA
        return eta * id_scores + (1 - eta) * ood_scores
