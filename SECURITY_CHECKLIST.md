# Review before publication

- [ ] Confirm university, industry, and coauthor authorization to release source code.
- [ ] Review the full Git history: publish this directory in a **new** repository, never push an existing research repository or its history.
- [ ] Inspect all source and documentation manually for project-specific paths, credentials, identifiers, metadata, and inappropriate third-party code.
- [ ] Exclude all local files, generated artifacts, trained parameters, logs, notebooks, private configurations, and executable checkpoints.
- [ ] Verify license requirements of prior architecture implementations and add proper attribution.
- [ ] Choose and add an approved open-source license (or keep access restricted).
- [ ] Run synthetic-only tests; review the exact staged file list before any commit.
- [ ] Check commit author email; use GitHub noreply email if preferred.
- [ ] Validate the manuscript architecture against this reference code.
- [ ] Confirm the linked tag/release and citation metadata after publication.

A `.gitignore` file is a secondary safeguard, not a substitute for checking staged files.
Never run `git add -A` from your private research repository.
