# Public release sequence

1. Create a GitHub account if one is not already available, enable two-factor authentication, and create a public repository such as `STNLR-materials`.
2. Install Git LFS, run `git lfs install`, and confirm that `.gitattributes` tracks `*.pt` and `*.npz` before the first commit.
3. Commit the complete contents of this release directory. Do not upload access tokens, SSH keys, private paths, unpublished reviewer material, or unrelated experiment folders.
4. Push the repository and create a versioned release tag such as `v1.0.0`. Record the exact public repository URL.
5. Connect the repository to Zenodo, archive the tagged release, and upload the complete data/checkpoint archive if GitHub LFS bandwidth is not sufficient. Reserve or mint a DOI.
6. Add the repository URL to `CITATION.cff` as `repository-code` and `url`; add the DOI as `doi`. Run `python scripts/build_release_manifest.py` and then `python scripts/verify_release.py` before publishing the amended release.
7. Replace the provisional manuscript availability text with the exact GitHub URL and archival DOI. Use the archived version cited by the manuscript, not a moving branch URL.

The repository can be prepared under a private visibility setting, but the URL and DOI should be public no later than the availability point promised in the accepted manuscript. A public archival DOI is stronger than a GitHub account or repository link alone because it fixes the cited version.
