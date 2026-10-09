# Releasing

Each GitHub release is archived on Zenodo, which mints a DOI for that version. The concept DOI always resolves to the latest version.

## Before the first release (once)

1. Sign in to [Zenodo](https://zenodo.org) with GitHub, open **Settings → GitHub**, and switch on `soroushdty/LM-ContextProbe`. Zenodo only archives releases published *after* this is switched on.
2. Zenodo builds the record from `CITATION.cff` (title, authors with ORCID, abstract, keywords, license, version). Check those fields first; `cffconvert -i CITATION.cff -f zenodo` previews what Zenodo will see.

## Each release

1. **Protocol.** If any change since the last release alters reported values for the same data and model (an endpoint definition, statistic, test, or a default that feeds them), bump `PROTOCOL_VERSION` in `shared/endpoints.py` and add a row to the protocol history in [`methodology.md`](methodology.md#protocol-version).
2. **Version.** Set the same version in `pyproject.toml` and `CITATION.cff`.
3. **Changelog.** Replace "unreleased" in the version's heading in `CHANGELOG.md` with the release date.
4. **Check.** `pytest` passes, CI is green on `main`, and `cffconvert --validate -i CITATION.cff` passes.
5. **Tag and publish.** Merge to `main`, then publish a GitHub release with tag `vX.Y.Z` and the version's changelog section as release notes. Zenodo archives it within a few minutes.
6. **After the first DOI.** Add the concept DOI to `CITATION.cff` (`doi:`) and a DOI badge under the README title, in a follow-up commit. Later releases keep the same concept DOI.
