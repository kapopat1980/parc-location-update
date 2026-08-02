# Publishing: GitHub → Zenodo DOI

## Before you start

Replace every placeholder. Search for `[` across these files:

| File | Placeholders |
|---|---|
| `README.md` | `[AUTHOR NAME]`, `[USERNAME]` |
| `LICENSE` | `[AUTHOR NAME]` |
| `.zenodo.json` | `[SURNAME, Given Name]`, `[AFFILIATION]`, `[ORCID]` |
| `CITATION.cff` | `[SURNAME]`, `[GIVEN NAME]`, `[AFFILIATION]`, `[ORCID]` |

No ORCID yet? Get one free at **orcid.org** — takes two minutes and journals increasingly
require it anyway.

## 1. Push to GitHub

The repository must be **public**; Zenodo cannot see private repositories.

```bash
cd parc-location-update
git init
git add .
git commit -m "PARC: adaptive predictive location updating — simulation code and results"
git branch -M main
git remote add origin https://github.com/<USERNAME>/parc-location-update.git
git push -u origin main
```

## 2. Connect Zenodo

**zenodo.org** → *Log in* → **Log in with GitHub** → authorise (it needs `admin:repo_hook`
to watch for releases).

## 3. Enable the repository — BEFORE releasing

**zenodo.org/account/settings/github/** → find `parc-location-update` → toggle **ON**.

> ⚠️ **The one trap.** Zenodo only archives releases created *after* the toggle is on.
> A release made first will be silently ignored. If that happens, just cut another release.

## 4. Create a GitHub release

Repo → *Releases* → *Create a new release* → tag `v1.0` → title
"PARC v1.0 — paper submission" → *Publish release*.

## 5. Wait ~2 minutes

Zenodo archives the tarball and mints the DOI. It appears next to the repo at
zenodo.org/account/settings/github/, along with a badge snippet.

## 6. Check the metadata

`.zenodo.json` should populate the record correctly. Open the Zenodo record anyway and confirm
the author name, ORCID and affiliation are right — if `.zenodo.json` had a syntax error Zenodo
falls back to GitHub metadata, which usually lists your *username* as the author. Edit and
republish if needed; editing metadata does not change the DOI.

## 7. Use the CONCEPT DOI

You get two:

| DOI | Resolves to | Use |
|---|---|---|
| **Concept DOI** | Always the latest version | ✅ **Cite this in the paper** |
| Version DOI | This release only | Exact-version reproduction |

The concept DOI is the one to cite. If you later fix the radius selector and release v1.1, the
paper's link still resolves.

## 8. Add the link to the paper

Section 11 (Reproducibility) currently describes the code without pointing to it. Add:

> The complete simulator, raw per-subscriber results and analysis scripts are available at
> https://doi.org/&lt;CONCEPT-DOI&gt;.

**Do not add this line until the DOI exists.** A dead link in a submitted paper is worse than no
link at all.

## 9. Optional: README badge

Zenodo gives you a Markdown badge. Paste it under the README title:

```markdown
[![DOI](https://zenodo.org/badge/DOI/<CONCEPT-DOI>.svg)](https://doi.org/<CONCEPT-DOI>)
```

---

## Notes

- **Repository size** is 2.0 MB. Well within GitHub's comfortable range — no Git LFS needed.
- **Timing.** Mint the DOI *before* submitting, so the paper can cite it. If the journal requires
  anonymised review, some editors accept an anonymised Zenodo link; check the author guidelines.
- **Updating after review.** Cut a new release (`v1.1`) and Zenodo mints a new version DOI
  automatically under the same concept DOI. Nothing in the paper needs changing.
