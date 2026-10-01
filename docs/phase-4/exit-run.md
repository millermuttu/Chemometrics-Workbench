# Phase 4 exit run

`PROPOSAL.md` §16, Phase 4 exit criterion: *a non-developer on a clean machine downloads, installs
and completes a PCA in under ten minutes.* This is also one of §18's metrics for 1.0.

**Status: not yet run.** This page holds the protocol and a blank record for each session. A real
person runs it; the maintainer arranges the session (decided 2026-09-28). No automated run stands in
for it. The release, the documentation site and the example data have to be public first, which
means `dev` merged to `main` and the Pages deploy answering.

## Protocol

- **Machine.** No Python, Node, Git or developer tools installed. Record the platform and the exact
  OS version. At least one run on Windows, which §0 assumes is the largest audience; ideally one per
  platform.
- **Tester.** A non-developer. They get two links and nothing else:
  - the page of the release under test, by tag: for now
    `https://github.com/millermuttu/Chemometrics-Workbench/releases/tag/v0.9.0-rc2`. Not `/releases/latest`:
    GitHub skips pre-releases there, and on 2026-10-01 it led to v0.8.0, which has no packages;
  - the documentation site, `https://millermuttu.github.io/Chemometrics-Workbench/`.
- **No help.** The observer does not speak unless the tester is stuck for more than two minutes. A
  question asked, or help given, is recorded as a finding, with the time it happened.
- **Clock.** It starts when the release page opens. It stops when the scores plot of a PCA on the
  Tecator example (`docs/examples/tecator.csv`, downloadable from the site's worked example) is on
  screen.
- **Pass.** Under ten minutes, with no help given. A run with help given is recorded but does not
  pass.
- **Findings.** Every obstacle becomes a GitHub issue, fixed or consciously accepted with the reason
  here before the phase closes.

## Observer's sheet

Write down the elapsed time (mm:ss) at each milestone. A hesitation is any pause of 15 s or more, or
a wrong click that had to be undone.

| # | Milestone | What counts as reached |
| --- | --- | --- |
| 1 | Package downloaded | The `.zip` / `.dmg` / `.tar.gz` is on disk |
| 2 | Unpacked | The application folder or app bundle is visible |
| 3 | OS warning passed | Past SmartScreen or Gatekeeper, or none shown |
| 4 | Workbench open | The browser shows **This project is empty** |
| 5 | Example data downloaded | `tecator.csv` is on disk |
| 6 | Imported | The dataset's table is showing |
| 7 | Pipeline built | Any valid pipeline ending in PCA is saved |
| 8 | Run done | The status bar says **Done** |
| 9 | Scores plot | The PCA's scores plot is on screen, and the clock stops |

## Session record

Copy this block once per session.

```
Date:
Tester (role, not name):
Observer:
Platform and OS version:
Release tag:
Browser the workbench opened in:

Times (mm:ss):
  1 downloaded       :
  2 unpacked         :
  3 warning passed   :   warning shown (verbatim):
  4 workbench open   :
  5 data downloaded  :
  6 imported         :
  7 pipeline built   :
  8 run done         :
  9 scores plot      :

Hesitations (time, where, what they did):
Questions asked or help given (time, what):
Errors or warnings seen in the application:

Total time:
Verdict: pass / fail (and why)
Issues opened:
```

## Pre-flight

Not a session, and not a stand-in for one: checks a developer made so that a tester does not hit
an obstacle already known.

- **2026-10-01.** `/releases/latest` led to v0.8.0, which has no packages, because GitHub skips
  pre-releases there. The protocol now names the tag. `v0.9.0-rc2` was tagged on `main` at
  fd39b41 and published all three packages: Linux 66 MB, macOS 46 MB, Windows 59 MB. That is inside
  the ranges `docs/how-to-open.md` gives. The Linux archive, downloaded from the release page,
  unpacks to 178 MB and passes `tests.smoke_package` (`served a 240 x 2 PCA`), and its bundle carries
  #262's reload. The Windows and macOS packages are smoke-tested only by CI, in the release build.

## Sessions

None yet.

## Verdict

Not met until at least one Windows session passes and every issue opened from the sessions is closed
or accepted here.
