# Clean-state checklist

Run through this before ending a session. It exists so the next session can start working immediately instead of first repairing the last one.

Each check has a pass condition. A check that fails is fixed now, not noted for later — "noted for later" is how a repository stops being trustworthy.

This file is deliberately generic and should not need editing as the project grows. **The project's canonical setup and verification commands live in `CONTRIBUTING.md`.** When they change, update them there; the checks below refer to them by role, not by name.

---

## 1. Standard verification still runs

Run the project's full verification suite as documented — lint, formatting, type checking and tests.

**Pass when:** every command exits 0. A skipped, xfailed or excluded test is acceptable only if its reason is written down somewhere durable.

**Never accept a green run you did not actually see.** If the suite was not run this session, this check has not passed.

## 2. Handoff note is current

`session-handoff.md` is a snapshot, not a log — rewrite it, do not append to it.

**Pass when** it describes the state as of right now: current branch and sync status, the active feature or none, the next concrete action, anything waiting on a decision from the user, and any new gotcha discovered this session. Its `Updated:` date is today's.

## 3. Feature list reflects actual state

The failure this catches: a feature marked `passing` because the code looked right, with no evidence behind it.

```bash
uv run python -m tests.check_feature_list
```

It checks the mechanical half — statuses, unique ids, dependencies that exist and are finished, `passing` with evidence, `blocked` with a reason, at most one `in_progress` — and prints the problems it finds, one per line.

**Pass when** it prints `feature_list.json consistent`, **and** you have read the `evidence` of every feature changed this session and confirmed it records a real command with its real output and a date — not a description of what would happen. **The command cannot do that second half**, which is the half that catches the failure above: only a person reading the string can tell a recorded run from a description of one.

## 4. No half-finished work left unrecorded

**Pass when:** the working tree is clean or every remaining change is named in `session-handoff.md`; there are no forgotten stashes; the branch is pushed; and any feature still `in_progress` carries a note saying exactly where it stands and what comes next.

## 5. Next session needs no manual fixes

The judgement call, and the one worth being honest about. Ask:

- Would a fresh session, reading only `session-handoff.md`, know what to do first?
- Does anything work **only** because of state on this machine — an unpushed commit, an uninstalled dependency, an environment variable, a file outside the repository?
- Was anything discovered this session that is not written down anywhere?
- Is any decision waiting on the user that is not listed under *Waiting on the user* in the handoff?

**Pass when** all four answers are satisfactory. When one is not, the fix is a paragraph in `session-handoff.md` — not memory.

---

