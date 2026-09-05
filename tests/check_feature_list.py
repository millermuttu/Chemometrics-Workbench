"""Check a feature list against the rules the working protocol states.

    uv run python -m tests.check_feature_list
    uv run python -m tests.check_feature_list docs/phase-1-3/feature_list.json

`clean-state-checklist.md` check 3 exists to catch a feature marked `passing`
because the code looked right, with no evidence behind it. It said to pass when
something printed `feature_list.json consistent`, and nothing printed that —
the check passed by being read. This is the command it was describing.

**It checks what is mechanical and says so.** Whether an `evidence` string
records a real command with its real output is a human reading it; this module
only asserts that the string is not empty. Check 3 keeps both halves and this
module is the first one. Claiming the second would be the failure the check was
written to catch, one level up.

Every rule below is in `CLAUDE.md`'s working protocol. Nothing here is a new
convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "feature_list.json"


def problems(document: dict[str, Any]) -> list[str]:
    """Every rule the document breaks, as sentences naming the feature."""
    found: list[str] = []
    features = document["features"]
    allowed = set(document["status_values"])

    ids: dict[str, int] = {}
    for feature in features:
        ids[feature["id"]] = ids.get(feature["id"], 0) + 1
    found.extend(
        f"{name!r} appears {count} times; ids are the handle, so they must be unique."
        for name, count in sorted(ids.items())
        if count > 1
    )

    status = {feature["id"]: feature["status"] for feature in features}

    for feature in features:
        name = feature["id"]
        if feature["status"] not in allowed:
            found.append(
                f"{name!r} has status {feature['status']!r}, which is not one of {sorted(allowed)}."
            )
        if feature["status"] == "passing" and not feature["evidence"].strip():
            found.append(
                f"{name!r} is passing with empty evidence. A feature becomes passing only "
                "after its verification steps have actually been run."
            )
        if feature["status"] == "blocked" and not feature["notes"].strip():
            found.append(
                f"{name!r} is blocked with empty notes. Record what is blocking it and what "
                "would unblock it."
            )
        for dependency in feature["depends_on"]:
            if dependency not in status:
                found.append(f"{name!r} depends on {dependency!r}, which is not in this list.")
            elif feature["status"] == "passing" and status[dependency] != "passing":
                found.append(
                    f"{name!r} is passing but depends on {dependency!r}, which is "
                    f"{status[dependency]}."
                )

    in_progress = [f["id"] for f in features if f["status"] == "in_progress"]
    if len(in_progress) > 1:
        found.append(
            f"{len(in_progress)} features are in_progress ({', '.join(sorted(in_progress))}). "
            "At most one may be."
        )
    return found


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    path = Path(arguments[0]) if arguments else DEFAULT
    document = json.loads(path.read_text(encoding="utf-8"))

    found = problems(document)
    if found:
        for problem in found:
            print(problem)
        return 1
    print(f"{path.name} consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
