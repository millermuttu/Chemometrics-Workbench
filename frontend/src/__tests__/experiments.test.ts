/** #209: what one line of the history reads as.
 *
 * A run keeps its number as later ones arrive, and the one figure a row
 * carries is whichever headline that run has - or its status, when it did not
 * succeed, because a failed experiment is a result and the status is the thing
 * to read (PROPOSAL.md section 8.2).
 */
import { describe, expect, it } from "vitest";

import type { ExperimentRow } from "@/api/queries";
import { runFigure, runLabel } from "@/shell/Sidebar";

function row(overrides: Partial<ExperimentRow> = {}): ExperimentRow {
  return {
    experiment_id: "e1",
    status: "succeeded",
    started_at: "2026-09-18T10:00:00+00:00",
    finished_at: "2026-09-18T10:00:30+00:00",
    pipeline_hash: "sha256:" + "1".repeat(64),
    n_nodes: 4,
    dataset_version_id: "v1",
    error: null,
    metrics: { rmsecv: null, q2: null, accuracy: null, explained_variance: null },
    ...overrides,
  };
}

describe("a row in the history", () => {
  it("numbers runs from the project's first, so a number does not move", () => {
    // Newest first: index 0 of three is the third run this project has done.
    expect(runLabel(row(), 0, 3)).toBe("Run 3");
    expect(runLabel(row(), 2, 3)).toBe("Run 1");
    // A fourth arrives; the first is still Run 1.
    expect(runLabel(row(), 3, 4)).toBe("Run 1");
  });

  it("carries the headline the run has, by task", () => {
    expect(runFigure(row({ metrics: { rmsecv: 0.3891, q2: 0.97, accuracy: null, explained_variance: null } })))
      .toBe("RMSECV 0.3891");
    expect(runFigure(row({ metrics: { rmsecv: null, q2: null, accuracy: 0.9583, explained_variance: null } })))
      .toBe("accuracy 0.958");
    expect(runFigure(row({ metrics: { rmsecv: null, q2: null, accuracy: null, explained_variance: 0.689 } })))
      .toBe("PC1 68.9%");
  });

  it("says the status instead when the run did not succeed, or scored nothing", () => {
    expect(runFigure(row({ status: "failed", error: "node 'pca' failed" }))).toBe("failed");
    expect(runFigure(row({ status: "cancelled" }))).toBe("cancelled");
    expect(runFigure(row({ metrics: null }))).toBe("succeeded");
  });
});
