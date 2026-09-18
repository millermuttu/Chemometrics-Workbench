/** #219: what one line of the model registry reads as.
 *
 * A saved model's row carries one figure, and which one depends on what the
 * model is: a cross-validated regression leads with RMSECV, one fitted without
 * a split has only RMSEC, a classification has accuracy, and a decomposition
 * has no error to quote at all. A metric a model does not carry is absent
 * rather than zero (`metrics-and-validation.md` section 11), so the ladder
 * falls through rather than printing `0.0000`.
 */
import { describe, expect, it } from "vitest";

import type { ModelRow } from "@/api/queries";
import { modelFigure } from "@/shell/Sidebar";

const NONE = {
  rmsec: null,
  rmsecv: null,
  rmsep: null,
  r2: null,
  q2: null,
  accuracy: null,
  explained_variance: null,
};

function model(metrics: Partial<ModelRow["metrics"]> = {}, task = "regression"): ModelRow {
  return {
    model_id: "m1",
    experiment_id: "e1",
    name: "Fat, SNV + centre",
    task,
    node_id: "pls",
    artifact_path: "models/m1.cwmodel",
    artifact_hash: "sha256:" + "1".repeat(64),
    created_at: "2026-09-18T10:00:00+00:00",
    metrics: { ...NONE, ...metrics },
  };
}

describe("a saved model's row", () => {
  it("leads with the cross-validated error when the model has one", () => {
    expect(modelFigure(model({ rmsec: 0.21, rmsecv: 0.3891 }))).toBe("RMSECV 0.3891");
  });

  it("falls back to the calibration error for a model fitted above a split", () => {
    expect(modelFigure(model({ rmsec: 0.2143 }))).toBe("RMSEC 0.2143");
  });

  it("reads a classification as its accuracy", () => {
    expect(modelFigure(model({ accuracy: 0.9583 }, "classification"))).toBe("accuracy 0.958");
  });

  it("reads a decomposition as its first component's share", () => {
    expect(modelFigure(model({ explained_variance: 0.689 }, "decomposition"))).toBe("PC1 68.9%");
  });

  it("says what the model is when it carries no metric at all", () => {
    // Absent, not zero: a row reading `RMSEC 0.0000` would be a claim the
    // model never made.
    expect(modelFigure(model({}, "decomposition"))).toBe("decomposition");
  });
});
