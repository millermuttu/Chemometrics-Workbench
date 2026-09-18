import { describe, expect, it } from "vitest";

import type { PcaPayload } from "@/api/queries";
import { nodeMetrics } from "@/shell/nodeMetrics";

const pca = {
  task: "decomposition",
  n_components: 3,
  explained_variance_ratio: [0.6, 0.2, 0.1],
  cumulative_explained_variance: [0.6, 0.8, 0.9],
} as PcaPayload;

describe("nodeMetrics", () => {
  it("reads a decomposition's own variances", () => {
    expect(nodeMetrics(pca)).toEqual({
      "PC1 variance": 0.6,
      "PC1-3 cumulative": 0.9,
      components: 3,
    });
  });

  it("gives a regression its own metrics, and an absent one as null", () => {
    const pls = { ...pca, task: "regression", n_components: 5, metrics: { rmsec: 1.5, r2: 0.9 } };
    expect(nodeMetrics(pls as PcaPayload)).toEqual({
      RMSEC: 1.5,
      "R²": 0.9,
      RMSECV: null,
      "Q²": null,
      components: 5,
    });
  });

  it("has nothing to say about a node with no result", () => {
    expect(nodeMetrics(undefined)).toBeUndefined();
  });
});

it("labels a classification by its accuracies, absent as null (#185)", () => {
  const plsda = {
    task: "classification",
    n_components: 5,
    metrics: { accuracy: 0.95, sensitivity: 0.9, specificity: 1.0 },
  } as unknown as PcaPayload;
  expect(nodeMetrics(plsda)).toEqual({
    Accuracy: 0.95,
    "Accuracy (CV)": null,
    Sensitivity: 0.9,
    Specificity: 1.0,
    components: 5,
  });
});
