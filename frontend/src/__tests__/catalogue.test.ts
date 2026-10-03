/** #182: the drop menu's PLS entry models a column the dataset has, or is
 * not offered. It used to write `target: "fat"` into every PLS node. */
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import { STEPS, stepMenu } from "@/canvas/catalogue";
import { parameterLine } from "@/canvas/graph";
import { nodeLabel } from "@/shell/Sidebar";

describe("the PLS menu entry", () => {
  it("models the dataset's first target", () => {
    const pls = stepMenu(["moisture", "fat"]).find((step) => step.kind === "PLS 5 LV")!;
    expect(pls.payload.spec).toMatchObject({ kind: "pls", target: "moisture" });
    expect(pls.parameters).toBe("5 components · moisture");
  });

  it("offers PLS-DA only for a dataset with a two-valued column, on that column", () => {
    const plsda = stepMenu(["fat"], ["fat_class"]).find((step) => step.kind === "PLS-DA 5 LV")!;
    expect(plsda.payload.spec).toMatchObject({ kind: "plsda", class_column: "fat_class" });
    expect(stepMenu(["fat"]).map((step) => step.kind)).not.toContain("PLS-DA 5 LV");
  });

  it("is absent when the dataset has nothing to model", () => {
    const kinds = stepMenu([]).map((step) => step.kind);
    expect(kinds).not.toContain("PLS 5 LV");
    expect(kinds).toContain("K-fold 10");
    expect(kinds).toContain("Train/test 25%");
  });
});

describe("the step list", () => {
  it("offers a normalisation, which the server has always accepted (#238)", () => {
    const normalise = stepMenu([]).find((step) => step.kind === "Normalise")!;
    expect(normalise.payload.step).toEqual({ kind: "normalise", norm: "l2" });
  });
});

it("offers the four smoothers, each labelled as the canvas labels it (#271)", () => {
  const smoothers = STEPS.filter((step) =>
    ["moving_average", "median", "gaussian", "whittaker"].includes(
      String((step.payload.step as { kind: string } | undefined)?.kind),
    ),
  );
  expect(smoothers.map((step) => step.kind)).toEqual([
    "Moving avg w5",
    "Median w5",
    "Gaussian σ1.5",
    "Whittaker λ100",
  ]);
  for (const step of smoothers) {
    const node = { id: "n", type: "preprocess", inputs: ["source"], ...step.payload } as PipelineNode;
    expect(nodeLabel(node)).toBe(step.kind);
    expect(parameterLine(node)).toBe(step.parameters);
  }
});
