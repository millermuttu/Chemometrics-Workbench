/** #182: the drop menu's PLS entry models a column the dataset has, or is
 * not offered. It used to write `target: "fat"` into every PLS node. */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import { STEPS, stepMenu } from "@/canvas/catalogue";
import { parameterLine } from "@/canvas/graph";
import { StepList } from "@/canvas/StepList";
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

it("offers PCR beside PLS when the dataset has a target, labelled alike (#272)", () => {
  const pcr = stepMenu(["fat"]).find((step) => step.kind === "PCR 5 PC")!;
  expect(pcr.payload.spec).toEqual({ kind: "pcr", n_components: 5, target: "fat" });
  const node = { id: "pcr", type: "estimator", inputs: ["centre"], ...pcr.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("PCR 5 PC · fat");
  expect(parameterLine(node)).toBe(pcr.parameters);
  expect(stepMenu([]).map((step) => step.kind)).not.toContain("PCR 5 PC");
});

it("offers SIMCA beside PLS-DA on a class column, labelled alike (#275)", () => {
  const simca = stepMenu([], ["grade"]).find((step) => step.kind === "SIMCA 3 PC")!;
  expect(simca.payload.spec).toEqual({ kind: "simca", n_components: 3, class_column: "grade" });
  const node = { id: "simca", type: "estimator", inputs: ["snv"], ...simca.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("SIMCA 3 PC · grade");
  expect(parameterLine(node)).toBe(simca.parameters);
  expect(stepMenu(["fat"]).map((step) => step.kind)).not.toContain("SIMCA 3 PC");
});

it("offers LDA beside PLS-DA on a class column, labelled alike (#276)", () => {
  const lda = stepMenu([], ["grade"]).find((step) => step.kind === "LDA 5 PC")!;
  expect(lda.payload.spec).toEqual({ kind: "lda", n_components: 5, class_column: "grade" });
  const node = { id: "lda", type: "estimator", inputs: ["snv"], ...lda.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("LDA 5 PC · grade");
  expect(parameterLine(node)).toBe(lda.parameters);
});

it("offers kNN beside PLS-DA on a class column, labelled alike (#277)", () => {
  const knn = stepMenu([], ["grade"]).find((step) => step.kind === "kNN k5")!;
  expect(knn.payload.spec).toEqual({ kind: "knn", k: 5, n_components: 5, class_column: "grade" });
  const node = { id: "knn", type: "estimator", inputs: ["snv"], ...knn.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("kNN k5 · grade");
  expect(parameterLine(node)).toBe(knn.parameters);
});

it("offers in the Step list what the port menu offers, splits and estimators included (#337)", () => {
  const menu = stepMenu(["fat"], ["fat_class"]);
  const markup = renderToStaticMarkup(
    <StepList
      menu={menu}
      steps={[]}
      onChange={() => {}}
      onValidate={() => {}}
      onSave={() => {}}
      saving={false}
      validation={null}
      edited={false}
    />,
  );
  const options = [...markup.matchAll(/<option value="([^"]+)"/g)].map((match) => match[1]);
  expect(options).toEqual(menu.map((step) => step.kind));
  expect(options).toEqual(expect.arrayContaining(["K-fold 10", "Train/test 25%", "PLS-DA 5 LV"]));
});
