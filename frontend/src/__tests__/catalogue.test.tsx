/** #182: the drop menu's PLS entry models a column the dataset has, or is
 * not offered. It used to write `target: "fat"` into every PLS node. */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import {
  type CatalogueStep,
  GROUPS,
  grouped,
  NO_CLASS,
  NO_TARGET,
  STEPS,
  stepMenu,
} from "@/canvas/catalogue";
import { parameterLine } from "@/canvas/graph";
import { StepList } from "@/canvas/StepList";
import { nodeLabel } from "@/shell/Sidebar";

/** The kinds that can be added: a disabled entry is listed, not offered. */
const enabled = (menu: CatalogueStep[]) =>
  menu.filter((step) => !step.disabled).map((step) => step.kind);

describe("the PLS menu entry", () => {
  it("models the dataset's first target", () => {
    const pls = stepMenu(["moisture", "fat"]).find((step) => step.kind === "PLS 5 LV")!;
    expect(pls.payload.spec).toMatchObject({ kind: "pls", target: "moisture" });
    expect(pls.parameters).toBe("5 components · moisture");
  });

  it("offers PLS-DA only for a dataset with a two-valued column, on that column", () => {
    const plsda = stepMenu(["fat"], ["fat_class"]).find((step) => step.kind === "PLS-DA 5 LV")!;
    expect(plsda.payload.spec).toMatchObject({ kind: "plsda", class_column: "fat_class" });
    expect(enabled(stepMenu(["fat"]))).not.toContain("PLS-DA 5 LV");
  });

  it("is disabled, with the reason, when the dataset has nothing to model (#372)", () => {
    const kinds = enabled(stepMenu([]));
    expect(kinds).not.toContain("PLS 5 LV");
    expect(stepMenu([]).find((step) => step.kind === "PLS 5 LV")!.disabled).toBe(NO_TARGET);
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
  expect(enabled(stepMenu([]))).not.toContain("PCR 5 PC");
});

it("offers SIMCA beside PLS-DA on a class column, labelled alike (#275)", () => {
  const simca = stepMenu([], ["grade"]).find((step) => step.kind === "SIMCA 3 PC")!;
  expect(simca.payload.spec).toEqual({ kind: "simca", n_components: 3, class_column: "grade" });
  const node = { id: "simca", type: "estimator", inputs: ["snv"], ...simca.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("SIMCA 3 PC · grade");
  expect(parameterLine(node)).toBe(simca.parameters);
  expect(enabled(stepMenu(["fat"]))).not.toContain("SIMCA 3 PC");
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

it("offers an SVM beside kNN on a class column, labelled alike (#338)", () => {
  const svm = stepMenu([], ["grade"]).find((step) => step.kind === "SVM rbf 5 PC")!;
  expect(svm.payload.spec).toEqual({ kind: "svm", kernel: "rbf", C: 1, n_components: 5, class_column: "grade" });
  const node = { id: "svm", type: "estimator", inputs: ["snv"], ...svm.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("SVM rbf 5 PC · grade");
  expect(parameterLine(node)).toBe(svm.parameters);
  expect(enabled(stepMenu(["fat"]))).not.toContain("SVM rbf 5 PC");
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

it("puts every step under a heading, the models under Regression and Classification (#370)", () => {
  const menu = stepMenu(["fat"], ["fat_class"]);
  const headings = Object.fromEntries(grouped(menu).map(([group, steps]) => [group, steps.map((s) => s.kind)]));
  expect(Object.keys(headings)).toEqual([...GROUPS]);
  expect(headings.Regression).toEqual([
    "PLS 5 LV",
    "PCR 5 PC",
    "Ridge α1",
    "Elastic Net α1",
    "Lasso α1",
    "MLR",
    "SVR rbf 5 PC",
  ]);
  expect(headings.Classification).toEqual(["PLS-DA 5 LV", "LDA 5 PC", "kNN k5", "SVM rbf 5 PC", "SIMCA 3 PC"]);
  expect(headings.Validation).toEqual(["K-fold 10", "Train/test 25%"]);
  expect(headings.Explore).toEqual(["PCA"]);
  // The menu is already in heading order, so grouping it moves nothing.
  expect(grouped(menu).flatMap(([, steps]) => steps)).toEqual(menu);
  // Without a target the heading stays, its models disabled (#372).
  expect(grouped(stepMenu([])).map(([group]) => group)).toContain("Regression");
});

it("draws the Step list's options under one optgroup per heading (#370)", () => {
  const markup = renderToStaticMarkup(
    <StepList
      menu={stepMenu(["fat"], ["fat_class"])}
      steps={[]}
      onChange={() => {}}
      onValidate={() => {}}
      onSave={() => {}}
      saving={false}
      validation={null}
      edited={false}
    />,
  );
  const labels = [...markup.matchAll(/<optgroup label="([^"]+)"/g)].map((match) => match[1]);
  expect(labels).toEqual([...GROUPS]);
  const regression = markup.split('<optgroup label="Regression">')[1].split("</optgroup>")[0];
  expect(regression).toContain('value="PLS 5 LV"');
  expect(regression).toContain('value="PCR 5 PC"');
});

it("disables every model the dataset has no column for, naming the column it needs (#372)", () => {
  const bare = stepMenu([]);
  const models = bare.filter((step) => step.group === "Regression" || step.group === "Classification");
  expect(models.map((step) => step.kind)).toEqual(
    stepMenu(["fat"], ["fat_class"])
      .filter((step) => step.group === "Regression" || step.group === "Classification")
      .map((step) => step.kind),
  );
  for (const step of models) {
    expect(step.disabled).toBe(step.group === "Regression" ? NO_TARGET : NO_CLASS);
  }
  expect(bare.filter((step) => step.disabled).every((step) => models.includes(step))).toBe(true);
  // A target alone enables regression and leaves the classifiers disabled.
  expect(enabled(stepMenu(["fat"]))).toContain("PLS 5 LV");
  expect(stepMenu(["fat"]).find((step) => step.kind === "LDA 5 PC")!.disabled).toBe(NO_CLASS);
});

it("draws a disabled model as a disabled option, and its reason under the list (#372)", () => {
  const markup = renderToStaticMarkup(
    <StepList
      menu={stepMenu([])}
      onOpenDataset={() => {}}
      steps={[]}
      onChange={() => {}}
      onValidate={() => {}}
      onSave={() => {}}
      saving={false}
      validation={null}
      edited={false}
    />,
  );
  expect(markup).toMatch(/<option value="PLS 5 LV" disabled="" title="No target column/);
  expect(markup).toContain(NO_TARGET);
  expect(markup).toContain(NO_CLASS);
  expect(markup).toContain("Open Dataset view");
});

it("offers Ridge beside PLS when the dataset has a target, labelled alike (#374)", () => {
  const ridge = stepMenu(["fat"]).find((step) => step.kind === "Ridge α1")!;
  expect(ridge.group).toBe("Regression");
  expect(ridge.payload.spec).toEqual({ kind: "ridge", alpha: 1, target: "fat" });
  const node = { id: "ridge", type: "estimator", inputs: ["snv"], ...ridge.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("Ridge α1 · fat");
  expect(parameterLine(node)).toBe(ridge.parameters);
  expect(stepMenu([]).find((step) => step.kind === "Ridge α1")!.disabled).toBe(NO_TARGET);
});

it("offers Elastic Net and the Lasso as one estimator, named by l1_ratio (#375)", () => {
  const menu = stepMenu(["fat"]);
  const enet = menu.find((step) => step.kind === "Elastic Net α1")!;
  const lasso = menu.find((step) => step.kind === "Lasso α1")!;
  expect(enet.payload.spec).toEqual({ kind: "elastic_net", alpha: 1, l1_ratio: 0.5, target: "fat" });
  expect(lasso.payload.spec).toEqual({ kind: "elastic_net", alpha: 1, l1_ratio: 1, target: "fat" });
  for (const step of [enet, lasso]) {
    const node = { id: "e", type: "estimator", inputs: ["snv"], ...step.payload } as PipelineNode;
    expect(nodeLabel(node)).toBe(`${step.kind} · fat`);
    expect(parameterLine(node)).toBe(step.parameters);
  }
});

it("offers MLR beside PLS when the dataset has a target, labelled alike (#376)", () => {
  const mlr = stepMenu(["fat"]).find((step) => step.kind === "MLR")!;
  expect(mlr.payload.spec).toEqual({ kind: "mlr", target: "fat" });
  const node = { id: "mlr", type: "estimator", inputs: ["pick"], ...mlr.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("MLR · fat");
  expect(parameterLine(node)).toBe(mlr.parameters);
});

it("offers an SVR beside PLS when the dataset has a target, labelled alike (#377)", () => {
  const svr = stepMenu(["fat"]).find((step) => step.kind === "SVR rbf 5 PC")!;
  expect(svr.payload.spec).toEqual({
    kind: "svr",
    kernel: "rbf",
    C: 1,
    epsilon: 0.1,
    n_components: 5,
    target: "fat",
  });
  const node = { id: "svr", type: "estimator", inputs: ["snv"], ...svr.payload } as PipelineNode;
  expect(nodeLabel(node)).toBe("SVR rbf 5 PC · fat");
  expect(parameterLine(node)).toBe(svr.parameters);
});
