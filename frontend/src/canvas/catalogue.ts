/** The steps a canvas can add, in one place.
 *
 * Shared by the side list and by the menu that opens when a connector is
 * dropped on empty canvas: two ways in to the same set, so a kind added here
 * appears in both rather than in whichever one someone remembered. Until #337
 * the list took only `STEPS`, so splits and every estimator but PCA were
 * reachable from a port alone; both now take `stepMenu`.
 *
 * The parameters are the defaults `models.py` already carries, written out
 * rather than left implicit: what is sent is what the canvas shows, and a
 * default that changed in the schema should change the label beside it.
 * Editing them is the inspector's job once the node is saved.
 *
 * **Only kinds this build can actually run.** `models.py` defines sixteen;
 * these are the ones with a kernel behind them. Every estimator has one since
 * #185; of the splitters k-fold, leave-one-out and train/test execute -
 * `repeated_kfold` and `external` raise at run time. Offering those here
 * would let the canvas build a pipeline that looks fine and dies when it is
 * run, which is a worse answer than a shorter menu.
 */
import type { DraftStep } from "@/canvas/graph";

/** The headings both menus show, in the order they show them (#370). One
 * flat list put the models between the smoothers and the splits, where they
 * were hard to notice; every entry now says which heading it sits under. */
export const GROUPS = [
  "Preprocess",
  "Derivative and smoothing",
  "Scaling",
  "Explore",
  "Regression",
  "Classification",
  "Validation",
] as const;

export type StepGroup = (typeof GROUPS)[number];

/** A step the canvas can create, from the list or from the menu. */
export type CatalogueStep = Pick<DraftStep, "kind" | "type" | "parameters" | "payload"> & {
  group: StepGroup;
};

/** The menu under its headings, in `GROUPS` order; an empty group is left out. */
export function grouped(menu: CatalogueStep[]): [StepGroup, CatalogueStep[]][] {
  return GROUPS.map((group) => [group, menu.filter((step) => step.group === group)] as [
    StepGroup,
    CatalogueStep[],
  ]).filter(([, steps]) => steps.length > 0);
}

export const STEPS: CatalogueStep[] = [
  {
    kind: "SNV",
    group: "Preprocess",
    type: "preprocess",
    parameters: "population statistics per row",
    payload: { step: { kind: "snv" } },
  },
  {
    kind: "MSC",
    group: "Preprocess",
    type: "preprocess",
    parameters: "reference: mean",
    payload: { step: { kind: "msc", reference: "mean" } },
  },
  {
    kind: "Normalise",
    group: "Preprocess",
    type: "preprocess",
    parameters: "l2 norm per row",
    payload: { step: { kind: "normalise", norm: "l2" } },
  },
  {
    kind: "SG d1 w11",
    group: "Derivative and smoothing",
    type: "preprocess",
    parameters: "window 11 · poly 2 · deriv 1",
    payload: {
      step: { kind: "savgol", window_length: 11, polyorder: 2, deriv: 1 },
    },
  },
  // #271: smoothers without a derivative. The defaults are the parity
  // fixture's (smoothing-and-baselines.md section 10.4).
  {
    kind: "Moving avg w5",
    group: "Derivative and smoothing",
    type: "preprocess",
    parameters: "window 5",
    payload: { step: { kind: "moving_average", window_length: 5 } },
  },
  {
    kind: "Median w5",
    group: "Derivative and smoothing",
    type: "preprocess",
    parameters: "window 5",
    payload: { step: { kind: "median", window_length: 5 } },
  },
  {
    kind: "Gaussian σ1.5",
    group: "Derivative and smoothing",
    type: "preprocess",
    parameters: "sigma 1.5",
    payload: { step: { kind: "gaussian", sigma: 1.5 } },
  },
  {
    kind: "Whittaker λ100",
    group: "Derivative and smoothing",
    type: "preprocess",
    parameters: "lambda 100",
    payload: { step: { kind: "whittaker", lam: 100 } },
  },
  {
    kind: "Mean centre",
    group: "Scaling",
    type: "preprocess",
    parameters: "column means",
    payload: { step: { kind: "mean_centre" } },
  },
  {
    kind: "Autoscale",
    group: "Scaling",
    type: "preprocess",
    parameters: "ddof 1",
    payload: { step: { kind: "autoscale", ddof: 1 } },
  },
  {
    kind: "PCA",
    group: "Explore",
    type: "estimator",
    parameters: "5 components",
    payload: { spec: { kind: "pca", n_components: 5 } },
  },
];


/** The drop menu, for a dataset with these target columns.
 *
 * A function rather than a constant because a PLS node has to model
 * *something*: this used to write `target: "fat"` - Tecator's column - into
 * every PLS node, so on any other dataset the node validated, ran and failed
 * by name (#182). The first target is the default and the inspector offers
 * the rest; a dataset with no targets is offered no PLS at all.
 */
export function stepMenu(targets: string[], classColumns: string[] = []): CatalogueStep[] {
  const [target] = targets;
  const [classColumn] = classColumns;
  return [
    ...STEPS,
    ...(target
      ? [
          {
            kind: "PLS 5 LV",
            group: "Regression" as const,
            type: "estimator" as const,
            parameters: `5 components · ${target}`,
            payload: { spec: { kind: "pls", n_components: 5, algorithm: "nipals", target } },
          },
          {
            kind: "PCR 5 PC",
            group: "Regression" as const,
            type: "estimator" as const,
            parameters: `5 components · ${target}`,
            payload: { spec: { kind: "pcr", n_components: 5, target } },
          },
        ]
      : []),
    // Offered when the dataset has a metadata column with two or more values
    // (#185, #274, pls-da.md section 2), and it models the first.
    ...(classColumn
      ? [
          {
            kind: "PLS-DA 5 LV",
            group: "Classification" as const,
            type: "estimator" as const,
            parameters: `5 components · ${classColumn}`,
            payload: {
              spec: { kind: "plsda", n_components: 5, algorithm: "nipals", class_column: classColumn },
            },
          },
          {
            kind: "LDA 5 PC",
            group: "Classification" as const,
            type: "estimator" as const,
            parameters: `5 components · ${classColumn}`,
            payload: { spec: { kind: "lda", n_components: 5, class_column: classColumn } },
          },
          {
            kind: "kNN k5",
            group: "Classification" as const,
            type: "estimator" as const,
            parameters: `k 5 · 5 components · ${classColumn}`,
            payload: { spec: { kind: "knn", k: 5, n_components: 5, class_column: classColumn } },
          },
          {
            kind: "SVM rbf 5 PC",
            group: "Classification" as const,
            type: "estimator" as const,
            parameters: `rbf · C 1 · 5 components · ${classColumn}`,
            payload: {
              spec: { kind: "svm", kernel: "rbf", C: 1, n_components: 5, class_column: classColumn },
            },
          },
          {
            kind: "SIMCA 3 PC",
            group: "Classification" as const,
            type: "estimator" as const,
            parameters: `3 components · ${classColumn}`,
            payload: { spec: { kind: "simca", n_components: 3, class_column: classColumn } },
          },
        ]
      : []),
    {
      kind: "K-fold 10",
      group: "Validation" as const,
      type: "split",
      parameters: "10 folds · shuffle · seed 42",
      payload: { spec: { kind: "kfold", n_splits: 10, shuffle: true, seed: 42 } },
    },
    {
      // One hold-out (#183): RMSEP on the held-out rows, no RMSECV.
      kind: "Train/test 25%",
      group: "Validation" as const,
      type: "split",
      parameters: "25% held out · seed 42",
      payload: { spec: { kind: "train_test", test_size: 0.25, seed: 42 } },
    },
  ];
}
