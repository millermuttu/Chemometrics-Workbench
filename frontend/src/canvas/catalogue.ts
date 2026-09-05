/** The steps a canvas can add, in one place.
 *
 * Shared by the side list and by the menu that opens when a connector is
 * dropped on empty canvas: two ways in to the same set, so a kind added here
 * appears in both rather than in whichever one someone remembered.
 *
 * The parameters are the defaults `models.py` already carries, written out
 * rather than left implicit: what is sent is what the canvas shows, and a
 * default that changed in the schema should change the label beside it.
 * Editing them is the inspector's job once the node is saved.
 *
 * **Only kinds this build can actually run.** `models.py` defines sixteen;
 * these are the ones with a kernel behind them. PLS-DA has no kernel
 * (`executor.py` `_FITTED`), and of the splitters only k-fold and
 * leave-one-out execute - `train_test`, `repeated_kfold` and `external` raise
 * at run time. Offering them here would let the canvas build a pipeline that
 * looks fine and dies when it is run, which is a worse answer than a shorter
 * menu.
 */
import type { DraftStep } from "@/canvas/graph";

/** A step the canvas can create.
 *
 * `type` is wider than `DraftStep`'s, which knows only about the two kinds the
 * side list can draw as a draft chain. A node added from the menu is created
 * directly rather than drafted, so it may also be a split.
 */
export type CatalogueStep = Omit<Pick<DraftStep, "kind" | "type" | "parameters" | "payload">, "type"> & {
  type: DraftStep["type"] | "split";
};

/** The subset the side list can draft, which is everything but a split. */
export type DraftableStep = Pick<DraftStep, "kind" | "type" | "parameters" | "payload">;

export const STEPS: DraftableStep[] = [
  {
    kind: "SNV",
    type: "preprocess",
    parameters: "population statistics per row",
    payload: { step: { kind: "snv" } },
  },
  {
    kind: "MSC",
    type: "preprocess",
    parameters: "reference: mean",
    payload: { step: { kind: "msc", reference: "mean" } },
  },
  {
    kind: "SG d1 w11",
    type: "preprocess",
    parameters: "window 11 · poly 2 · deriv 1",
    payload: {
      step: { kind: "savgol", window_length: 11, polyorder: 2, deriv: 1 },
    },
  },
  {
    kind: "Mean centre",
    type: "preprocess",
    parameters: "column means",
    payload: { step: { kind: "mean_centre" } },
  },
  {
    kind: "Autoscale",
    type: "preprocess",
    parameters: "ddof 1",
    payload: { step: { kind: "autoscale", ddof: 1 } },
  },
  {
    kind: "PCA",
    type: "estimator",
    parameters: "5 components",
    payload: { spec: { kind: "pca", n_components: 5 } },
  },
];


export const STEP_MENU: CatalogueStep[] = [
  ...STEPS,
  {
    kind: "PLS 5 LV",
    type: "estimator",
    parameters: "5 components · fat",
    payload: { spec: { kind: "pls", n_components: 5, algorithm: "nipals", target: "fat" } },
  },
  {
    kind: "K-fold 10",
    type: "split",
    parameters: "10 folds · shuffle · seed 42",
    payload: { spec: { kind: "kfold", n_splits: 10, shuffle: true, seed: 42 } },
  },
];
