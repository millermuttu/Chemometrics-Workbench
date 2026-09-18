import type { PcaPayload } from "@/api/queries";

/** The inspector's headline numbers for one estimator node, from its own result.
 *
 * These used to come from the experiment, whose metrics are the *last*
 * estimator's in topological order, so every estimator in a branching pipeline
 * showed the same numbers - and a PLS node was labelled with a PCA's (#175).
 * `null` renders as an em dash: a metric that could not be computed is absent,
 * never zero. */
export function nodeMetrics(result: PcaPayload | undefined): Record<string, number | null> | undefined {
  if (!result) return undefined;
  if (result.task === "classification") {
    const metric = (name: string) => result.metrics?.[name] ?? null;
    return {
      Accuracy: metric("accuracy"),
      "Accuracy (CV)": metric("accuracy_cv"),
      Sensitivity: metric("sensitivity"),
      Specificity: metric("specificity"),
      components: result.n_components,
    };
  }
  if (result.task === "regression") {
    const metric = (name: string) => result.metrics?.[name] ?? null;
    return {
      RMSEC: metric("rmsec"),
      "R²": metric("r2"),
      RMSECV: metric("rmsecv"),
      "Q²": metric("q2"),
      components: result.n_components,
    };
  }
  return {
    "PC1 variance": result.explained_variance_ratio[0] ?? null,
    [`PC1-${result.n_components} cumulative`]: result.cumulative_explained_variance.at(-1) ?? null,
    components: result.n_components,
  };
}
