import { useExperimentRecord, type ExperimentRecord } from "@/api/queries";
import { CannotLoad } from "@/states/CannotLoad";
import { diffPipelines, type NodeDiff } from "@/lineage/diff";
import { nodeLabel } from "@/shell/Sidebar";
import { parameterLine } from "@/canvas/graph";

/** Two runs side by side, and exactly which nodes they differ by (#215).
 *
 * `PROPOSAL.md` section 8.3: *a model comparison view that shows exactly which
 * steps differ between two models is the single feature most likely to make a
 * researcher prefer this tool to a notebook - and it is free once the model
 * above is correct.* It is free because an `Experiment` carries its pipeline by
 * value, so this compares two recipes rather than reconstructing them.
 *
 * **Nothing here is computed.** The diff is a comparison of two snapshots and
 * every figure is one the record carries; a metric one run has and the other
 * does not is an em dash, per `metrics-and-validation.md` section 11.
 */

const MARK: Record<NodeDiff["change"], string> = {
  added: "+",
  removed: "−",
  changed: "≠",
  unchanged: "",
};

const COLOUR: Record<NodeDiff["change"], string | undefined> = {
  added: "var(--ok)",
  removed: "var(--fail)",
  changed: "var(--stale)",
  unchanged: undefined,
};

function metric(value: number | null | undefined, digits = 4): string {
  return value == null ? "—" : value.toFixed(digits);
}

function when(value: string | null): string {
  return value ? value.replace("T", " ").slice(0, 16) : "—";
}

/** One side of a node row: what that run ran, or nothing when it had no such
 * node. An absent side is said rather than left blank, because a blank cell
 * and "this run did not have this step" are different claims. */
function Side({ node, missing }: { node: NodeDiff["left"]; missing: string }) {
  if (!node) {
    return (
      <span className="mono" style={{ color: "var(--ink3)" }}>
        {missing}
      </span>
    );
  }
  return (
    <span className="mono">
      {nodeLabel(node)}
      <span style={{ color: "var(--ink3)" }}> · {parameterLine(node)}</span>
    </span>
  );
}

function Scores({ left, right }: { left: ExperimentRecord; right: ExperimentRecord }) {
  const rows: [string, string, string][] = [
    ["RMSEC", metric(left.metrics?.rmsec), metric(right.metrics?.rmsec)],
    ["RMSECV", metric(left.metrics?.rmsecv), metric(right.metrics?.rmsecv)],
    ["RMSEP", metric(left.metrics?.rmsep), metric(right.metrics?.rmsep)],
    ["R²", metric(left.metrics?.r2), metric(right.metrics?.r2)],
    ["Q²", metric(left.metrics?.q2), metric(right.metrics?.q2)],
    ["Accuracy", metric(left.metrics?.accuracy, 3), metric(right.metrics?.accuracy, 3)],
  ];
  return (
    <table data-testid="lineage-scores">
      <thead>
        <tr>
          <th style={{ width: 140 }}>Scored</th>
          <th className="n">Left</th>
          <th className="n">Right</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, a, b]) => (
          <tr key={label}>
            <td className="mono" style={{ color: "var(--ink)" }}>
              {label}
            </td>
            <td className="n">{a}</td>
            <td className="n">{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LineageView({
  left: leftId,
  right: rightId,
}: {
  left: string;
  right: string;
}) {
  const left = useExperimentRecord(leftId);
  const right = useExperimentRecord(rightId);

  const failed = left.error ?? right.error;
  if (failed) return <CannotLoad error={failed} />;
  if (!left.data || !right.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading both runs…
        </div>
      </div>
    );
  }

  const diff = diffPipelines(left.data.pipeline_snapshot.nodes, right.data.pipeline_snapshot.nodes);
  // Two runs of the same recipe hash the same, which is the cheap version of
  // the comparison below and has to agree with it.
  const sameDataset = left.data.dataset_content_hash === right.data.dataset_content_hash;

  return (
    <div className="pane" data-testid="lineage-view">
      <div
        style={{
          height: 40,
          flex: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 14px",
          borderBottom: "1px solid var(--rule2)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span style={{ fontWeight: 600, fontSize: 13.5 }}>Two runs compared</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            {when(left.data.started_at)} against {when(right.data.started_at)}
          </span>
        </div>
        <span
          className="pill mono"
          data-testid="lineage-summary"
          style={diff.identical ? undefined : { color: "var(--stale)", borderColor: "var(--stale)" }}
        >
          {diff.identical
            ? "the same recipe"
            : `${diff.differing} node${diff.differing === 1 ? "" : "s"} differ`}
        </span>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {!sameDataset ? (
          // Two runs on different data are comparable only in the weakest
          // sense, and saying so is cheaper than a reader noticing later.
          <p
            role="note"
            data-testid="lineage-different-datasets"
            style={{ margin: 12, padding: "8px 12px", border: "1px solid var(--stale)", borderRadius: 3, lineHeight: 1.4 }}
          >
            These runs used different datasets, so their metrics are not
            comparable — only their recipes are.
          </p>
        ) : null}

        <section style={{ borderBottom: "1px solid var(--rule)", padding: "8px 0" }}>
          <div className="ilabel" style={{ padding: "2px 14px 6px" }}>
            What differs
          </div>
          <table data-testid="lineage-nodes">
            <thead>
              <tr>
                <th style={{ width: 28 }} />
                <th style={{ width: 110 }}>Node</th>
                <th>Left</th>
                <th>Right</th>
              </tr>
            </thead>
            <tbody>
              {diff.nodes.map((node) => (
                <tr
                  key={node.id}
                  data-testid={`lineage-node-${node.change}`}
                  data-node={node.id}
                  style={{ opacity: node.change === "unchanged" ? 0.55 : 1 }}
                >
                  <td className="mono" style={{ color: COLOUR[node.change], fontWeight: 600 }}>
                    {MARK[node.change]}
                  </td>
                  <td className="mono" style={{ color: "var(--ink)" }}>
                    {node.id}
                    {node.fields.length > 0 ? (
                      <span style={{ color: "var(--stale)" }}> · {node.fields.join(", ")}</span>
                    ) : null}
                  </td>
                  <td>
                    <Side node={node.left} missing="not in this run" />
                  </td>
                  <td>
                    <Side node={node.right} missing="not in this run" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section style={{ borderBottom: "1px solid var(--rule)", padding: "8px 0" }}>
          <div className="ilabel" style={{ padding: "2px 14px 6px" }}>
            What each scored
          </div>
          <Scores left={left.data} right={right.data} />
        </section>
      </div>
    </div>
  );
}
