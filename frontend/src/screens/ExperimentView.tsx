import { useExperimentRecord, useExperiments, type ExperimentRecord } from "@/api/queries";
import { CannotLoad } from "@/states/CannotLoad";
import { nodeLabel } from "@/shell/Sidebar";
import { parameterLine } from "@/canvas/graph";

/** One run's record: what it ran, against what, what it scored, and where (#209).
 *
 * `PROPOSAL.md` section 8.2 lists what every experiment must capture, and this
 * is that list rendered. **Nothing here is computed.** The pipeline is the
 * snapshot the experiment carries by value - not the pipeline as it is now,
 * which is the whole point of snapshotting it - and every figure is one the
 * record holds. A metric it does not hold is an em dash, never a zero
 * (`metrics-and-validation.md` section 11).
 */

function when(value: string | null): string {
  if (!value) return "—";
  // The record's own ISO string, to the minute: a run is identified by its
  // recipe and its data, and the clock is context rather than identity.
  return value.replace("T", " ").slice(0, 16);
}

function metric(value: number | null | undefined, digits = 4): string {
  return value == null ? "—" : value.toFixed(digits);
}

function Kv({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="kv">
      <b>{label}</b>
      <span className={mono ? "mono" : undefined} title={value}>
        {value}
      </span>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section style={{ borderBottom: "1px solid var(--rule)", padding: "8px 0" }}>
      <div className="ilabel" style={{ padding: "2px 14px 6px" }}>
        {label}
      </div>
      {children}
    </section>
  );
}

function Metrics({ record }: { record: ExperimentRecord }) {
  const m = record.metrics;
  if (!m) {
    return (
      <div className="empty" style={{ padding: "0 14px 6px" }}>
        This run recorded no metrics.
      </div>
    );
  }
  const rows: [string, string][] = [
    ["RMSEC", metric(m.rmsec)],
    ["RMSECV", metric(m.rmsecv)],
    ["RMSEP", metric(m.rmsep)],
    ["R²", metric(m.r2)],
    ["Q²", metric(m.q2)],
    ["Bias", metric(m.bias)],
    ["Accuracy", metric(m.accuracy, 3)],
  ];
  const variance = m.explained_variance;
  return (
    <>
      {rows.map(([label, value]) => (
        <div className="kv" key={label}>
          <b>{label}</b>
          <span className="mono" data-testid={`run-metric-${label}`}>
            {value}
          </span>
        </div>
      ))}
      {variance?.length ? (
        <Kv
          label="Explained variance"
          mono
          value={variance.map((share) => `${(share * 100).toFixed(1)}%`).join(" · ")}
        />
      ) : null}
    </>
  );
}

/** Pick another run to compare this one with (#215).
 *
 * A run names the others and opening one pairs them, which is the cheapest
 * pairing that exists: the history is already loaded, and a run knows its own
 * id. The canvas's node comparison (#51) is a different pairing of a different
 * thing and is left alone.
 */
function CompareWith({
  experimentId,
  onCompareRuns,
}: {
  experimentId: string;
  onCompareRuns: (left: string, right: string) => void;
}) {
  const runs = useExperiments();
  const others = (runs.data ?? []).filter((row) => row.experiment_id !== experimentId);
  if (others.length === 0) return null;
  return (
    <select
      aria-label="Compare with another run"
      data-testid="compare-with"
      value=""
      onChange={(event) => {
        if (event.target.value) onCompareRuns(experimentId, event.target.value);
      }}
    >
      <option value="">Compare with…</option>
      {others.map((row) => (
        <option key={row.experiment_id} value={row.experiment_id}>
          {when(row.started_at)} · {row.status} · {row.n_nodes} nodes
        </option>
      ))}
    </select>
  );
}

export function ExperimentView({
  experimentId,
  title,
  onCompareRuns,
}: {
  experimentId: string;
  title: string;
  onCompareRuns: (left: string, right: string) => void;
}) {
  const record = useExperimentRecord(experimentId);

  if (record.isError) return <CannotLoad error={record.error} />;
  if (!record.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading the run…
        </div>
      </div>
    );
  }

  const run = record.data;
  const nodes = run.pipeline_snapshot.nodes;
  const split = run.resolved_splits[0];
  const failed = run.status === "failed";

  return (
    <div className="pane" data-testid="experiment-view">
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
          <span style={{ fontWeight: 600, fontSize: 13.5 }}>{title}</span>
          <span
            className="pill mono"
            style={failed ? { color: "var(--fail)", borderColor: "var(--fail)" } : undefined}
          >
            {run.status}
          </span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            {when(run.started_at)}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <CompareWith experimentId={experimentId} onCompareRuns={onCompareRuns} />
          <span className="pill mono" title={run.pipeline_snapshot.pipeline_id}>
            {nodes.length} nodes
          </span>
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {/* A failed experiment is a result, and the cause is what it holds. */}
        {run.error ? (
          <div
            role="alert"
            data-testid="run-error"
            style={{
              margin: 12,
              padding: "10px 12px",
              borderRadius: 3,
              border: "1px solid var(--fail)",
              background: "var(--failSoft)",
              lineHeight: 1.4,
            }}
          >
            {run.error}
          </div>
        ) : null}

        <Section label="What it ran">
          <table data-testid="run-pipeline">
            <thead>
              <tr>
                <th style={{ width: 120 }}>Node</th>
                <th style={{ width: 90 }}>Type</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {nodes.map((node) => (
                <tr key={node.id}>
                  <td className="mono" style={{ color: "var(--ink)" }}>
                    {nodeLabel(node)}
                  </td>
                  <td className="mono">{node.type}</td>
                  <td className="mono">{parameterLine(node)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section label="Against what">
          <Kv label="Dataset version" mono value={run.dataset_version_id} />
          <Kv label="Content hash" mono value={run.dataset_content_hash} />
          <Kv label="Pipeline hash" mono value={run.pipeline_snapshot.pipeline_id} />
          {split ? (
            <Kv
              label="Split"
              mono
              value={`${split.node_id} · ${split.test_indices.length} fold${
                split.test_indices.length === 1 ? "" : "s"
              } · ${split.test_indices.reduce((total, fold) => total + fold.length, 0)} held out`}
            />
          ) : null}
        </Section>

        <Section label="What it scored">
          <Metrics record={run} />
        </Section>

        <Section label="Where">
          <Kv label="Started" value={when(run.started_at)} />
          <Kv label="Finished" value={when(run.finished_at)} />
          {run.environment ? (
            <>
              <Kv label="Application" mono value={run.environment.app_version} />
              <Kv label="Python" mono value={run.environment.python_version} />
              <Kv label="Platform" mono value={run.environment.platform} />
              {Object.entries(run.environment.packages).map(([name, version]) => (
                <Kv key={name} label={name} mono value={version} />
              ))}
            </>
          ) : (
            // `models.py` refuses a succeeded experiment with no environment,
            // so this is a cancelled or failed run rather than a gap.
            <div className="empty" style={{ padding: "0 14px 6px" }}>
              This run recorded no environment, which only a run that did not
              succeed does.
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}
