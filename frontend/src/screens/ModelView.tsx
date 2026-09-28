import { useModel, type ModelRow } from "@/api/queries";
import { CannotLoad } from "@/states/CannotLoad";

/** One saved model's record: what it is, what it scored, and where its file is (#219).
 *
 * **Nothing here is computed and nothing here is the model.** The fitted
 * parameters are in the artifact, which `docs/model-artifact.md` specifies and
 * which `zipfile` and NumPy open without this application. What this screen
 * holds is the registry row: `PROPOSAL.md` section 11 puts the reference in the
 * database and the contents in the project directory, and the two are not
 * mixed here either.
 *
 * A metric the model does not carry is an em dash, never a zero
 * (`metrics-and-validation.md` section 11).
 */

function metric(value: number | null | undefined, digits = 4): string {
  return value == null ? "—" : value.toFixed(digits);
}

function when(value: string): string {
  return value.replace("T", " ").slice(0, 16);
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

function Scored({ model }: { model: ModelRow }) {
  const m = model.metrics;
  const rows: [string, string][] = [
    ["RMSEC", metric(m.rmsec)],
    ["RMSECV", metric(m.rmsecv)],
    ["RMSEP", metric(m.rmsep)],
    ["R²", metric(m.r2)],
    ["Q²", metric(m.q2)],
    ["Accuracy", metric(m.accuracy, 3)],
  ];
  return (
    <>
      {rows.map(([label, value]) => (
        <div className="kv" key={label}>
          <b>{label}</b>
          <span className="mono" data-testid={`model-metric-${label}`}>
            {value}
          </span>
        </div>
      ))}
      {m.explained_variance != null ? (
        <Kv label="PC1" mono value={`${(m.explained_variance * 100).toFixed(1)}%`} />
      ) : null}
    </>
  );
}

export function ModelView({ modelId, title }: { modelId: string; title: string }) {
  const record = useModel(modelId);

  if (record.isError) return <CannotLoad error={record.error} />;
  if (!record.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading the model…
        </div>
      </div>
    );
  }

  const model = record.data;

  return (
    <div className="pane" data-testid="model-view">
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
          <span className="pill mono">{model.task}</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            {when(model.created_at)}
          </span>
        </div>
        <span className="pill mono" title={model.node_id}>
          {model.node_id}
        </span>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        <Section label="What it scored">
          <Scored model={model} />
        </Section>

        <Section label="Where it came from">
          <Kv label="Experiment" mono value={model.experiment_id} />
          <Kv label="Node" mono value={model.node_id} />
        </Section>

        <Section label="Its file">
          <Kv label="Artifact" mono value={model.artifact_path} />
          <Kv label="Content hash" mono value={model.artifact_hash} />
          {/* Said rather than assumed: the reason the artifact is a zip of
              JSON and .npy is that opening it needs nothing from here. */}
          <p style={{ margin: "6px 14px 2px", lineHeight: 1.45, color: "var(--ink2)" }}>
            The file sits in the project directory, beside this database. It is a
            zip archive holding a manifest and one NumPy array per fitted
            quantity, and it opens without this application.
          </p>
        </Section>
      </div>
    </div>
  );
}
