import type { DatasetEntry, PipelineNode, PipelineState } from "@/api/queries";
import type { Tab } from "@/shell/tabs";

/** The inspector's frame. What fills it per selection is #47; what it must do
 * here is be context-sensitive - a different heading and a different set of
 * facts depending on what the active tab shows - and collapse. */

interface Props {
  tab: Tab | undefined;
  datasets: DatasetEntry[] | undefined;
  nodes: PipelineNode[] | undefined;
  state: PipelineState | undefined;
  collapsed: boolean;
}

function Kv({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="kv">
      <b>{label}</b>
      <span>{value}</span>
    </div>
  );
}

function short(hash: string): string {
  return hash.length > 20 ? `${hash.slice(0, 11)}…${hash.slice(-4)}` : hash;
}

export function Inspector({ tab, datasets, nodes, state, collapsed }: Props) {
  if (collapsed) return <aside className="insp rail" aria-label="Inspector" />;

  if (!tab) {
    return (
      <aside className="insp" aria-label="Inspector">
        <div className="ihead">
          <span className="ilabel">Nothing selected</span>
        </div>
        <div className="empty">Select a dataset or a node in the outline.</div>
      </aside>
    );
  }

  const version = datasets
    ?.flatMap((entry) => entry.versions.map((v) => ({ entry, v })))
    .find((pair) => pair.v.version_id === tab.id);
  const node = nodes?.find((candidate) => candidate.id === tab.id);

  return (
    <aside className="insp" aria-label="Inspector">
      <div className="ihead">
        <span className="ilabel">
          {version ? "Dataset version" : node ? "Node" : tab.kind === "experiment" ? "Experiment" : "Selection"}
        </span>
        <span style={{ fontWeight: 600, fontSize: 13 }}>
          {version ? `${version.entry.dataset.name} · v${version.v.version}` : tab.title}
        </span>
      </div>

      {version ? (
        <>
          <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
            <Kv label="Samples" value={version.v.n_samples} />
            <Kv label="Variables" value={version.v.n_variables} />
            <Kv label="Content hash" value={short(version.v.content_hash)} />
            <Kv label="Derived from" value={version.v.derived_from ? "v1" : "—"} />
            <Kv label="Created" value={version.v.created_at.slice(0, 10)} />
          </div>
          <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
            <div className="ilabel" style={{ padding: "2px 12px 4px" }}>
              Variable axis
            </div>
            <Kv label="Kind" value={version.v.axis.kind} />
            <Kv label="Unit" value={version.v.axis.unit ?? "—"} />
          </div>
          {version.v.source ? (
            <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
              <div className="ilabel" style={{ padding: "2px 12px 4px" }}>
                Source
              </div>
              <Kv label="File" value={version.v.source.filename} />
              <Kv
                label="Reader"
                value={`${version.v.source.reader} ${version.v.source.reader_version}`}
              />
              <Kv label="File hash" value={short(version.v.source.file_hash)} />
            </div>
          ) : null}
        </>
      ) : null}

      {node ? (
        <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
          <Kv label="Type" value={node.type} />
          {node.step?.kind ? <Kv label="Step" value={String(node.step.kind)} /> : null}
          <Kv label="Inputs" value={node.inputs.join(", ") || "—"} />
          <Kv label="Run state" value={state?.nodes[node.id]?.state ?? "unknown"} />
        </div>
      ) : null}

      {/* The parameter editors, the metrics table and the provenance record
          are #47. This frame is what they mount into. */}
      <div style={{ padding: "9px 12px", color: "var(--ink2)" }}>Provenance record</div>
    </aside>
  );
}
