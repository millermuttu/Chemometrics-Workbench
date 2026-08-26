import { useEffect, useState } from "react";

import type { DatasetEntry, Pipeline, PipelineNode, PipelineState, Project } from "@/api/queries";
import { useStepSchema } from "@/api/queries";
import { ParameterForm } from "@/inspector/ParameterForm";
import { Provenance } from "@/inspector/Provenance";
import { specFor } from "@/inspector/schema";
import type { Tab } from "@/shell/tabs";

/** The right sidebar, and in Phase 1.1 the only place a parameter is edited.
 *
 * Context-sensitive: a dataset shows its metadata and its source, a
 * preprocessing node shows a typed form built from the schema, an estimator
 * shows its metrics. Provenance sits at the foot of all of them, collapsed.
 */

interface Props {
  tab: Tab | undefined;
  project: Project | undefined;
  datasets: DatasetEntry[] | undefined;
  pipeline: Pipeline | undefined;
  state: PipelineState | undefined;
  metrics: Record<string, number | null> | undefined;
  collapsed: boolean;
  onEdit: (nodeId: string) => void;
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
  return hash.length > 22 ? `${hash.slice(0, 11)}…${hash.slice(-4)}` : hash;
}

/** The step's current values, as strings, so the form has one representation. */
function valuesOf(node: PipelineNode): Record<string, string> {
  const step = (node.step ?? node.spec ?? {}) as Record<string, unknown>;
  return Object.fromEntries(
    Object.entries(step)
      .filter(([name]) => name !== "kind")
      .map(([name, value]) => [name, value === null ? "" : String(value)]),
  );
}

export function Inspector({
  tab,
  project,
  datasets,
  pipeline,
  state,
  metrics,
  collapsed,
  onEdit,
}: Props) {
  const schema = useStepSchema();
  const node = pipeline?.nodes.find((candidate) => candidate.id === tab?.id);
  const [values, setValues] = useState<Record<string, string>>({});

  // A different node is a different form. Reset to what the pipeline says
  // rather than carrying the last node's numbers across.
  useEffect(() => setValues(node ? valuesOf(node) : {}), [node]);

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
  const status = node ? state?.nodes[node.id] : undefined;
  const spec = node?.step ? specFor(schema.data, String(node.step.kind)) : undefined;

  return (
    <aside className="insp" aria-label="Inspector" style={{ overflowY: "auto" }}>
      <div className="ihead">
        <span className="ilabel">
          {version ? "Dataset version" : node ? node.type : tab.kind}
        </span>
        <span style={{ fontWeight: 600, fontSize: 13 }}>
          {version ? `${version.entry.dataset.name} · v${version.v.version}` : tab.title}
        </span>
        {status ? (
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: status.state === "stale" ? "var(--stale)" : status.state === "failed" ? "var(--fail)" : "var(--ink3)",
            }}
          >
            {status.state}
            {status.reason ? ` · ${status.reason}` : ""}
          </span>
        ) : null}
      </div>

      {version ? (
        <>
          <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
            <Kv label="Samples" value={version.v.n_samples} />
            <Kv label="Variables" value={version.v.n_variables} />
            <Kv label="Targets" value={Object.keys(version.v.targets).join(", ") || "—"} />
            <Kv label="Created" value={version.v.created_at.slice(0, 10)} />
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
            </div>
          ) : null}
        </>
      ) : null}

      {spec ? (
        <ParameterForm
          spec={spec}
          values={values}
          onChange={setValues}
          onApply={() => onEdit(node!.id)}
        />
      ) : node ? (
        <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
          <Kv label="Type" value={node.type} />
          <Kv label="Inputs" value={node.inputs.join(", ") || "—"} />
          {node.spec
            ? Object.entries(node.spec)
                .filter(([name]) => name !== "kind")
                .map(([name, value]) => <Kv key={name} label={name} value={String(value)} />)
            : null}
        </div>
      ) : null}

      {metrics ? (
        <div style={{ padding: "8px 0", borderBottom: "1px solid var(--rule)" }}>
          <div className="ilabel" style={{ padding: "2px 12px 4px" }}>
            Metrics
          </div>
          {Object.entries(metrics).map(([label, value]) => (
            <Kv key={label} label={label} value={value === null ? "—" : value.toFixed(4)} />
          ))}
        </div>
      ) : null}

      <Provenance
        entries={[
          ["Content hash", version?.v.content_hash ?? datasets?.[0]?.versions[0]?.content_hash ?? "—"],
          ["Pipeline", pipeline?.pipeline_id ?? "—"],
          ["Project", project?.project_id ?? "—"],
          ["App version", "0.2.0"],
        ]}
      />
      {version ? (
        <div className="kv" style={{ paddingTop: 8 }}>
          <b>Array</b>
          <span title={version.v.array_path}>{short(version.v.array_path)}</span>
        </div>
      ) : null}
    </aside>
  );
}
