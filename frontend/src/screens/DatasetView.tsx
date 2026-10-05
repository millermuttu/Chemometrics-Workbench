import { useState } from "react";

import {
  type DatasetEntry,
  type DatasetVersion,
  useExcludeSamples,
  usePipeline,
  useSavePipeline,
} from "@/api/queries";

/** The dataset a user has just imported: what came in, what is excluded, and
 * where it came from. Built from the artboard's table rules - mono uppercase
 * headers, tabular numbers right-aligned.
 *
 * Ticking rows and excluding them writes a new version without them (#270),
 * and the pipeline moves onto it. A version made that way names its parent and
 * the samples it left out, and offers to put the pipeline back on the parent. */

const ROWS = 60;

function summary(version: DatasetVersion) {
  const axis = version.axis.values;
  const range = axis?.length
    ? `${axis[0].toFixed(0)}–${axis.at(-1)!.toFixed(0)} ${version.axis.unit ?? ""}`
    : version.axis.kind;
  return [
    `${version.n_samples} × ${version.n_variables}`,
    range,
    `${Object.keys(version.targets).length} targets`,
  ];
}

export function DatasetView({ entry, version }: { entry: DatasetEntry; version: DatasetVersion }) {
  const targets = Object.keys(version.targets);
  const metadata = Object.keys(version.metadata_columns);
  // `excluded_samples` are rows of the parent, which this version no longer
  // has, so they are named by the parent's ids rather than drawn as rows.
  const parent = entry.versions.find((v) => v.version_id === version.derived_from);
  const left = version.excluded_samples.map(
    (row) => parent?.sample_ids[row] ?? `row ${row}`,
  );
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const exclude = useExcludeSamples();
  const pipeline = usePipeline();
  const save = useSavePipeline();
  const source = pipeline.data?.nodes.find((node) => node.type === "source");
  const inUse = source?.version_id === version.version_id;
  const restore = () => {
    if (!pipeline.data || !parent) return;
    save.mutate(
      pipeline.data.nodes.map((node) =>
        node.type === "source" ? { ...node, version_id: parent.version_id } : node,
      ),
    );
  };
  const toggle = (row: number) =>
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(row)) next.delete(row);
      else next.add(row);
      return next;
    });

  return (
    <div className="pane">
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
          <span style={{ fontWeight: 600, fontSize: 13.5 }}>{entry.dataset.name}</span>
          <span className="pill mono">v{version.version}</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            {version.content_hash.slice(0, 18)}…
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          {summary(version).map((item) => (
            <span key={item} className="pill mono">
              {item}
            </span>
          ))}
          {parent ? (
            <span
              className="pill mono"
              data-testid="excluded-pill"
              title={left.join(", ")}
              style={{ color: "var(--stale)", borderColor: "var(--stale)" }}
            >
              {left.length} excluded from v{parent.version}
            </span>
          ) : null}
          {parent && inUse ? (
            <button type="button" className="btn" onClick={restore} disabled={save.isPending}>
              Restore v{parent.version}
            </button>
          ) : null}
          {picked.size > 0 ? (
            <button
              type="button"
              className="btn"
              disabled={exclude.isPending}
              onClick={() =>
                exclude.mutate(
                  {
                    datasetId: entry.dataset.dataset_id,
                    fromVersionId: version.version_id,
                    exclude: [...picked],
                  },
                  { onSuccess: () => setPicked(new Set()) },
                )
              }
            >
              Exclude {picked.size} selected
            </button>
          ) : null}
        </div>
      </div>

      {parent || exclude.error ? (
        <div
          className="mono"
          style={{ padding: "6px 14px", fontSize: 11, borderBottom: "1px solid var(--rule2)" }}
        >
          {exclude.error ? (
            <span role="alert" style={{ color: "var(--fail)" }}>
              {exclude.error.message}
            </span>
          ) : (
            <span data-testid="excluded-ids" style={{ color: "var(--ink3)" }}>
              Left out of v{parent!.version}: {left.join(", ")}
            </span>
          )}
        </div>
      ) : null}

      <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        <table>
          <thead>
            <tr>
              <th style={{ width: 26 }} aria-label="Select" />
              <th className="n" style={{ width: 38 }}>
                #
              </th>
              <th style={{ width: 90 }}>Sample</th>
              {metadata.map((column) => (
                <th key={column}>{column}</th>
              ))}
              {targets.map((target) => (
                <th key={target} className="n">
                  {target}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {version.sample_ids.slice(0, ROWS).map((id, index) => (
              <tr key={id}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Select row ${index + 1}`}
                    checked={picked.has(index)}
                    onChange={() => toggle(index)}
                  />
                </td>
                <td className="n" style={{ color: "var(--ink3)" }}>
                  {index + 1}
                </td>
                <td className="mono" style={{ color: "var(--ink)" }}>
                  {id}
                </td>
                {metadata.map((column) => (
                  <td key={column}>{String(version.metadata_columns[column][index])}</td>
                ))}
                {targets.map((target) => (
                  <td key={target} className="n">
                    {version.targets[target][index]?.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {version.sample_ids.length > ROWS ? (
          <div className="empty">
            {/* Virtualised scrolling is #45's problem, where the payload is
                large enough to need it. */}
            Showing {ROWS} of {version.sample_ids.length} samples.
          </div>
        ) : null}
      </div>
    </div>
  );
}
