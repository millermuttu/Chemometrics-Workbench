import type { DatasetEntry, DatasetVersion } from "@/api/queries";

/** The dataset a user has just imported: what came in, what is excluded, and
 * where it came from. Built from the artboard's table rules - mono uppercase
 * headers, tabular numbers right-aligned, excluded rows dimmed and labelled. */

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
  const excluded = new Set(version.excluded_samples);

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
          {excluded.size > 0 ? (
            <span className="pill mono" style={{ color: "var(--stale)", borderColor: "var(--stale)" }}>
              {excluded.size} excluded
            </span>
          ) : null}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        <table>
          <thead>
            <tr>
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
              <th style={{ textAlign: "right" }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {version.sample_ids.slice(0, ROWS).map((id, index) => (
              <tr key={id} style={excluded.has(index) ? { opacity: 0.45 } : undefined}>
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
                <td style={{ textAlign: "right" }}>
                  {excluded.has(index) ? (
                    <span className="mono" style={{ color: "var(--stale)", fontSize: 10 }}>
                      excluded
                    </span>
                  ) : null}
                </td>
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
