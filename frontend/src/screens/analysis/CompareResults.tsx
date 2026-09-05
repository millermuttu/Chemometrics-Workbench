import { useResults, type PcaPayload } from "@/api/queries";
import { Panel } from "@/screens/analysis/Panel";

/** Two terminal nodes side by side — DESIGN_BRIEF.md §5's reason for the canvas.
 *
 * The brief draws one dataset forked into four preprocessing paths with an
 * RMSECV against each, and says the canvas exists so those can be compared.
 * This is the two-model case of that picture.
 *
 * **Nothing here is computed, including the difference.** Every figure is a
 * number the server sent; the delta column is a subtraction of two of them,
 * which is arithmetic on displayed values rather than a statistic. A screen
 * that computed a metric would be a second implementation of it.
 *
 * The rows are the union of what the two carry, so comparing a regression with
 * a decomposition is legible rather than refused: what one has and the other
 * does not reads as an em dash, which is the same rule §11 sets for a metric
 * that could not be computed.
 */

const METRICS: [string, string][] = [
  ["RMSEC", "rmsec"],
  ["RMSECV", "rmsecv"],
  ["RMSEP", "rmsep"],
  ["R²", "r2"],
  ["Q²", "q2"],
  ["Bias", "bias"],
  ["SEC", "sec"],
  ["SEP", "sep"],
];

function figure(value: number | undefined, digits = 4) {
  return value === undefined ? "—" : value.toFixed(digits);
}

/** Lower is better for an error, higher for a fit. Only used to point an
 * arrow, never to declare a winner: which model is better is the user's call
 * and depends on what they are calibrating for. */
const LOWER_IS_BETTER = new Set(["rmsec", "rmsecv", "rmsep", "sec", "sep"]);

export interface CompareRow {
  label: string;
  /** The `metrics` key. Not called `key`: it is spread into a React element,
   * where that name is reserved and would be swallowed. */
  metric: string;
  a?: number;
  b?: number;
  /** `b - a`, or undefined when only one of them carries the metric. */
  delta?: number;
  /** Whether `b` is the better of the two, or undefined when they tie or when
   * only one reported. Points an arrow; never declares a winner - which model
   * is better depends on what is being calibrated for and is the user's call. */
  better?: boolean;
}

/** The rows two results have between them, in the order §11 lists them.
 *
 * A metric neither carries is dropped rather than shown as two dashes: an
 * all-empty row says nothing and pushes the ones that matter off the panel.
 * A metric only one carries is kept, because that difference between the two
 * models is itself the finding.
 */
export function compareRows(
  a: PcaPayload["metrics"],
  b: PcaPayload["metrics"],
): CompareRow[] {
  return METRICS.filter(
    ([, key]) => a?.[key] !== undefined || b?.[key] !== undefined,
  ).map(([label, key]) => {
    const left = a?.[key];
    const right = b?.[key];
    const delta =
      left !== undefined && right !== undefined ? right - left : undefined;
    return {
      label,
      metric: key,
      a: left,
      b: right,
      delta,
      better:
        delta === undefined || delta === 0
          ? undefined
          : LOWER_IS_BETTER.has(key)
            ? delta < 0
            : delta > 0,
    };
  });
}

function Row({ label, a, b, delta, better }: CompareRow) {
  return (
    <tr>
      <td style={{ color: "var(--ink2)" }}>{label}</td>
      <td className="n">{figure(a)}</td>
      <td className="n">{figure(b)}</td>
      <td
        className="n"
        style={{
          color: better === undefined ? "var(--ink3)" : "var(--accentInk)",
        }}
        data-testid={`delta-${label}`}
      >
        {delta === undefined
          ? "—"
          : `${delta > 0 ? "+" : ""}${delta.toFixed(4)}`}
        {better === undefined ? "" : better ? " ▾" : " ▴"}
      </td>
    </tr>
  );
}

function Column({ payload }: { payload: PcaPayload }) {
  const regression = payload.task === "regression";
  return (
    <>
      <div className="kv">
        <b>Model</b>
        <span className="mono">
          {regression ? `PLS · ${payload.regression?.target ?? "?"}` : "PCA"}
        </span>
      </div>
      <div className="kv">
        <b>Components</b>
        <span className="mono">{payload.n_components}</span>
      </div>
      <div className="kv">
        <b>Fitted on</b>
        <span className="mono">
          {payload.n_samples} × {payload.n_variables}
        </span>
      </div>
    </>
  );
}

export function CompareResults({
  left,
  right,
}: {
  left: string;
  right: string;
}) {
  const a = useResults(left);
  const b = useResults(right);

  if (!a.data || !b.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading both results…
        </div>
      </div>
    );
  }

  const rows = compareRows(a.data.metrics, b.data.metrics);

  return (
    <div className="pane">
      <div
        style={{
          flex: 1,
          minHeight: 0,
          padding: "12px 14px",
          display: "flex",
          gap: 12,
        }}
        data-testid="compare-view"
      >
        <Panel title={left} width={220}>
          <div style={{ padding: "6px 0" }}>
            <Column payload={a.data} />
          </div>
        </Panel>
        <Panel title={right} width={220}>
          <div style={{ padding: "6px 0" }}>
            <Column payload={b.data} />
          </div>
        </Panel>
        <Panel
          title="Metrics"
          note={rows.length ? "" : "neither node reports one"}
        >
          <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th style={{ width: 110 }}>Metric</th>
                  <th className="n">{left}</th>
                  <th className="n">{right}</th>
                  <th className="n" style={{ width: 92 }}>
                    Difference
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <Row key={row.metric} {...row} />
                ))}
              </tbody>
            </table>
            {rows.length === 0 ? (
              <div className="empty" style={{ padding: 12 }}>
                Neither of these carries a regression metric. A decomposition
                reports explained variance rather than an error, and those are
                on each node&rsquo;s own tab.
              </div>
            ) : null}
          </div>
        </Panel>
      </div>
    </div>
  );
}
