import Plotly from "plotly.js-gl2d-dist-min";
import { useLayoutEffect, useRef, useState } from "react";

import {
  useCoefficients,
  useContributions,
  useResults,
  useSaveModel,
  type PcaPayload,
} from "@/api/queries";
import {
  coefficientTrace,
  contributionTrace,
  ellipseTrace,
  loadingsTraces,
  outliers,
  predictedTraces,
  rmsecvTrace,
  scoresTrace,
  varianceFigure,
  vipFigure,
} from "@/plot/analysis";
import { PLOT_CONFIG, axisLayout, baseLayout, readTheme } from "@/plot/theme";
import { Panel } from "@/screens/analysis/Panel";
import { CannotLoad } from "@/states/CannotLoad";

/** One analysis tab, a grid of titled panels - the artboard's answer to open
 * design question 11.1, and the layout Phase 2's predicted-vs-measured panel
 * drops into rather than one that has to be torn up.
 *
 * Nothing here is computed. Scores, loadings, variances, T², SPE and both
 * limits arrive as data; the ellipse is drawn from the limit the server sent.
 */

function usePlot(
  build: (theme: ReturnType<typeof readTheme>) => {
    data: Record<string, unknown>[];
    layout: Record<string, unknown>;
  },
  deps: unknown[],
) {
  const host = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const element = host.current;
    if (!element) return;
    const theme = readTheme(element);
    const { data, layout } = build(theme);
    void Plotly.react(element, data, { ...baseLayout(theme), ...layout }, PLOT_CONFIG);
    return () => Plotly.purge(element);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return host;
}

function Scores({ pca }: { pca: PcaPayload }) {
  const [x, setX] = useState(0);
  const [y, setY] = useState(1);
  const host = usePlot(
    (theme) => ({
      data: [ellipseTrace(pca, x, y, theme), scoresTrace(pca, x, y, theme)],
      layout: {
        xaxis: axisLayout(theme, `PC ${x + 1} (${(pca.explained_variance_ratio[x] * 100).toFixed(1)}%)`),
        yaxis: axisLayout(theme, `PC ${y + 1} (${(pca.explained_variance_ratio[y] * 100).toFixed(1)}%)`),
        margin: { l: 48, r: 12, t: 8, b: 38 },
      },
    }),
    [pca, x, y],
  );

  const choose = (value: number, onChange: (value: number) => void, label: string) => (
    <select
      aria-label={label}
      className="mono"
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      style={{
        height: 18,
        borderRadius: 3,
        border: "1px solid var(--rule)",
        background: "var(--surface)",
        color: "var(--ink2)",
        font: "inherit",
        fontSize: 9.5,
      }}
    >
      {pca.explained_variance_ratio.map((_, index) => (
        <option key={index} value={index}>
          PC {index + 1}
        </option>
      ))}
    </select>
  );

  return (
    <Panel
      title="Scores"
      note={
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {choose(x, setX, "Scores x axis")}
          {choose(y, setY, "Scores y axis")}
          <span className="mono" style={{ fontSize: 9.5, color: "var(--ink3)" }}>
            Hotelling T² {Math.round((1 - pca.diagnostics.alpha) * 100)}%
          </span>
        </span>
      }
    >
      <div ref={host} data-testid="scores-plot" style={{ flex: 1, minHeight: 0 }} />
    </Panel>
  );
}

function Loadings({ pca }: { pca: PcaPayload }) {
  const [count, setCount] = useState(2);
  const components = Array.from({ length: count }, (_, index) => index);
  const host = usePlot(
    (theme) => ({
      data: loadingsTraces(pca, components, theme),
      layout: {
        xaxis: axisLayout(theme, `${pca.loadings.axis.kind} (${pca.loadings.axis.unit ?? ""})`),
        yaxis: axisLayout(theme, "Loading"),
        margin: { l: 48, r: 12, t: 8, b: 38 },
      },
    }),
    [pca, count],
  );

  return (
    <Panel
      title="Loadings"
      note={
        <select
          aria-label="Loadings components"
          className="mono"
          value={count}
          onChange={(event) => setCount(Number(event.target.value))}
          style={{
            height: 18,
            borderRadius: 3,
            border: "1px solid var(--rule)",
            background: "var(--surface)",
            color: "var(--ink2)",
            font: "inherit",
            fontSize: 9.5,
          }}
        >
          {[1, 2, 3, 5].map((value) => (
            <option key={value} value={value}>
              PC 1–{value}
            </option>
          ))}
        </select>
      }
    >
      <div ref={host} data-testid="loadings-plot" style={{ flex: 1, minHeight: 0 }} />
    </Panel>
  );
}

/** VIP on the node's axis, or the coefficient vector folded back to the raw
 * axis (#184). Both are things a reader of a calibration looks at beside the
 * loadings, which is why this sits in the loadings' row. Nothing is computed:
 * VIP arrives in the result and the folded vector from its own endpoint,
 * which answers with a sentence when a step in the chain - SNV, MSC, a
 * baseline - is not a fixed linear map and cannot be folded. */
function VariableImportance({ pca }: { pca: PcaPayload }) {
  const [view, setView] = useState<"vip" | "coefficients">("vip");
  const coefficients = useCoefficients(pca.node_id);
  const folded = coefficients.data;

  const vipHost = usePlot(
    (theme) => {
      const figure = vipFigure(pca, theme);
      return {
        data: figure.data,
        layout: {
          shapes: figure.shapes,
          xaxis: axisLayout(theme, `${pca.loadings.axis.kind} (${pca.loadings.axis.unit ?? ""})`),
          yaxis: { ...axisLayout(theme, "VIP"), rangemode: "tozero" },
          margin: { l: 48, r: 12, t: 8, b: 38 },
        },
      };
    },
    [pca, view],
  );
  const coefficientHost = usePlot(
    (theme) => {
      const trace = folded ? coefficientTrace(folded, theme) : null;
      return {
        data: trace ? [trace] : [],
        layout: {
          xaxis: axisLayout(
            theme,
            folded?.axis ? `${folded.axis.kind} (${folded.axis.unit ?? ""})` : "",
          ),
          yaxis: axisLayout(theme, "Coefficient"),
          margin: { l: 48, r: 12, t: 8, b: 38 },
        },
      };
    },
    [folded, view],
  );

  const choose = (
    <select
      aria-label="Variable importance view"
      className="mono"
      value={view}
      onChange={(event) => setView(event.target.value as "vip" | "coefficients")}
      style={{
        height: 18,
        borderRadius: 3,
        border: "1px solid var(--rule)",
        background: "var(--surface)",
        color: "var(--ink2)",
        font: "inherit",
        fontSize: 9.5,
      }}
    >
      <option value="vip">VIP</option>
      <option value="coefficients">Coefficients, raw axis</option>
    </select>
  );

  return (
    <Panel title="Variable importance" note={choose}>
      {view === "vip" ? (
        <div ref={vipHost} data-testid="vip-plot" style={{ flex: 1, minHeight: 0 }} />
      ) : folded?.available ? (
        <div ref={coefficientHost} data-testid="coefficients-plot" style={{ flex: 1, minHeight: 0 }} />
      ) : (
        <div
          role="note"
          data-testid="coefficients-unavailable"
          className="empty"
          style={{ padding: 12, lineHeight: 1.4 }}
        >
          {/* The server's own sentence, which names the step. "Not available"
              alone would leave the reader guessing which node to blame. */}
          {folded ? folded.reason : coefficients.isError ? "Could not load the coefficients." : "Loading…"}
        </div>
      )}
    </Panel>
  );
}

function Variance({ pca }: { pca: PcaPayload }) {
  const host = usePlot(
    (theme) => {
      const figure = varianceFigure(pca, theme);
      return {
      data: figure.data,
      layout: {
        shapes: figure.shapes,
        xaxis: axisLayout(theme, "Component"),
        yaxis: { ...axisLayout(theme, "Explained (%)"), rangemode: "tozero" },
        yaxis2: {
          ...axisLayout(theme, "Cumulative (%)"),
          overlaying: "y",
          side: "right",
          range: [0, 105],
        },
        margin: { l: 44, r: 42, t: 8, b: 38 },
      },
      };
    },
    [pca],
  );
  const cumulative = pca.cumulative_explained_variance.at(-1)! * 100;
  return (
    <Panel title="Explained variance" note={`cumulative ${cumulative.toFixed(1)}%`} width={340}>
      <div ref={host} data-testid="variance-plot" style={{ flex: 1, minHeight: 0 }} />
    </Panel>
  );
}

/** Exported for its test: the caveat is a sentence the kernel wrote, and the
 * panel's one job with it is to put it beside the number it qualifies. */
export function Diagnostics({
  pca,
  picked = null,
  onPick,
}: {
  pca: PcaPayload;
  /** The dataset row whose contributions are open, if any (#186). */
  picked?: number | null;
  onPick?: (index: number) => void;
}) {
  const beyond = outliers(pca);
  const { diagnostics } = pca;
  return (
    <Panel title="Diagnostics" note={`α = ${diagnostics.alpha}`}>
      <div style={{ padding: "6px 0", borderBottom: "1px solid var(--rule2)" }}>
        <div className="kv">
          <b>Hotelling T² limit</b>
          <span>{diagnostics.hotelling_t2_limit.toFixed(4)}</span>
        </div>
        <div className="kv">
          <b>SPE limit</b>
          <span>{diagnostics.spe_limit.toExponential(3)}</span>
        </div>
        {diagnostics.spe_limit_caveat ? (
          // #71: a limit outside its approximation's domain is drawn, and
          // said to be. The sentence is the kernel's, not one written here.
          <p
            role="note"
            data-testid="spe-limit-caveat"
            className="mono"
            style={{ margin: "2px 12px 4px", fontSize: 10, color: "var(--stale)", lineHeight: 1.35 }}
          >
            {diagnostics.spe_limit_caveat}
          </p>
        ) : null}
        <div className="kv">
          <b>Rank</b>
          <span>{pca.rank}</span>
        </div>
        <div className="kv">
          <b>Beyond a limit</b>
          <span>
            {beyond.length} of {pca.n_samples}
          </span>
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        <table>
          <thead>
            <tr>
              <th style={{ width: 74 }}>Sample</th>
              <th className="n">T²</th>
              <th className="n">SPE</th>
            </tr>
          </thead>
          <tbody>
            {beyond.slice(0, 40).map((row) => (
              // A row opens the sample's contributions (#186): which
              // variables put it beyond the limit is the next question.
              <tr
                key={row.sample}
                data-testid="outlier-row"
                data-index={row.index}
                role={onPick ? "button" : undefined}
                tabIndex={onPick ? 0 : undefined}
                aria-pressed={onPick ? picked === row.index : undefined}
                onClick={onPick ? () => onPick(row.index) : undefined}
                onKeyDown={
                  onPick
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") onPick(row.index);
                      }
                    : undefined
                }
                style={{
                  cursor: onPick ? "pointer" : undefined,
                  background: picked === row.index ? "var(--sunken)" : undefined,
                }}
              >
                <td className="mono" style={{ color: "var(--ink)" }}>
                  {row.sample}
                </td>
                <td
                  className="n"
                  style={
                    row.t2 > diagnostics.hotelling_t2_limit ? { color: "var(--stale)" } : undefined
                  }
                >
                  {row.t2.toFixed(3)}
                </td>
                <td
                  className="n"
                  style={row.spe > diagnostics.spe_limit ? { color: "var(--stale)" } : undefined}
                >
                  {row.spe.toExponential(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/** Which variables put a picked sample where the diagnostics show it (#186,
 * `pca.md` §7 and §8): signed `T²` contributions or squared residuals against
 * the node's axis, served by the contributions endpoint and drawn, not
 * computed. Empty until a row in the diagnostics table is picked. */
function Contributions({ pca, sample }: { pca: PcaPayload; sample: number | null }) {
  const [which, setWhich] = useState<"hotelling_t2" | "spe">("hotelling_t2");
  const contributions = useContributions(sample === null ? undefined : pca.node_id, sample);
  const payload = contributions.data;
  const host = usePlot(
    (theme) => ({
      data: payload ? [contributionTrace(payload, which, theme)] : [],
      layout: {
        xaxis: axisLayout(theme, `${pca.loadings.axis.kind} (${pca.loadings.axis.unit ?? ""})`),
        yaxis: axisLayout(theme, which === "spe" ? "e²" : "T² contribution"),
        margin: { l: 48, r: 12, t: 8, b: 38 },
      },
    }),
    [payload, which],
  );
  const total = payload ? (which === "spe" ? payload.spe.total : payload.hotelling_t2.total) : null;
  const choose = (
    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <select
        aria-label="Contribution view"
        className="mono"
        value={which}
        onChange={(event) => setWhich(event.target.value as "hotelling_t2" | "spe")}
        style={{
          height: 18,
          borderRadius: 3,
          border: "1px solid var(--rule)",
          background: "var(--surface)",
          color: "var(--ink2)",
          font: "inherit",
          fontSize: 9.5,
        }}
      >
        <option value="hotelling_t2">T²</option>
        <option value="spe">SPE</option>
      </select>
      {payload ? (
        <span className="mono" style={{ fontSize: 9.5, color: "var(--ink3)" }} data-testid="contributions-note">
          {payload.sample.sample_id} · Σ = {total!.toPrecision(4)}
        </span>
      ) : null}
    </span>
  );
  return (
    <Panel title="Contributions" note={choose}>
      {sample === null ? (
        <div className="empty" data-testid="contributions-empty" style={{ padding: 12 }}>
          Pick a sample in the diagnostics table to see which variables put it there.
        </div>
      ) : contributions.isError ? (
        <div className="empty" role="note" data-testid="contributions-unavailable" style={{ padding: 12 }}>
          {contributions.error instanceof Error ? contributions.error.message : "Could not load."}
        </div>
      ) : (
        <div ref={host} data-testid="contributions-plot" style={{ flex: 1, minHeight: 0 }} />
      )}
    </Panel>
  );
}

/** `metrics-and-validation.md` §11: a metric that could not be computed is
 * absent, never zero and never NaN, and absence renders as an em dash. RMSECV
 * and Q² really are missing above a split and SEC really is missing when
 * `n - A - 1 <= 0`, so printing 0.0000 would assert something false. */
function metric(value: number | undefined, digits = 4) {
  return value === undefined ? "—" : value.toFixed(digits);
}

function PredictedVsMeasured({ pca }: { pca: PcaPayload }) {
  const host = usePlot(
    (theme) => ({
      data: predictedTraces(pca, theme),
      layout: {
        xaxis: axisLayout(theme, `Measured ${pca.regression?.target ?? ""}`),
        yaxis: axisLayout(theme, "Predicted"),
        margin: { l: 48, r: 12, t: 8, b: 38 },
        showlegend: true,
        legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 9.5 } },
      },
    }),
    [pca],
  );
  const held = pca.validation?.observed?.length ?? 0;
  return (
    <Panel
      title="Predicted vs measured"
      note={held ? `${pca.n_samples} calibration · ${held} held out` : `${pca.n_samples} samples`}
    >
      <div ref={host} data-testid="predicted-plot" style={{ flex: 1, minHeight: 0 }} />
    </Panel>
  );
}

function RmsecvCurve({ pca }: { pca: PcaPayload }) {
  const curve = pca.rmsecv_curve ?? [];
  const host = usePlot(
    (theme) => ({
      data: [rmsecvTrace(pca, theme)],
      layout: {
        xaxis: { ...axisLayout(theme, "Components"), dtick: 1 },
        yaxis: { ...axisLayout(theme, "RMSECV"), rangemode: "tozero" },
        margin: { l: 48, r: 12, t: 8, b: 38 },
      },
    }),
    [pca],
  );
  // The minimum is reported, never chosen: §9 says picking A there and then
  // quoting that minimum as the model's expected error is optimistic, and that
  // it is the user's call rather than the application's.
  const best = curve.length ? curve.indexOf(Math.min(...curve)) + 1 : undefined;
  return (
    <Panel title="RMSECV" note={best ? `lowest at A = ${best}` : "needs a split"} width={340}>
      <div ref={host} data-testid="rmsecv-plot" style={{ flex: 1, minHeight: 0 }} />
    </Panel>
  );
}

/** The confusion matrices a two-class PLS-DA reports (#185, `pls-da.md` §6):
 * rows observed, columns assigned, in the classes' order, for the calibration
 * set and - below a split - the cross-validated and held-out sets. Counts,
 * not a plot: four numbers per set are read, not drawn. Exported for its test. */
export function ConfusionMatrix({ pca }: { pca: PcaPayload }) {
  const classification = pca.classification;
  if (!classification) return null;
  const sets: [string, string][] = [
    ["calibration", "Calibration"],
    ["cross_validation", "Cross-validated"],
    ["held_out", "Held out (fold 0)"],
  ];
  const [c0, c1] = classification.classes;
  return (
    <Panel title="Confusion" note={`${c0} · ${c1}`}>
      <div
        data-testid="confusion-matrix"
        style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "6px 0" }}
      >
        {sets
          .filter(([key]) => classification.confusion[key])
          .map(([key, label]) => {
            const [[tn, fp], [fn, tp]] = classification.confusion[key];
            return (
              <table key={key} data-testid={`confusion-${key}`} style={{ marginBottom: 8 }}>
                <thead>
                  <tr>
                    <th style={{ width: 110 }}>{label}</th>
                    <th className="n">→ {c0}</th>
                    <th className="n">→ {c1}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="mono">{c0}</td>
                    <td className="n">{tn}</td>
                    <td className="n">{fp}</td>
                  </tr>
                  <tr>
                    <td className="mono">{c1}</td>
                    <td className="n">{fn}</td>
                    <td className="n">{tp}</td>
                  </tr>
                </tbody>
              </table>
            );
          })}
      </div>
    </Panel>
  );
}

function RegressionMetrics({ pca }: { pca: PcaPayload }) {
  const m = pca.metrics ?? {};
  const rows: [string, string][] =
    pca.task === "classification"
      ? [
          // pls-da.md section 6, by set: none is calibration, _cv the pooled
          // held-out assignments, _p fold zero's held-out rows.
          ["Accuracy", metric(m.accuracy, 3)],
          ["Accuracy (CV)", metric(m.accuracy_cv, 3)],
          ["Accuracy (held out)", metric(m.accuracy_p, 3)],
          ["Sensitivity", metric(m.sensitivity, 3)],
          ["Sensitivity (CV)", metric(m.sensitivity_cv, 3)],
          ["Specificity", metric(m.specificity, 3)],
          ["Specificity (CV)", metric(m.specificity_cv, 3)],
          ["RMSECV (dummy)", metric(m.rmsecv)],
        ]
      : [
          ["RMSEC", metric(m.rmsec)],
          ["RMSECV", metric(m.rmsecv)],
          ["RMSEP", metric(m.rmsep)],
          ["R²", metric(m.r2)],
          ["Q²", metric(m.q2)],
          ["Bias", metric(m.bias)],
          ["SEC", metric(m.sec)],
          ["SEP", metric(m.sep)],
          ["RMSECV spread", metric(m.rmsecv_std)],
        ];
  return (
    <Panel title="Calibration metrics" note={pca.regression?.target ?? ""} width={260}>
      <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "6px 0" }}>
        {rows.map(([label, value]) => (
          <div className="kv" key={label}>
            <b>{label}</b>
            <span className="mono" data-testid={`metric-${label}`}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/** Save this fitted estimator as a model the project holds (#219).
 *
 * The name is the user's, so it is an input rather than something generated:
 * a registry of "pls", "pls (2)" and "pls (3)" is a list nobody can read. It
 * starts as the node's title because that is the obvious first answer, and a
 * saved model is not a correction of the last one - saving twice is two
 * entries, which is what the server does.
 */
function SaveModel({ nodeId, title }: { nodeId: string; title: string }) {
  const save = useSaveModel();
  const [name, setName] = useState<string | null>(null);
  const value = name ?? title;

  return (
    <form
      style={{ display: "flex", alignItems: "center", gap: 6 }}
      onSubmit={(event) => {
        event.preventDefault();
        if (value.trim()) save.mutate({ nodeId, name: value.trim() });
      }}
    >
      <input
        aria-label="Model name"
        data-testid="model-name"
        value={value}
        onChange={(event) => setName(event.target.value)}
        style={{ width: 150 }}
      />
      <button type="submit" data-testid="save-model" disabled={save.isPending || !value.trim()}>
        {save.isPending ? "Saving…" : "Save model"}
      </button>
      {/* A failed save says why. The server's sentence is the message, as
          every other failure on this screen has it. */}
      {save.isError ? (
        <span className="mono" role="alert" style={{ fontSize: 10.5, color: "var(--fail)" }}>
          {save.error instanceof Error ? save.error.message : "The save failed."}
        </span>
      ) : save.isSuccess ? (
        <span className="pill mono" data-testid="model-saved" style={{ fontSize: 10.5 }}>
          saved
        </span>
      ) : null}
    </form>
  );
}

export function AnalysisResults({ nodeId, title }: { nodeId: string; title: string }) {
  const results = useResults(nodeId);
  const [picked, setPicked] = useState<number | null>(null);

  // A node with no result answers 404, and this used to render as a loading
  // message that never resolved (#181). The server's sentence says what to do.
  if (results.isError) {
    return <CannotLoad error={results.error} />;
  }
  if (!results.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading results…
        </div>
      </div>
    );
  }

  const pca = results.data;
  const classification = pca.task === "classification";
  // A classification is the regression on a dummy response (pls-da.md
  // section 2), so every regression panel applies; only what it is called and
  // which panel sits first differ.
  const regression = pca.task === "regression" || classification;
  return (
    <div className="pane">
      <div
        data-testid="analysis-header"
        style={{
          height: 52,
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
          <span className="mono" style={{ fontSize: 11, color: "var(--ink3)" }}>
            {classification
              ? `PLS-DA on ${pca.classification?.class_column ?? "?"}`
              : regression
                ? `PLS on ${pca.regression?.target ?? "?"}`
                : "PCA"}{" "}
            {pca.n_components}{" "}
            components · {pca.n_samples} × {pca.n_variables}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <div style={{ paddingRight: 11 }}>
            <SaveModel nodeId={nodeId} title={title} />
          </div>
          {(classification
            ? ([
                ["ACCURACY (CV)", metric(pca.metrics?.accuracy_cv, 3)],
                ["ACCURACY", metric(pca.metrics?.accuracy, 3)],
              ] as [string, string][])
            : regression
            ? // What a reader of a calibration looks at first, and the pair
              // that says whether it generalises. Absent renders as an em dash.
              ([
                ["RMSECV", metric(pca.metrics?.rmsecv)],
                ["Q²", metric(pca.metrics?.q2, 3)],
              ] as [string, string][])
            : ([
                ["PC1", `${(pca.explained_variance_ratio[0] * 100).toFixed(1)}%`],
                ["CUMULATIVE", `${(pca.cumulative_explained_variance.at(-1)! * 100).toFixed(1)}%`],
              ] as [string, string][])
          ).map(([label, value]) => (
            <div
              key={label}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 1,
                padding: "0 11px",
                borderLeft: "1px solid var(--rule2)",
              }}
            >
              <span className="ilabel" style={{ fontSize: 9 }}>
                {label}
              </span>
              <span
                className="mono"
                style={{ fontSize: 14, fontWeight: 600, color: "var(--accentInk)" }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          padding: "12px 14px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
          <Scores pca={pca} />
          <Loadings pca={pca} />
          {regression && <VariableImportance pca={pca} />}
        </div>
        <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
          <Variance pca={pca} />
          <Diagnostics pca={pca} picked={picked} onPick={setPicked} />
          <Contributions pca={pca} sample={picked} />
        </div>
        {/* The row this comment reserved in Phase 1.1, now filled. It arrives
            beside the two above rather than replacing them, exactly as the
            layout was drawn for - and only for a regression, because these
            three have no counterpart on a decomposition. */}
        {regression && (
          <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
            {classification ? <ConfusionMatrix pca={pca} /> : <PredictedVsMeasured pca={pca} />}
            <RmsecvCurve pca={pca} />
            <RegressionMetrics pca={pca} />
          </div>
        )}
      </div>
    </div>
  );
}
