import Plotly from "plotly.js-gl2d-dist-min";
import { useLayoutEffect, useRef, useState } from "react";

import { useResults, type PcaPayload } from "@/api/queries";
import {
  ellipseTrace,
  loadingsTraces,
  outliers,
  scoresTrace,
  varianceFigure,
} from "@/plot/analysis";
import { PLOT_CONFIG, axisLayout, baseLayout, readTheme } from "@/plot/theme";
import { Panel } from "@/screens/analysis/Panel";

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

function Diagnostics({ pca }: { pca: PcaPayload }) {
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
              <tr key={row.sample}>
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

export function AnalysisResults({ nodeId, title }: { nodeId: string; title: string }) {
  const results = useResults(nodeId);

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
            PCA {pca.n_components} components · {pca.n_samples} × {pca.n_variables}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "stretch" }}>
          {[
            ["PC1", `${(pca.explained_variance_ratio[0] * 100).toFixed(1)}%`],
            ["CUMULATIVE", `${(pca.cumulative_explained_variance.at(-1)! * 100).toFixed(1)}%`],
          ].map(([label, value]) => (
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
        </div>
        <div style={{ display: "flex", gap: 12, flex: 1, minHeight: 0 }}>
          <Variance pca={pca} />
          <Diagnostics pca={pca} />
          {/* Predicted vs measured belongs here, and is Phase 2. The row is
              laid out so it arrives beside these rather than replacing them. */}
        </div>
      </div>
    </div>
  );
}
