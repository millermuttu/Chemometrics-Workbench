import Plotly from "plotly.js-gl2d-dist-min";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { useSpectra, type SpectraPayload } from "@/api/queries";
import { PLOT_CONFIG, axisLayout, baseLayout, readTheme } from "@/plot/theme";
import { bandTraces, spectraTraces } from "@/plot/traces";

/** The most-looked-at screen in the product.
 *
 * The performance envelope is the design constraint here, not a later
 * optimisation: 240 spectra arrive as 60 drawn traces plus a 5/50/95 density
 * band, and PROPOSAL.md section 13 is why. Decimation is consumed - the
 * fixture carries both forms and 1.2 computes them server-side.
 */

type Layer = "raw" | "processed" | "both";

interface PlotlyDiv extends HTMLElement {
  on?: (event: string, handler: (data: { points: { customdata?: number }[] }) => void) => void;
  removeAllListeners?: (event: string) => void;
}

function Plot({
  payloads,
  selected,
  onPick,
}: {
  payloads: { payload: SpectraPayload; faded: boolean }[];
  selected: number[];
  onPick: (index: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  // The handler is read through a ref so it can be attached once, when the
  // plot first exists, rather than re-attached on every render.
  const pick = useRef(onPick);
  pick.current = onPick;

  // The theme is read out of the DOM rather than passed in, so the plot picks
  // up the palette that is actually applied - including a theme switch.
  useLayoutEffect(() => {
    const element = host.current;
    if (!element || payloads.length === 0) return;
    const theme = readTheme(element);
    const primary = payloads[0].payload;

    const data = payloads.flatMap(({ payload, faded }) => [
      ...(payload.decimation.banded ? bandTraces(payload, theme, { faded }) : []),
      ...spectraTraces(payload, theme, { selected, faded }),
    ]);

    void Plotly.react(
      element,
      data,
      {
        ...baseLayout(theme),
        xaxis: axisLayout(theme, `${primary.axis.kind} (${primary.axis.unit ?? ""})`),
        yaxis: axisLayout(theme, primary.ordinate.label),
      },
      PLOT_CONFIG,
    ).then(() => {
      // Plotly attaches `on` to the div while it draws, so the listener can
      // only go on once that has happened - and only once.
      const node = element as PlotlyDiv;
      if (node.dataset.clickBound) return;
      node.dataset.clickBound = "yes";
      node.on?.("plotly_click", (event) => {
        const index = event.points[0]?.customdata;
        if (typeof index === "number") pick.current(index);
      });
    });
  }, [payloads, selected]);

  useEffect(() => {
    const element = host.current;
    return () => {
      if (element) Plotly.purge(element);
    };
  }, []);

  return <div ref={host} data-testid="spectra-plot" style={{ flex: 1, minHeight: 0 }} />;
}

function Segmented({
  layer,
  onChange,
  disabled,
}: {
  layer: Layer;
  onChange: (layer: Layer) => void;
  disabled: boolean;
}) {
  return (
    <div
      role="group"
      aria-label="Layers"
      style={{
        display: "flex",
        border: "1px solid var(--rule)",
        borderRadius: 3,
        overflow: "hidden",
        background: "var(--sunken)",
      }}
    >
      {(["raw", "processed", "both"] as const).map((option) => {
        const active = option === layer;
        return (
          <button
            key={option}
            disabled={disabled && option !== "raw"}
            aria-pressed={active}
            onClick={() => onChange(option)}
            style={{
              padding: "0 10px",
              height: 24,
              border: "none",
              font: "inherit",
              fontSize: 11.5,
              cursor: "pointer",
              background: active ? "var(--surface)" : "transparent",
              color: active ? "var(--ink)" : "var(--ink3)",
              fontWeight: active ? 600 : 400,
            }}
          >
            {option[0].toUpperCase() + option.slice(1)}
          </button>
        );
      })}
    </div>
  );
}

export function SpectraView({ nodeId, title }: { nodeId: string; title: string }) {
  const [layer, setLayer] = useState<Layer>("both");
  const [selected, setSelected] = useState<number[]>([]);

  const processed = useSpectra(nodeId);
  const raw = useSpectra("source");
  const isSource = nodeId === "source";

  const pick = (index: number) =>
    setSelected((current) =>
      current.includes(index) ? current.filter((item) => item !== index) : [...current, index],
    );

  if (!processed.data || !raw.data) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Loading spectra…
        </div>
      </div>
    );
  }

  const primary = isSource ? raw.data : processed.data;
  const payloads =
    isSource || layer === "raw"
      ? [{ payload: raw.data, faded: false }]
      : layer === "processed"
        ? [{ payload: processed.data, faded: false }]
        : [
            { payload: raw.data, faded: true },
            { payload: processed.data, faded: false },
          ];

  const { decimation } = primary;

  return (
    <div className="pane">
      <div
        style={{
          height: 42,
          flex: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          padding: "0 14px",
          borderBottom: "1px solid var(--rule2)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Segmented layer={isSource ? "raw" : layer} onChange={setLayer} disabled={isSource} />
          <span className="mono" style={{ fontSize: 11, color: "var(--ink3)" }}>
            {primary.n_spectra} spectra · {selected.length} highlighted ·{" "}
            {decimation.variables_total} variables
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span className="pill mono">
            {decimation.traces_drawn} of {decimation.traces_total} drawn
          </span>
          <span className="pill mono">
            {decimation.variables_kept === decimation.variables_total
              ? "no x decimation"
              : `decimated to ${decimation.variables_kept} px`}
          </span>
          {/* Clicking a trace is the workflow this screen exists for, but
              sixty overlapping lines are a small target - so the drawn samples
              are also a list. */}
          <select
            className="mono"
            aria-label="Highlight sample"
            value=""
            onChange={(event) => pick(Number(event.target.value))}
            style={{
              height: 22,
              padding: "0 6px",
              borderRadius: 3,
              border: "1px solid var(--rule)",
              background: "var(--surface)",
              color: "var(--ink2)",
              font: "inherit",
              fontSize: 11,
            }}
          >
            <option value="">Highlight sample…</option>
            {primary.traces.map((trace) => (
              <option key={trace.index} value={trace.index}>
                {trace.sample_id}
              </option>
            ))}
          </select>
          {selected.length > 0 ? (
            <button className="btn" style={{ height: 22 }} onClick={() => setSelected([])}>
              Clear highlight
            </button>
          ) : null}
        </div>
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          padding: "8px 14px 10px",
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <span style={{ fontSize: 11.5, fontWeight: 600 }}>{title}</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            shaded band = full set ({primary.band.n_spectra}) · lines drawn at full resolution
          </span>
        </div>
        <Plot payloads={payloads} selected={selected} onPick={pick} />
        {selected.length > 0 ? (
          <div className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            Highlighted:{" "}
            {selected
              .map(
                (index) =>
                  primary.traces.find((trace) => trace.index === index)?.sample_id ?? `#${index}`,
              )
              .join(" · ")}
          </div>
        ) : null}
      </div>
    </div>
  );
}
