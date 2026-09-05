import { Handle, Position, type NodeProps } from "@xyflow/react";

import type { NodeData } from "@/canvas/graph";

/** The artboard's node, at the artboard's measurements.
 *
 * 132 x 70, or 74 where a footer carries a content hash. Header 17px on
 * --sunken in mono 9px uppercase; body a 12px label over its parameters in
 * mono 9.5px; footer above a --rule2 rule.
 *
 * The five states are encoded in form as well as colour, which is what makes
 * them survive greyscale and colour blindness: complete is solid, running
 * takes an accent border and shows progress inside the node, stale is a
 * dashed border over a 135 degree hatch, failed carries a left stripe, and
 * not-yet-run is a dashed outline.
 */

const HEADER = 17;

const PORT: React.CSSProperties = {
  width: 7,
  height: 7,
  background: "var(--surface)",
  border: "1px solid var(--rule)",
};

function frame(state: NodeData["state"]): React.CSSProperties {
  switch (state) {
    case "running":
      return { border: "1px solid var(--accent)", background: "var(--surface)" };
    case "stale":
      return {
        border: "1px dashed var(--stale)",
        background:
          "repeating-linear-gradient(135deg,var(--surface),var(--surface) 5px,var(--staleSoft) 5px,var(--staleSoft) 10px)",
        opacity: 0.72,
      };
    case "failed":
      return {
        border: "1px solid var(--rule)",
        borderLeft: "3px solid var(--fail)",
        background: "var(--surface)",
      };
    case "queued":
      // Dashed like `not_run`, because it has produced nothing either - but in
      // the accent, so a run visibly sweeps the graph instead of only the one
      // node the executor happens to be inside.
      return { border: "1px dashed var(--accent)", background: "var(--surface)" };
    case "not_run":
      return { border: "1px dashed var(--rule)", background: "var(--surface)" };
    default:
      // Complete. Solid *and* green: the brief asks for form as well as
      // colour, and solid is what the other five states are distinguished
      // from - a canvas read in greyscale still says which nodes hold a
      // result.
      return { border: "1px solid var(--ok)", background: "var(--surface)" };
  }
}

export function NodeCard({ data, selected }: NodeProps) {
  const node = data as NodeData;
  const stale = node.state === "stale";
  const failed = node.state === "failed";

  return (
    <div
      data-state={node.state}
      data-testid={`node-${node.state}`}
      style={{
        width: 132,
        minHeight: 70,
        borderRadius: 4,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        boxShadow: selected ? "0 0 0 2px var(--accentSoft)" : undefined,
        ...frame(node.state),
      }}
    >
      {/* The ports are what a branch is dragged from (#51), so they are
          visible rather than hit-areas only - a port nobody can see is a
          feature nobody finds. Small and low contrast: structure, not
          decoration, the same rule the edges follow. */}
      <Handle type="target" position={Position.Left} style={PORT} />
      <div
        className="mono"
        style={{
          height: HEADER,
          flex: "none",
          display: "flex",
          alignItems: "center",
          padding: "0 8px",
          background: stale ? "var(--staleSoft)" : "var(--sunken)",
          fontSize: 9,
          letterSpacing: ".1em",
          textTransform: "uppercase",
          color: stale ? "var(--stale)" : "var(--ink3)",
        }}
      >
        {node.type}
        {/* Removing reconnects the children to the parent - `edits.ts` owns
            that rule, and the source has no control at all because it is where
            the data enters. The click is stopped here: it would otherwise also
            open the node's tab, which is what a click on the body means. */}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 2 }}>
          {/* Comparing is a toggle, not a verb: the first click arms it and the
              second node opens the tab, so an armed node has to say it is armed.
              Offered only where there is something to compare. */}
          {node.onCompare ? (
            <button
              className="tabx nodrag"
              aria-label={`${node.comparing ? "Unpick" : "Pick"} ${node.label} for comparison`}
              aria-pressed={node.comparing}
              style={{ color: node.comparing ? "var(--accentInk)" : undefined }}
              onClick={(event) => {
                event.stopPropagation();
                node.onCompare!();
              }}
            >
              ⇄
            </button>
          ) : null}
          {node.onDuplicate ? (
            <button
              className="tabx nodrag"
              aria-label={`Duplicate node ${node.label}`}
              onClick={(event) => {
                event.stopPropagation();
                node.onDuplicate!();
              }}
            >
              ⧉
            </button>
          ) : null}
          {node.onRemove ? (
            <button
              className="tabx nodrag"
              aria-label={`Remove node ${node.label}`}
              onClick={(event) => {
                event.stopPropagation();
                node.onRemove!();
              }}
            >
              ×
            </button>
          ) : null}
        </span>
      </div>

      <div style={{ flex: 1, padding: "5px 8px 0", minHeight: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 500, color: "var(--ink)", lineHeight: 1.25 }}>
          {node.label}
        </div>
        <div className="mono" style={{ fontSize: 9.5, color: "var(--ink3)", marginTop: 2 }}>
          {node.parameters}
        </div>
        {node.state === "running" || node.state === "queued" ? (
          <div className="prog" style={{ width: "100%", marginTop: 6 }}>
            <i style={{ width: `${Math.round((node.progress ?? 0) * 100)}%` }} />
          </div>
        ) : null}
      </div>

      {node.footer ? (
        <div
          className="mono"
          style={{
            padding: "4px 8px 6px",
            fontSize: 10,
            borderTop: "1px solid var(--rule2)",
            color: stale ? "var(--stale)" : failed ? "var(--fail)" : "var(--ink2)",
            // A failure names what went wrong; the node is 132px wide and the
            // message is a sentence, so it wraps rather than being truncated
            // into something unreadable.
            lineHeight: 1.3,
          }}
        >
          {node.footer}
        </div>
      ) : null}
      <Handle type="source" position={Position.Right} style={PORT} />
    </div>
  );
}
