import { useState } from "react";

import type { DraftStep } from "@/canvas/graph";
import { STEPS } from "@/canvas/catalogue";

/** The pipeline builder beside the canvas: appending, and what a drag cannot do.
 *
 * The list is still the right tool for a linear chain - SNV -> Savitzky-Golay
 * -> PCA is three clicks here and three drags on the canvas. #51 made the
 * canvas editable for the case a list cannot express, a branch, and left this
 * where it was.
 *
 * The one thing it gained is a Save enabled by a graph edit as well as by a
 * draft step. Removing a node is a control on the node itself: clicking a node
 * opens its tab, so anything in here would be replaced by the tab before it
 * could be clicked.
 */

/** The catalogue, with the payload each entry becomes.
 *
 * The parameters are the defaults `models.py` already carries, written out
 * rather than left implicit: what is sent is what the canvas shows, and a
 * default that changed in the schema should change the label beside it.
 * Editing them is the inspector's job once the node is saved.
 */

interface Props {
  steps: DraftStep[];
  onChange: (steps: DraftStep[]) => void;
  onValidate: () => void;
  onSave: () => void;
  saving: boolean;
  validation: string | null;
  /** The canvas holds edits the server has not seen, so Save has work to do. */
  edited: boolean;
}

export function StepList({
  steps,
  onChange,
  onValidate,
  onSave,
  saving,
  validation,
  edited,
}: Props) {
  const [choice, setChoice] = useState(STEPS[0].kind);

  return (
    <aside
      style={{
        width: 236,
        flex: "none",
        borderLeft: "1px solid var(--rule)",
        background: "var(--panel)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div className="ilabel" style={{ padding: "10px 12px 6px" }}>
        Build a pipeline
      </div>

      {steps.length === 0 ? (
        <div className="empty">No steps yet. The draft appears on the canvas as you add them.</div>
      ) : (
        steps.map((step, index) => (
          <div key={`${step.kind}-${index}`} className="srow" style={{ paddingLeft: 12 }}>
            <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>
              {index + 1}
            </span>
            <span>{step.kind}</span>
            <button
              className="tabx"
              aria-label={`Remove ${step.kind}`}
              style={{ marginLeft: "auto" }}
              onClick={() => onChange(steps.filter((_, position) => position !== index))}
            >
              ×
            </button>
          </div>
        ))
      )}

      <div style={{ display: "flex", gap: 6, padding: "10px 12px" }}>
        <select
          aria-label="Step"
          className="mono"
          value={choice}
          onChange={(event) => setChoice(event.target.value)}
          style={{
            flex: 1,
            height: 24,
            borderRadius: 3,
            border: "1px solid var(--rule)",
            background: "var(--surface)",
            color: "var(--ink)",
            font: "inherit",
            fontSize: 11,
          }}
        >
          {STEPS.map((step) => (
            <option key={step.kind} value={step.kind}>
              {step.kind}
            </option>
          ))}
        </select>
        <button
          className="btn"
          style={{ height: 24 }}
          onClick={() => onChange([...steps, STEPS.find((step) => step.kind === choice)!])}
        >
          Add
        </button>
      </div>

      <div style={{ padding: "0 12px 10px", display: "flex", gap: 6, alignItems: "center" }}>
        <button className="btn" style={{ height: 24 }} disabled={steps.length === 0} onClick={onValidate}>
          Validate
        </button>
        {/* Until #108 the draft lived in this tab and nowhere else: closing it
            lost the recipe, and running the experiment ran the source node.
            Saving is what makes the pipeline the record §8 says it is. */}
        <button
          className="btn btn-p"
          style={{ height: 24 }}
          disabled={(steps.length === 0 && !edited) || saving}
          onClick={onSave}
        >
          {saving ? "Saving…" : "Save"}
        </button>
        {validation ? (
          <span className="mono" style={{ fontSize: 10.5, color: "var(--accent)" }}>
            {validation}
          </span>
        ) : null}
      </div>
    </aside>
  );
}
