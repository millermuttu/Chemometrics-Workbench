import { useState } from "react";

import { api } from "@/api/client";
import { checkBounds, type FieldSpec, type StepSpec } from "@/inspector/schema";

/** The typed editor for one preprocessing step.
 *
 * Field bounds come from the schema and are checked as you type. The rules
 * that span fields are checked by the server against `models.py` itself, so
 * "window_length must be odd" is the model's sentence, not one written here.
 */

interface Props {
  spec: StepSpec;
  values: Record<string, string>;
  onChange: (values: Record<string, string>) => void;
  onApply: () => void;
}

function Field({
  field,
  value,
  problem,
  onChange,
}: {
  field: FieldSpec;
  value: string;
  problem: string | null;
  onChange: (value: string) => void;
}) {
  const border = problem ? "var(--fail)" : "var(--rule)";
  return (
    <div style={{ padding: "3px 12px" }}>
      <div className="kv" style={{ padding: 0, alignItems: "center" }}>
        <b title={field.description}>{field.title}</b>
        {field.kind === "enum" ? (
          <select
            aria-label={field.title}
            className="mono"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            style={{
              width: 116,
              height: 22,
              borderRadius: 3,
              border: `1px solid ${border}`,
              background: "var(--surface)",
              color: "var(--ink)",
              font: "inherit",
              fontSize: 11.5,
            }}
          >
            {field.options?.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        ) : (
          <input
            aria-label={field.title}
            className="mono"
            inputMode="numeric"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            style={{
              width: 116,
              height: 22,
              padding: "0 6px",
              textAlign: "right",
              borderRadius: 3,
              border: `1px solid ${border}`,
              background: "var(--surface)",
              color: "var(--ink)",
              font: "inherit",
              fontSize: 11.5,
              fontVariantNumeric: "tabular-nums",
            }}
          />
        )}
      </div>
      {problem ? (
        <p role="alert" style={{ margin: "2px 0 0", fontSize: 10.5, color: "var(--fail)" }}>
          {problem}
        </p>
      ) : null}
    </div>
  );
}

export function ParameterForm({ spec, values, onChange, onApply }: Props) {
  const [serverProblems, setServerProblems] = useState<Record<string, string>>({});
  const [checking, setChecking] = useState(false);

  const bounds = Object.fromEntries(
    spec.fields.map((field) => [field.name, checkBounds(field, values[field.name] ?? "")]),
  );
  const problems = { ...serverProblems, ...bounds };
  const blocked = Object.values(problems).some(Boolean);

  const apply = async () => {
    setChecking(true);
    // The cross-field rules live in model_validator and have no JSON Schema
    // form; asking the model is cheaper and more honest than restating them.
    const payload: Record<string, unknown> = { kind: spec.kind };
    for (const field of spec.fields) {
      const raw = values[field.name];
      if (raw === "" || raw === undefined) continue;
      payload[field.name] = field.kind === "enum" ? raw : Number(raw);
    }
    const result = await api<{ valid: boolean; errors: { field: string; message: string }[] }>(
      "/steps/validate",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
    );
    setChecking(false);
    if (result.valid) {
      setServerProblems({});
      onApply();
      return;
    }
    // A field-level complaint lands on its field; a cross-field one - which
    // Pydantic reports against the model rather than a field - belongs to the
    // form as a whole.
    const names = new Set(spec.fields.map((field) => field.name));
    setServerProblems(
      Object.fromEntries(
        result.errors.map((error) => {
          const leaf = error.field.split(".").at(-1) ?? "step";
          return [names.has(leaf) ? leaf : "step", error.message];
        }),
      ),
    );
  };

  return (
    <div style={{ padding: "6px 0 10px", borderBottom: "1px solid var(--rule)" }}>
      <div className="ilabel" style={{ padding: "2px 12px 4px" }}>
        Parameters
      </div>
      {spec.fields.length === 0 ? (
        <div className="empty">This step has no parameters.</div>
      ) : (
        spec.fields.map((field) => (
          <Field
            key={field.name}
            field={field}
            value={values[field.name] ?? ""}
            problem={problems[field.name] ?? null}
            onChange={(next) => {
              setServerProblems({});
              onChange({ ...values, [field.name]: next });
            }}
          />
        ))
      )}
      {serverProblems.step ? (
        <p role="alert" style={{ margin: "2px 12px 0", fontSize: 10.5, color: "var(--fail)" }}>
          {serverProblems.step}
        </p>
      ) : null}
      {spec.fields.length > 0 ? (
        <div style={{ padding: "8px 12px 0" }}>
          <button className="btn" style={{ height: 24 }} disabled={blocked || checking} onClick={apply}>
            {checking ? "Checking…" : "Apply"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
