import { useEffect, useState } from "react";

import { ApiError } from "@/api/client";
import { useImportDataset, useImportPreview, type ImportPreview } from "@/api/queries";

/** The screen that decides whether a file becomes a dataset.
 *
 * No artboard was drawn for it - DESIGN_BRIEF.md section 6 asked for the first
 * four screens and this was among "the rest". It is built from the vocabulary
 * the artboards do establish: the table rules, the .kv rows, .pill and .btn. A
 * preview is a table plus a form, and nothing here is a new visual idea.
 *
 * Nothing is committed until the user confirms. The detection comes from the
 * server, never from a guess made here: the real readers land in 1.2 and have
 * to return this shape.
 */

interface Props {
  onImported: (versionId: string, name: string) => void;
  onCancel: () => void;
}

const LABELS: Record<string, string> = {
  whitespace: "whitespace",
  ",": "comma ,",
  ";": "semicolon ;",
  "\t": "tab",
  "|": "pipe |",
  ".": "point .",
  samples_in_rows: "samples in rows",
  samples_in_columns: "samples in columns",
  AB: "AB · absorbance",
  Refl: "Refl · reflectance",
  ScSm: "ScSm · sample single channel",
  ScRf: "ScRf · reference single channel",
  none: "none · numbered",
  wavelength_nm: "nm · wavelength",
  "wavenumber_cm-1": "cm⁻¹ · wavenumber",
  "raman_shift_cm-1": "cm⁻¹ · Raman shift",
  index: "index · numbered",
};

const label = (value: string) => LABELS[value] ?? value;

function Choice({
  name,
  detected,
  value,
  onChange,
}: {
  name: string;
  detected: { value: string; alternatives: string[] };
  value: string;
  onChange: (value: string) => void;
}) {
  const corrected = value !== detected.value;
  return (
    <div className="kv" style={{ alignItems: "center", padding: "5px 12px" }}>
      <b>{name}</b>
      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <select
          className="mono"
          aria-label={name}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          style={{
            height: 22,
            padding: "0 6px",
            borderRadius: 3,
            border: `1px solid ${corrected ? "var(--accent)" : "var(--rule)"}`,
            background: "var(--surface)",
            color: "var(--ink)",
            font: "inherit",
            fontSize: 11.5,
          }}
        >
          {[detected.value, ...detected.alternatives].map((option) => (
            <option key={option} value={option}>
              {label(option)}
            </option>
          ))}
        </select>
        {corrected ? (
          <span className="mono" style={{ fontSize: 10, color: "var(--accent)" }}>
            corrected
          </span>
        ) : null}
      </span>
    </div>
  );
}

/** A failure names the file and the setting. It is never a stack trace, and
 * the body it reads is the one the server documents. */
export function Failure({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const detail = error.detail as Record<string, unknown> | undefined;
  return (
    <div
      role="alert"
      style={{
        margin: 12,
        padding: "10px 12px",
        borderRadius: 3,
        border: "1px solid var(--fail)",
        background: "var(--failSoft)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="mono" style={{ fontSize: 10, color: "var(--fail)" }}>
          {error.code.toUpperCase()}
        </span>
        {onRetry ? (
          <button className="btn" style={{ marginLeft: "auto", height: 22 }} onClick={onRetry}>
            Try again
          </button>
        ) : null}
      </div>
      <p style={{ margin: "4px 0 0", color: "var(--ink)" }}>{error.message}</p>
      {detail ? (
        <p className="mono" style={{ margin: "4px 0 0", fontSize: 10.5, color: "var(--ink3)" }}>
          {Object.entries(detail)
            .map(([key, item]) => `${key}: ${String(item)}`)
            .join(" · ")}
        </p>
      ) : null}
    </div>
  );
}

function Preview({
  preview,
  file,
  onImported,
  onCancel,
}: Props & { preview: ImportPreview; file: File }) {
  const [corrections, setCorrections] = useState<Record<string, string>>({});
  const commit = useImportDataset();
  // #284: a correction can change which table is read - another matrix of a
  // MAT-file, another block of an OPUS file - so the preview is asked again
  // with it, and the counts, axis, targets and rows shown are the server's.
  //
  // Only for those: a delimiter or a text file's orientation re-reads the same
  // table, its reader does not recount, and the counts are turned here.
  // A MAT-file's reader does recount on an orientation (its axis and targets
  // follow), so there that one is asked again too.
  const first = preview.detected;
  // An axis kind asks again too, so the range is shown in the unit chosen.
  const rereads = [
    "matrix",
    "axis_variable",
    "block",
    "axis_kind",
    ...(first.matrix ? ["orientation"] : []),
  ];
  const refresh = useImportPreview();
  const asked = JSON.stringify(
    Object.fromEntries(Object.entries(corrections).filter(([key]) => rereads.includes(key))),
  );
  useEffect(() => {
    if (asked === "{}") refresh.reset();
    else refresh.mutate({ file, corrections: JSON.parse(asked) as Record<string, string> });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asked, file]);
  const current = asked !== "{}" && refresh.data ? refresh.data : preview;
  const { source } = preview;
  const { detected, head } = current;
  // The choices compare against what was first detected, so a correction
  // still reads as one after the preview has been asked again.

  const value = (
    key: "delimiter" | "decimal" | "orientation" | "block" | "matrix" | "axis_variable" | "axis_kind",
  ) => corrections[key] ?? detected[key]?.value ?? "";

  // Reading a file the other way round is the common wrong guess, and it
  // swaps what the counts mean. Say so before the import, not after. Once the
  // server has read it the other way round, its counts already say so.
  const flipped = value("orientation") !== detected.orientation.value;
  const samples = flipped ? detected.n_variables : detected.n_samples;
  const variables = flipped ? detected.n_samples : detected.n_variables;
  const error =
    commit.error instanceof ApiError
      ? commit.error
      : refresh.error instanceof ApiError
        ? refresh.error
        : null;

  return (
    <div className="pane" style={{ overflowY: "auto" }}>
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
          <span style={{ fontWeight: 600, fontSize: 13.5 }}>{source.filename}</span>
          <span className="pill mono">{source.reader}</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink3)" }}>
            {Math.round(source.size_bytes / 1024)} kB
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <button className="btn" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn-p"
            disabled={commit.isPending}
            onClick={async () => {
              const entry = await commit.mutateAsync({ file, corrections });
              const version = entry.versions.at(-1)!;
              onImported(version.version_id, entry.dataset.name);
            }}
          >
            Import {samples} × {variables}
          </button>
        </div>
      </div>

      {error ? <Failure error={error} /> : null}

      <div style={{ display: "flex", minHeight: 0, flex: 1 }}>
        <section style={{ width: 360, flex: "none", borderRight: "1px solid var(--rule)" }}>
          <div className="ilabel" style={{ padding: "10px 12px 4px" }}>
            Detected
          </div>
          <Choice
            name="Delimiter"
            detected={first.delimiter}
            value={value("delimiter")}
            onChange={(next) => setCorrections({ ...corrections, delimiter: next })}
          />
          <Choice
            name="Decimal"
            detected={first.decimal}
            value={value("decimal")}
            onChange={(next) => setCorrections({ ...corrections, decimal: next })}
          />
          <Choice
            name="Orientation"
            detected={first.orientation}
            value={value("orientation")}
            onChange={(next) => setCorrections({ ...corrections, orientation: next })}
          />
          {first.matrix ? (
            // A MAT-file is a bag of named arrays (#284): which is the
            // spectra, and which vector is their axis, are the user's to say.
            <Choice
              name="Matrix"
              detected={first.matrix}
              value={value("matrix")}
              onChange={(next) => {
                // Another matrix has its own axis candidates.
                const rest: Record<string, string> = { ...corrections, matrix: next };
                delete rest.axis_variable;
                setCorrections(rest);
              }}
            />
          ) : null}
          {detected.axis_variable ? (
            <Choice
              name="Axis from"
              detected={detected.axis_variable}
              value={value("axis_variable")}
              onChange={(next) => setCorrections({ ...corrections, axis_variable: next })}
            />
          ) : null}
          {first.block ? (
            // An OPUS file holds several spectra of one measurement (#187);
            // which one becomes the dataset is the user's to say.
            <Choice
              name="Block"
              detected={first.block}
              value={value("block")}
              onChange={(next) => setCorrections({ ...corrections, block: next })}
            />
          ) : null}
          <div className="kv">
            <b>Samples</b>
            <span>{samples}</span>
          </div>
          <div className="kv">
            <b>Variables</b>
            <span>{variables}</span>
          </div>

          <div className="ilabel" style={{ padding: "12px 12px 4px" }}>
            Variable axis
          </div>
          {first.axis_kind ? (
            // Guessed from the numbers alone, so an ascending FT-IR file reads
            // as nanometres until the user says otherwise (#336).
            <Choice
              name="Kind"
              detected={first.axis_kind}
              value={value("axis_kind")}
              onChange={(next) => setCorrections({ ...corrections, axis_kind: next })}
            />
          ) : (
            <div className="kv">
              <b>Kind</b>
              <span>{detected.axis.kind}</span>
            </div>
          )}
          <div className="kv">
            <b>Range</b>
            <span>
              {detected.axis.start}–{detected.axis.end} {detected.axis.unit}
            </span>
          </div>
          {detected.axis.reconstructed ? (
            <p style={{ margin: "4px 12px 0", fontSize: 11, color: "var(--stale)", lineHeight: 1.35 }}>
              {detected.axis.note}
            </p>
          ) : null}

          <div className="ilabel" style={{ padding: "12px 12px 4px" }}>
            Targets
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, padding: "0 12px" }}>
            {detected.targets.map((target) => (
              <span key={target} className="pill mono">
                {target}
              </span>
            ))}
            {detected.metadata_columns.map((column) => (
              <span key={column} className="pill mono">
                {column}
              </span>
            ))}
          </div>

          {detected.discarded.length > 0 ? (
            <>
              <div className="ilabel" style={{ padding: "12px 12px 4px" }}>
                Not imported
              </div>
              {detected.discarded.map((item) => (
                <p key={item.what} style={{ margin: "0 12px 6px", fontSize: 11, color: "var(--ink3)" }}>
                  <span style={{ color: "var(--ink2)" }}>{item.what}</span> — {item.why}
                </p>
              ))}
            </>
          ) : null}
        </section>

        <section style={{ flex: 1, minWidth: 0, overflow: "auto" }}>
          <div className="ilabel" style={{ padding: "10px 12px 6px" }}>
            First rows, as they would be read
          </div>
          <table>
            <thead>
              <tr>
                <th style={{ width: 74 }}>Sample</th>
                {head.rows[0]?.map((_, index) => (
                  <th key={index} className="n">
                    v{index + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {head.sample_ids.map((id, row) => (
                <tr key={id}>
                  <td className="mono" style={{ color: "var(--ink)" }}>
                    {id}
                  </td>
                  {head.rows[row]?.map((cell, column) => (
                    <td key={column} className="n">
                      {cell.toFixed(4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}

export function Import({ onImported, onCancel }: Props) {
  const preview = useImportPreview();
  const [file, setFile] = useState<File | null>(null);
  const error = preview.error instanceof ApiError ? preview.error : null;

  if (preview.data && file) {
    return (
      <Preview preview={preview.data} file={file} onImported={onImported} onCancel={onCancel} />
    );
  }

  return (
    <div className="pane">
      <div style={{ padding: 16, maxWidth: 620 }}>
        <h2 style={{ margin: "0 0 4px", fontSize: 13.5, fontWeight: 600 }}>Import data</h2>
        <p style={{ margin: "0 0 12px", color: "var(--ink3)" }}>
          Choose a file. Nothing is imported until you have seen what the reader found and said so.
        </p>
        <p className="mono" style={{ margin: "0 0 12px", fontSize: 10.5, color: "var(--ink3)" }}>
          CSV or TXT · XLSX · JCAMP-DX · Bruker OPUS (one file, or a .zip of a folder of them)
        </p>

        {error ? <Failure error={error} onRetry={() => preview.reset()} /> : null}

        {/* The file the user picks is the file that is read. In Phase 1.1 this
            input discarded it and the server answered from a fixture, which is
            why the failure state needed a button of its own; now an unreadable
            file reaches it by being unreadable. */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label className="btn" style={{ position: "relative", overflow: "hidden" }}>
            {preview.isPending ? "Reading…" : "Choose file…"}
            <input
              type="file"
              aria-label="Choose file"
              style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer" }}
              onChange={(event) => {
                const chosen = event.target.files?.[0];
                if (!chosen) return;
                setFile(chosen);
                preview.mutate({ file: chosen });
              }}
            />
          </label>
          {file ? (
            <span className="mono" style={{ fontSize: 11, color: "var(--ink3)" }}>
              {file.name}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
