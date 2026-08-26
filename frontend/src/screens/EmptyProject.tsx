/** First run: a project with nothing in it, and one obvious next action. */
export function EmptyProject({ onImport }: { onImport: () => void }) {
  return (
    <div className="pane">
      <div style={{ padding: 40, maxWidth: 520, margin: "0 auto", textAlign: "center" }}>
        <h2 style={{ margin: "0 0 6px", fontSize: 14, fontWeight: 600 }}>This project is empty</h2>
        <p style={{ margin: "0 0 16px", color: "var(--ink3)", lineHeight: 1.5 }}>
          Import spectra to begin. CSV, TXT, XLSX and JCAMP-DX are read here; the file stays on this
          machine and nothing is committed until you have seen what the reader found.
        </p>
        <button className="btn btn-p" style={{ margin: "0 auto" }} onClick={onImport}>
          Import data
        </button>
      </div>
    </div>
  );
}
