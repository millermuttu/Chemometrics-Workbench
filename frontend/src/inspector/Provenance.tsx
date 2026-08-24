import { useState } from "react";

/** Provenance, collapsed by default: content hash, pipeline hash, app version.
 * Hashes are mono and truncated in the middle - `sha256:9f3c…a71b` - because
 * the ends are what a person compares. */

function middle(hash: string): string {
  return hash.length > 22 ? `${hash.slice(0, 11)}…${hash.slice(-4)}` : hash;
}

export function Provenance({ entries }: { entries: [string, string][] }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: "1px solid var(--rule)" }}>
      <button
        className="srow"
        aria-expanded={open}
        style={{ height: 30, paddingLeft: 12, color: "var(--ink2)" }}
        onClick={() => setOpen(!open)}
      >
        <span>Provenance record</span>
        <span className="sdim mono">{open ? "−" : "+"}</span>
      </button>
      {open ? (
        <div style={{ padding: "0 0 8px" }}>
          {entries.map(([label, value]) => (
            <div className="kv" key={label}>
              <b>{label}</b>
              <span title={value}>{middle(value)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
