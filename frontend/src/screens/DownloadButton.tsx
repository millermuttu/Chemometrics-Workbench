import { useState } from "react";

import { download } from "@/api/client";

/** A button that saves one file the server serves (#247): the JSON model, the
 * prediction snippet, the HTML report. A refusal - an export asked of a chain
 * with a baseline in it, say - shows the server's own sentence beside it. */
export function DownloadButton({
  label,
  path,
  fallback,
  testId,
}: {
  label: string;
  path: string;
  fallback: string;
  testId: string;
}) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const save = async () => {
    setBusy(true);
    setProblem(null);
    try {
      await download(path, fallback);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "The download failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <button className="btn" data-testid={testId} disabled={busy} onClick={() => void save()}>
        {label}
      </button>
      {problem ? (
        <span
          className="mono"
          role="alert"
          data-testid={`${testId}-error`}
          style={{ fontSize: 10.5, color: "var(--fail)", maxWidth: 260 }}
        >
          {problem}
        </span>
      ) : null}
    </span>
  );
}
