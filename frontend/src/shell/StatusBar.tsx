import { useEffect, useState } from "react";

import type { Job } from "@/api/queries";
import { LockIcon, StopIcon } from "@/shell/icons";

/** The status bar. A run is a real job, so this shows real progress and can
 * really cancel it - and it is never a blocking modal. */

interface Props {
  job: Job | undefined;
  startedAt: number | null;
  onCancel: () => void;
}

function elapsed(since: number): string {
  const seconds = Math.floor((Date.now() - since) / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function StatusBar({ job, startedAt, onCancel }: Props) {
  const running = job?.status === "queued" || job?.status === "running";
  const [, tick] = useState(0);

  // The clock is the only thing here that changes without the server saying so.
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, [running]);

  const colour =
    job?.status === "failed"
      ? "var(--fail)"
      : job?.status === "cancelled"
        ? "var(--stale)"
        : "var(--accent)";

  return (
    <div className="status" role="status">
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        {job ? (
          <>
            <span style={{ color: colour, fontWeight: 500 }}>{job.message}</span>
            <div className="prog">
              <i style={{ width: `${Math.round(job.progress * 100)}%`, background: colour }} />
            </div>
            {startedAt ? <span className="mono">{elapsed(startedAt)}</span> : null}
            {running ? (
              <button
                className="srow"
                style={{ width: "auto", height: 20, padding: 0, gap: 4, color: "var(--ink3)" }}
                onClick={onCancel}
              >
                <StopIcon />
                Cancel
              </button>
            ) : null}
          </>
        ) : (
          <span style={{ color: "var(--ink3)" }}>Idle</span>
        )}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--ink3)" }}>
        <LockIcon />
        <span>Local · nothing leaves this machine</span>
      </div>
    </div>
  );
}
