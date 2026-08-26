import { ApiError } from "@/api/client";

/** The shell could not load its own project.
 *
 * A local-first application can be honest here in a way a web page cannot:
 * the server is on this machine, so a failure is knowable rather than a guess
 * about somebody's network. The three cases a user can act on are told apart -
 * the token was refused, the server is not answering, or the request failed
 * some other way - and each says what to do next.
 *
 * There is no retry loop. A 401 is not transient, and spinning on it is what
 * this replaces.
 */
export function CannotLoad({ error }: { error: unknown }) {
  const api = error instanceof ApiError ? error : null;
  const unauthorised = api?.status === 401;
  const unreachable = !api;

  const heading = unauthorised
    ? "Not authenticated"
    : unreachable
      ? "The workbench server is not answering"
      : "The workbench server refused the request";

  const explanation = unauthorised
    ? "This window has no session token, so the server refused every request. The token is handed over once, in the launch URL."
    : unreachable
      ? "Nothing is listening on this address. The server runs on this machine and is started with the application, so it has either stopped or was never started."
      : (api?.message ?? "The request failed.");

  const remedy = unauthorised
    ? "Reopen the application from its launch URL — the one printed when the server started, ending in ?token=…"
    : unreachable
      ? "Start the server again, then reload this window."
      : "Reload the window. If it keeps failing, restart the application.";

  return (
    <div className="pane">
      <div
        role="alert"
        data-testid="cannot-load"
        style={{
          margin: 16,
          maxWidth: 560,
          padding: "12px 14px",
          borderRadius: 3,
          border: "1px solid var(--fail)",
          background: "var(--failSoft)",
        }}
      >
        <div className="mono" style={{ fontSize: 10, color: "var(--fail)" }}>
          {api ? `${api.status} · ${api.code.toUpperCase()}` : "NO RESPONSE"}
        </div>
        <h2 style={{ margin: "4px 0 0", fontSize: 13.5, fontWeight: 600 }}>{heading}</h2>
        <p style={{ margin: "4px 0 0", color: "var(--ink)", lineHeight: 1.45 }}>{explanation}</p>
        <p style={{ margin: "6px 0 0", fontSize: 11.5, color: "var(--ink3)", lineHeight: 1.45 }}>
          {remedy}
        </p>
      </div>
    </div>
  );
}
