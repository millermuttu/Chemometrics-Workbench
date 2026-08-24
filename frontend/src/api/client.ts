/** The one seam between this application and its server.
 *
 * Everything the UI knows about the backend goes through `api()`. Phase 1.1
 * talks to the stub server (#53); Phase 1.2 replaces its handlers behind the
 * same URLs, and pointing at the real backend is a base-URL change and nothing
 * else. If it ever needs to be more than that, this seam is in the wrong place.
 */

/** The session token, passed once in the launch URL and then held here.
 *
 * PROPOSAL.md section 4.3: no cookie carries it, so a cross-site request
 * cannot ride an ambient session. It is read out of the query string and the
 * string is cleaned up, so the token does not sit in the address bar or in
 * anything that copies it.
 */
function readToken(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    sessionStorage.setItem("token", fromUrl);
    url.searchParams.delete("token");
    window.history.replaceState({}, "", url);
    return fromUrl;
  }
  // A dev-server reload has no launch URL to read; the token survives in the
  // tab, and VITE_STUB_TOKEN covers the first load against a pinned STUB_TOKEN.
  return sessionStorage.getItem("token") ?? import.meta.env.VITE_STUB_TOKEN ?? "";
}

let token = "";

export function apiToken(): string {
  if (!token) token = readToken();
  return token;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly detail: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Call the server. `path` is relative to /api - `api("/projects")`. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { ...init.headers, Authorization: `Bearer ${apiToken()}` },
  });
  if (!response.ok) {
    // Every failure has a body, not only a status code - the shape the stub
    // server documents and 1.2 keeps.
    const body = (await response.json().catch(() => null)) as {
      error?: { code?: string; message?: string; detail?: unknown };
    } | null;
    throw new ApiError(
      response.status,
      body?.error?.code ?? "request_failed",
      body?.error?.message ?? response.statusText,
      body?.error?.detail,
    );
  }
  return (await response.json()) as T;
}
