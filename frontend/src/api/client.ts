/** The one seam between this application and its server.
 *
 * Everything the UI knows about the backend goes through `api()`. Phase 1.1
 * talked to a stub server (#53) and Phase 1.2 replaced its handlers behind the
 * same URLs (#89), which cost this file nothing - which is what the seam was
 * for. If it ever needs to be more than that, it is in the wrong place.
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
  // tab, and VITE_WORKBENCH_TOKEN covers the first load against a server
  // started with a pinned WORKBENCH_TOKEN.
  return sessionStorage.getItem("token") ?? import.meta.env.VITE_WORKBENCH_TOKEN ?? "";
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

/** A read that fails before any response is tried again, twice (#351).
 *
 * Windows runners - and so, one day, a user's machine - occasionally fail a
 * single loopback request with `net::ERR_NO_BUFFER_SPACE`. Taken at its word,
 * that one failed `GET /api/projects` put "the server is not answering" over a
 * window whose server was answering everything else. Only a network failure is
 * retried, and only on a GET: an HTTP status is the server's answer and stands
 * (CannotLoad has no retry loop, for a 401 above all), and a POST that failed
 * midway is not one to send twice. A server that is really gone still says so,
 * a third of a second later. */
async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const attempts = (init.method ?? "GET").toUpperCase() === "GET" ? 3 : 1;
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await fetch(`/api${path}`, {
        ...init,
        headers: { ...init.headers, Authorization: `Bearer ${apiToken()}` },
      });
    } catch (error) {
      if (attempt >= attempts) throw error;
      await new Promise((resolve) => setTimeout(resolve, 100 * attempt));
    }
  }
}

/** Call the server. `path` is relative to /api - `api("/projects")`. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await send(path, init);
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

/** Every failure has a body, not only a status code - the shape the stub
 * server documents and 1.2 keeps. */
async function failure(response: Response): Promise<ApiError> {
  const body = (await response.json().catch(() => null)) as {
    error?: { code?: string; message?: string; detail?: unknown };
  } | null;
  return new ApiError(
    response.status,
    body?.error?.code ?? "request_failed",
    body?.error?.message ?? response.statusText,
    body?.error?.detail,
  );
}

/** Save what the server serves at `path` as a file (#247).
 *
 * A plain link cannot carry the bearer token, so the file is fetched, held as
 * a blob and handed to the browser through a temporary object URL. The name is
 * the server's `Content-Disposition` one when it gives one, else `fallback`.
 * A refusal throws the server's sentence, as `api()` does. */
export async function download(path: string, fallback: string): Promise<string> {
  const response = await send(path);
  if (!response.ok) throw await failure(response);
  const disposition = response.headers.get("content-disposition") ?? "";
  const name = /filename="([^"]+)"/.exec(disposition)?.[1] ?? fallback;
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
  return name;
}
