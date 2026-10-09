/** The token is on every request, and a failure arrives as an ApiError. */
import { afterEach, expect, it, vi } from "vitest";

import { ApiError, api } from "@/api/client";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);
vi.stubGlobal("sessionStorage", {
  getItem: () => "test-token",
  setItem: () => undefined,
});
vi.stubGlobal("window", { location: { href: "http://127.0.0.1/" }, history: { replaceState: () => undefined } });

afterEach(() => fetchMock.mockReset());

it("carries the bearer token on every request", async () => {
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ ok: 1 }) });
  await api("/projects");
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe("/api/projects");
  expect((init.headers as Record<string, string>).Authorization).toBe("Bearer test-token");
});

it("turns the documented error body into an ApiError", async () => {
  fetchMock.mockResolvedValue({
    ok: false,
    status: 422,
    statusText: "Unprocessable Entity",
    json: async () => ({ error: { code: "reader_failed", message: "truncated", detail: { row: 87 } } }),
  });
  await expect(api("/import")).rejects.toThrowError(ApiError);
  await expect(api("/import")).rejects.toMatchObject({ status: 422, code: "reader_failed" });
});

it("tries a read again when it fails before any response (#351)", async () => {
  fetchMock
    .mockRejectedValueOnce(new TypeError("Failed to fetch"))
    .mockResolvedValue({ ok: true, json: async () => ({ ok: 1 }) });
  await expect(api("/projects")).resolves.toEqual({ ok: 1 });
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("gives up on a read after three network failures", async () => {
  fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
  await expect(api("/projects")).rejects.toThrowError(TypeError);
  expect(fetchMock).toHaveBeenCalledTimes(3);
});

it("sends a write once, and never retries an answer the server gave", async () => {
  fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
  await expect(api("/import", { method: "POST" })).rejects.toThrowError(TypeError);
  expect(fetchMock).toHaveBeenCalledTimes(1);

  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: false, status: 401, statusText: "", json: async () => null });
  await expect(api("/projects")).rejects.toMatchObject({ status: 401 });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});
