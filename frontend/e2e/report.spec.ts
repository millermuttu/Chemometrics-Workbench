import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import { expect, test } from "@playwright/test";

/** The HTML report (#224): the file the server hands over opens on its own.
 *
 * Saved to disk and opened from `file://` with the network off, so a report
 * that leaned on the server, a CDN or anything else outside itself would show
 * a blank page here rather than a plot. `tests/test_report.py` checks the
 * numbers; this checks that a browser draws them. */

test("the exported report opens with no server and no network and shows the scores plot", async ({
  page,
  context,
}, testInfo) => {
  const response = await page.request.get("/api/experiments/current/report.html", {
    headers: { Authorization: "Bearer e2e-token" },
  });
  expect(response.status()).toBe(200);
  const file = testInfo.outputPath("report.html");
  await writeFile(file, await response.body());

  await context.setOffline(true);
  // A file URL, not `file://` plus a path: a Windows path has a drive letter.
  await page.goto(pathToFileURL(file).href);

  // Filtered on the caption: a figure's own text starts with its axis labels.
  const scores = page
    .locator("figure")
    .filter({ has: page.locator("figcaption", { hasText: /^Scores, \d+ samples$/ }) });
  await expect(scores).toBeVisible();
  const n = Number((await scores.locator("figcaption").textContent())?.match(/\d+/)?.[0]);
  expect(n).toBeGreaterThan(0);
  // One drawn point per sample: the plot is there, not only its caption.
  await expect(scores.locator("svg circle")).toHaveCount(n);
  const box = await scores.locator("svg").boundingBox();
  expect(box?.width).toBeGreaterThan(100);
});
