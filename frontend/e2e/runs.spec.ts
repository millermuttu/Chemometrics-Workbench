import { expect, test } from "@playwright/test";

/** Runs, against a project whose nodes have never been computed (8767).
 *
 * A run only advances if there is work to do. On the seeded project every
 * array is already on disk, so a run hits the cache and finishes before the
 * screen can render a second frame - which is correct, and useless for
 * watching one. These have their own project for that reason.
 *
 * Ordering matters within the file and Playwright honours it: cancelling
 * leaves the nodes that finished on disk, and the failing run afterwards still
 * fails, because the branch it fails on is not one of them.
 *
 * Real work needs a real budget: 2,000 x 800 through four branches and a
 * ten-fold split is a few seconds cold, and Playwright's default 30 s cap is
 * for tests that only click.
 */

test.describe.configure({ timeout: 180_000 });

test("a run advances through real work, in all three places at once", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();

  const status = page.getByRole("status").first();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(status).toContainText(/Queued|Preprocessing/);

  // The status bar, the tab and the node all carry the same run.
  await expect(page.getByTestId("tab-progress")).toBeVisible();
  await expect(page.getByTestId("node-running").locator(".prog i")).toBeVisible();

  // Progress is counted, never interpolated (#85): it only ever goes up, and
  // it moves because nodes finish rather than because the clock ticks.
  const seen: number[] = [];
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const style = await page.locator(".status .prog i").first().getAttribute("style");
    const percent = Number(/width:\s*([\d.]+)%/.exec(style ?? "")?.[1] ?? "-1");
    if (percent >= 0 && percent !== seen.at(-1)) seen.push(percent);
    if (await status.getByRole("button", { name: "Cancel" }).count()) {
      if (seen.length > 2) break;
    }
    await page.waitForTimeout(100);
  }
  expect(seen.length, `progress advanced through ${seen.join(", ")}`).toBeGreaterThan(2);
  expect(seen).toEqual([...seen].sort((a, b) => a - b));

  await status.getByRole("button", { name: "Cancel" }).click();
  await expect(status).toContainText("Cancelled");

  // Cancelled means stopped: the bar does not move on afterwards.
  const frozen = await page.locator(".status .prog i").first().getAttribute("style");
  await page.waitForTimeout(2_000);
  await expect(status).toContainText("Cancelled");
  expect(await page.locator(".status .prog i").first().getAttribute("style")).toBe(frozen);
  await expect(page.getByTestId("tab-progress")).toHaveCount(0);
});

test("the failure names its cause, marks the node, and shows no trace", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const failure = page.getByTestId("run-failed");
  await expect(failure).toBeVisible({ timeout: 60_000 });
  await expect(failure).toContainText("RUN FAILED");

  // The kernel's own sentence. `decomposition.py` refuses to return fewer
  // components than asked; the rank is not asserted exactly because a centred
  // matrix read back as float32 reports one more than it has (#101).
  await expect(failure).toContainText(/components were asked of a matrix of rank \d+/);
  await expect(failure).not.toContainText("Traceback");
  await expect(page.getByRole("status").first()).toContainText(/rank \d+/);

  // --fail is semantic and separate from the data palette on purpose: a
  // failing thing must never read as a red spectrum.
  const colours = await page.evaluate(() => {
    const style = getComputedStyle(document.querySelector(".app")!);
    return {
      fail: style.getPropertyValue("--fail").trim(),
      series: ["d1", "d2", "d3", "d4", "d5", "d6"].map((token) =>
        style.getPropertyValue(`--${token}`).trim(),
      ),
    };
  });
  expect(colours.series).not.toContain(colours.fail);
});

test("the canvas marks the node that failed, and the ones that never ran", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();

  // The node the executor named, not the last one to report progress: those
  // are different nodes, and `ExecutorError` carries the id so the canvas does
  // not have to parse it back out of English.
  const failed = page.getByTestId("node-failed").first();
  await expect(failed).toBeVisible();
  await expect(failed).toContainText(/rank \d+/);
  await expect(page.getByTestId("node-not_run").first()).toBeVisible();

  // Form, not only colour: a left stripe for failed, dashed for never-run.
  const border = (state: string) =>
    page
      .getByTestId(`node-${state}`)
      .first()
      .evaluate((element) => {
        const style = getComputedStyle(element);
        return { style: style.borderTopStyle, left: style.borderLeftWidth };
      });
  expect((await border("failed")).left).toBe("3px");
  expect((await border("not_run")).style).toBe("dashed");
});
