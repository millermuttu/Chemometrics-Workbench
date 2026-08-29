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

  // `.status` is the run status bar; a second role="status" carries the stale
  // banner, and DOM order puts that one first.
  const status = page.locator(".status");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  // Everything below is *observed while the run is on*, not demanded at an
  // instant. A node is `running` for as long as that node takes, which is a
  // fraction of a run that is itself a few seconds: an assertion that arrives
  // a moment late finds a finished run and fails on a machine being quick.
  // That is what made this flaky on macOS, and it is the same shape of mistake
  // three other tests here made. What the test wants to know is whether these
  // three places carried the run at all, so it watches for them.
  const seen: number[] = [];
  let sawRunningNode = false;
  let sawTabProgress = false;
  let sawStatus = false;

  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (!sawTabProgress && (await page.getByTestId("tab-progress").count())) sawTabProgress = true;
    if (!sawRunningNode && (await page.getByTestId("node-running").count())) sawRunningNode = true;
    if (!sawStatus && /Queued|Preprocessing|Fitting/.test((await status.innerText()) || "")) {
      sawStatus = true;
    }

    const style = await page.locator(".status .prog i").first().getAttribute("style");
    const percent = Number(/width:\s*([\d.]+)%/.exec(style ?? "")?.[1] ?? "-1");
    if (percent >= 0 && percent !== seen.at(-1)) seen.push(percent);

    // Stop once there is enough to assert and there is still a run to cancel.
    const cancellable = await status.getByRole("button", { name: "Cancel" }).count();
    if (cancellable && sawRunningNode && sawTabProgress && seen.length >= 2) break;
    if (!cancellable && seen.length) break;
    await page.waitForTimeout(100);
  }

  // The status bar, the tab and the node all carried the same run.
  expect(sawStatus, "the status bar named the run").toBe(true);
  expect(sawTabProgress, "the tab carried the run").toBe(true);
  expect(sawRunningNode, "a node showed as running").toBe(true);

  // It advanced, and it never went backwards. Not *how many* frames were
  // caught: that is a fact about how fast the runner is. #85's real claim,
  // that progress is counted from nodes finishing rather than interpolated
  // against the clock, is asserted where it can be observed exactly - in the
  // Python job tests.
  expect(seen.length, `progress advanced through ${seen.join(", ")}`).toBeGreaterThanOrEqual(2);
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
  await expect(page.locator(".status")).toContainText(/rank \d+/);

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
