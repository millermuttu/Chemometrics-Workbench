import { expect, test, type Page } from "@playwright/test";

/** The six application states from DESIGN_BRIEF.md section 8, each entered
 * against the stub server rather than by editing code. A state that can only
 * be reached from a hard-coded flag is a state nobody tests.
 */

async function app(page: Page, flags = "") {
  await page.goto(`/?token=e2e-token${flags}`);
  await expect(page.getByText("Tecator meat study")).toBeVisible();
}

test("empty: a project with nothing in it offers the one action that matters", async ({ page }) => {
  await app(page, "&empty");
  await expect(page.getByRole("heading", { name: "This project is empty" })).toBeVisible();
});

test("importing: the preview states what was read before anything is committed", async ({
  page,
}) => {
  await app(page);
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByRole("button", { name: "Use the example file" }).click();
  await expect(page.getByText("tecator.txt")).toBeVisible();
  await expect(page.getByRole("button", { name: /Import 240 × 100/ })).toBeVisible();
});

test("running: one job, three places, and cancel stops it", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await page.getByRole("button", { name: "Run pipeline" }).click();

  // The status bar, the tab and the node all carry the same run.
  const status = page.getByRole("status").first();
  await expect(status).toContainText(/Queued|Preprocessing/);
  await expect(page.getByTestId("tab-progress")).toBeVisible();
  await expect(page.getByTestId("node-running").locator(".prog i")).toBeVisible();

  await expect(status).toContainText("Preprocessing: SNV", { timeout: 6000 });
  await status.getByRole("button", { name: "Cancel" }).click();
  await expect(status).toContainText("Cancelled");
  await expect(page.getByTestId("tab-progress")).toHaveCount(0);
});

test("failed: the run names its cause and shows no trace", async ({ page }) => {
  await app(page, "&failrun");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const failure = page.getByTestId("run-failed");
  await expect(failure).toBeVisible({ timeout: 15000 });
  await expect(failure).toContainText("RUN FAILED");
  await expect(failure).toContainText("matrix of rank 4");
  await expect(failure).not.toContainText("Traceback");

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

test("stale: downstream dims and stays, with a re-run offered", async ({ page }) => {
  await app(page);
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /^MSC/ }).first().dblclick();
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await inspector.getByLabel("Reference").selectOption("median");
  await inspector.getByRole("button", { name: "Apply" }).click();

  await expect(page.getByText("Downstream results are stale.")).toBeVisible();
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();

  const stale = page.getByTestId("node-stale").first();
  await expect(stale).toBeVisible();
  // The artboard's encoding: dashed border over a 135 degree hatch, and a
  // footer saying why. Dimmed, not deleted.
  const drawn = await stale.evaluate((element) => {
    const style = getComputedStyle(element);
    return {
      border: style.borderTopStyle,
      hatch: style.backgroundImage,
      opacity: Number(style.opacity),
    };
  });
  expect(drawn.border).toBe("dashed");
  expect(drawn.hatch).toContain("135deg");
  expect(drawn.opacity).toBeLessThan(1);
  await expect(page.locator(".react-flow__node")).toHaveCount(14);
});

test("overloaded: past the envelope the limit is stated, not hidden", async ({ page }) => {
  await app(page, "&oversize");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /^SNV/ }).first().dblclick();

  const notice = page.getByTestId("overloaded");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("BEYOND THE V1 ENVELOPE");
  await expect(notice).toContainText("42,000 × 6,200");
  await expect(notice).toContainText("about 320 MB");

  // Nothing was plotted, which is the point: the tab is responsive.
  await expect(page.getByTestId("spectra-plot")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Import…" })).toBeEnabled();
});
