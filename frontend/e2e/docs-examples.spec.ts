import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

/** The worked examples' screenshots (#237), taken by doing what the pages
 * tell a reader to do, on a project that starts empty (8769). The numbers the
 * pages quote are checked by tests/test_examples.py; this checks that the
 * steps work as written, through the screens, and pictures each one. The
 * pictures go to docs/images/screens/, which is not committed.
 *
 * One test, because every step builds on the one before it. */

test.describe.configure({ timeout: 300_000 });

const ROOT = path.join(import.meta.dirname, "..", "..");
const OUT = path.join(ROOT, "docs", "images", "screens");
const CSV = path.join(ROOT, "docs", "examples", "tecator.csv");

async function shoot(page: Page, name: string) {
  // Plotly draws after its first frame; let it settle before the capture.
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
}

const outline = (page: Page) => page.getByRole("complementary", { name: "Project outline" });
const inspector = (page: Page) => page.getByRole("complementary", { name: "Inspector" });
const node = (page: Page, label: string) =>
  page.locator(".react-flow__node").filter({ hasText: label });

/** Back to the canvas from whichever tab is in front. */
async function canvas(page: Page) {
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
}

/** Drag from a node's output port onto empty canvas, and pick from the menu. */
async function branch(
  page: Page,
  from: ReturnType<typeof node>,
  step: string,
  { shot }: { shot?: string } = {},
) {
  // The canvas refits as it mounts and again once its nodes are measured; a
  // port read before the second fit is no longer under the pointer (#260).
  const handle = from.locator(".react-flow__handle-right");
  let port = (await handle.boundingBox())!;
  await expect
    .poll(async () => {
      await page.waitForTimeout(150);
      const now = (await handle.boundingBox())!;
      const still = now.x === port.x && now.y === port.y;
      port = now;
      return still;
    })
    .toBe(true);
  await page.mouse.move(port.x + port.width / 2, port.y + port.height / 2);
  await page.mouse.down();
  // A node lands where it is dropped. Down and slightly left by default: a
  // chain that walked right would put the next port under the step list.
  await page.mouse.move(port.x - 60, port.y + 110, { steps: 12 });
  await page.mouse.up();
  const menu = page.getByTestId("add-step-menu");
  await expect(menu).toBeVisible();
  if (shot) await shoot(page, shot);
  await menu.getByRole("menuitem", { name: step, exact: true }).click();
}

async function apply(page: Page, fields: Record<string, string>) {
  for (const [label, value] of Object.entries(fields)) {
    const field = inspector(page).getByLabel(label);
    if ((await field.evaluate((element) => element.tagName)) === "SELECT") {
      await field.selectOption(value);
    } else {
      await field.fill(value);
    }
  }
  await inspector(page).getByRole("button", { name: "Apply and re-run" }).click();
}

test("the two worked examples, step by step", async ({ page }) => {
  // PCA, step 1: import.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await page.getByLabel("Choose file").setInputFiles(CSV);
  const commit = page.getByRole("button", { name: "Import 240 × 100" });
  await expect(commit).toBeVisible();
  await shoot(page, "ex-import");
  await commit.click();
  // The dataset's tab, not a row: the preview already shows C001.
  await expect(page.getByRole("tab", { name: /^tecator/ })).toBeVisible();

  // Step 2: the pipeline, through the step list, saved and run.
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  for (const step of ["SNV", "SG d1 w11", "Mean centre", "PCA"]) {
    await page.getByLabel("Step").selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  await page.getByRole("button", { name: "Validate" }).click();
  await expect(page.getByText("valid · 4 steps")).toBeVisible();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 120_000 });
  await expect(page.getByTestId("node-complete")).toHaveCount(5);
  await canvas(page);
  await shoot(page, "ex-pca-pipeline");

  await outline(page).getByRole("button", { name: /^SG d1 w11/ }).first().dblclick();
  await expect(page.getByTestId("spectra-plot")).toBeVisible();
  await shoot(page, "ex-spectra");

  // Step 3: the PCA's results.
  await outline(page).getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.getByTestId("analysis-header")).toContainText("240 × 100");
  await shoot(page, "ex-pca");

  // Step 4: the outlier the page names, and its contributions.
  await page.getByTestId("outlier-row").filter({ hasText: "M002" }).click();
  await expect(page.getByTestId("contributions-plot")).toBeVisible();
  await shoot(page, "ex-outlier");

  // PLS, step 1: a branch off the derivative, by dragging from its port.
  await canvas(page);
  await branch(page, node(page, "SG d1 w11"), "K-fold 10", { shot: "ex-branch-menu" });
  await branch(page, node(page, "K-fold 10"), "Mean centre");
  await branch(page, node(page, "Mean centre").last(), "PLS 5 LV");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(node(page, "PLS 5 LV")).toHaveCount(1);

  await outline(page).getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  await apply(page, { Target: "fat", "N Components": "10" });
  const header = page.getByTestId("analysis-header");
  await expect(header).toContainText("PLS on fat 10 components", { timeout: 120_000 });

  // Step 2: the curve over ten components, then the chosen four.
  await shoot(page, "ex-rmsecv");
  await apply(page, { "N Components": "4" });
  await expect(header).toContainText("PLS on fat 4 components", { timeout: 120_000 });
  await shoot(page, "ex-pls");

  await canvas(page);
  await expect(node(page, "PLS 4 LV")).toHaveCount(1);
  await shoot(page, "ex-pls-pipeline");

  // Step 5: a wider derivative window, and the two runs compared.
  await outline(page).getByRole("button", { name: /^SG d1 w11/ }).first().dblclick();
  await apply(page, { "Window Length": "15" });
  await expect(outline(page).getByRole("button", { name: /^SG d1 w15/ })).toHaveCount(1, {
    timeout: 120_000,
  });
  await expect(page.locator(".status")).toContainText("Done", { timeout: 120_000 });

  await outline(page).getByRole("button", { name: /^Run \d+/ }).first().dblclick();
  await expect(page.getByTestId("experiment-view")).toBeVisible();
  await page.getByTestId("compare-with").selectOption({ index: 1 });
  const lineage = page.getByTestId("lineage-view");
  await expect(lineage).toBeVisible();
  await expect(lineage).toContainText("window_length");
  await shoot(page, "ex-compare");
});
