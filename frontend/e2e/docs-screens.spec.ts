import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

/** The documentation site's screenshots (#235), taken from the running
 * application on the seeded Tecator project rather than by hand, so they
 * cannot drift from the screens they show. They are written into
 * docs/images/screens/, which is not committed: CI's docs job runs this spec
 * and then builds the site with `--strict`, which fails on a missing image.
 *
 * Named so it sorts before inspector.spec.ts, which edits the seeded project. */

const OUT = path.join(import.meta.dirname, "..", "..", "docs", "images", "screens");

async function shoot(page: Page, name: string) {
  // Plotly draws after its first frame; let it settle before the capture.
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
}

async function open(page: Page, node: RegExp) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: node }).first().dblclick();
}

test("the pipeline canvas", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await shoot(page, "pipeline");
});

test("the spectra view", async ({ page }) => {
  await open(page, /^SNV/);
  await expect(page.getByTestId("spectra-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
  await shoot(page, "spectra");
});

test("PCA results", async ({ page }) => {
  await open(page, /PCA 5 PC/);
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
  await shoot(page, "pca");
});

test("PLS results", async ({ page }) => {
  await open(page, /PLS 5 LV/);
  await expect(page.getByTestId("analysis-header")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
  await shoot(page, "pls");
});
