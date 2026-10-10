import path from "node:path";

import { expect, test } from "@playwright/test";

import { apply, branch, canvas, node, outline, ROOT, shoot } from "./docs-helpers";

/** docs/examples/classification.md (#288), done as the page says, on a project
 * that starts empty (8770) and is its own: the Tecator examples' project
 * already holds a dataset and a pipeline. The numbers are checked by
 * tests/test_examples.py; this checks the steps work and pictures them. */

test.describe.configure({ timeout: 300_000 });

const CSV = path.join(ROOT, "docs", "examples", "meat.csv");

test("the classification example, step by step", async ({ page }) => {
  // Step 1: import.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await page.getByLabel("Choose file").setInputFiles(CSV);
  const commit = page.getByRole("button", { name: "Import 60 × 448" });
  await expect(commit).toBeVisible();
  await shoot(page, "ex-meat-import");
  await commit.click();
  await expect(page.getByRole("tab", { name: /^meat/ })).toBeVisible();

  // Step 2: split, centre, PLS-DA, then the split stratified by meat.
  // Off the ports, as the page tells it; the Step list offers the same (#337).
  await canvas(page);
  await branch(page, page.locator(".react-flow__node").first(), "K-fold 10");
  await branch(page, node(page, "K-fold 10"), "Mean centre");
  await branch(page, node(page, "Mean centre"), "PLS-DA 5 LV");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(node(page, "PLS-DA 5 LV")).toHaveCount(1);
  await outline(page).getByRole("button", { name: /^K-fold 10/ }).first().dblclick();
  await apply(page, { "Stratify By": "meat" });
  await canvas(page);
  await expect(page.getByTestId("node-complete")).toHaveCount(4, { timeout: 120_000 });
  await shoot(page, "ex-meat-pipeline");

  // Step 3: the PLS-DA's results, every sample on the diagonal.
  await outline(page).getByRole("button", { name: /PLS-DA 5 LV/ }).first().dblclick();
  const crossValidated = page.getByTestId("confusion-cross_validation");
  await expect(crossValidated).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("rmsecv-plot")).toBeVisible();
  await shoot(page, "ex-plsda");

  // Step 4: three more classifiers off the centring, fanned out so each drop
  // lands on empty canvas rather than on the node before it.
  await canvas(page);
  const centre = node(page, "Mean centre");
  await branch(page, centre, "LDA 5 PC", { dx: -220, dy: 110 });
  await branch(page, centre, "kNN k5", { dx: 120, dy: 110 });
  await branch(page, centre, "SIMCA 3 PC", { dx: 280, dy: 110 });
  await page.getByRole("button", { name: "Save", exact: true }).click();
  for (const label of ["LDA 5 PC", "kNN k5", "SIMCA 3 PC"]) {
    await expect(node(page, label)).toHaveCount(1);
  }
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.getByTestId("node-complete")).toHaveCount(7, { timeout: 120_000 });

  // Step 5: SIMCA reads as acceptance, not confusion.
  await outline(page).getByRole("button", { name: /SIMCA 3 PC/ }).first().dblclick();
  await expect(page.getByTestId("acceptance-cross_validation")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("confusion-matrix")).toHaveCount(0);
  await shoot(page, "ex-simca");
});
