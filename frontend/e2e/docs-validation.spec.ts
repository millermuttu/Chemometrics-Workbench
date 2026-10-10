import path from "node:path";

import { expect, test } from "@playwright/test";

import { apply, branch, canvas, node, outline, ROOT, shoot } from "./docs-helpers";

/** docs/examples/validation.md (#340), done as the page says, on a project of
 * its own that starts empty (8771). The numbers are checked by
 * tests/test_examples.py; this checks the steps work and pictures them. */

test.describe.configure({ timeout: 400_000 });

const CSV = path.join(ROOT, "docs", "examples", "meat-raw.csv");

test("the validation example, step by step", async ({ page }) => {
  // Step 1: import both runs of every sample.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await page.getByLabel("Choose file").setInputFiles(CSV);
  const commit = page.getByRole("button", { name: "Import 120 × 448" });
  await expect(commit).toBeVisible();
  await commit.click();
  await expect(page.getByRole("tab", { name: /^meat-raw/ })).toBeVisible();

  // Step 2: split, centre, PLS-DA, then the split grouped by sample.
  await canvas(page);
  await branch(page, page.locator(".react-flow__node").first(), "K-fold 10");
  await branch(page, node(page, "K-fold 10"), "Mean centre");
  await branch(page, node(page, "Mean centre"), "PLS-DA 5 LV");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(node(page, "PLS-DA 5 LV")).toHaveCount(1);
  await outline(page).getByRole("button", { name: /^K-fold 10/ }).first().dblclick();
  await apply(page, { "Group By": "sample" });
  await canvas(page);
  await expect(page.getByTestId("node-complete")).toHaveCount(4, { timeout: 120_000 });
  await shoot(page, "ex-grouped-pipeline");

  // Step 3: the model on screen is fitted on all 120; fold zero is held out.
  await outline(page).getByRole("button", { name: /PLS-DA 5 LV/ }).first().dblclick();
  await expect(page.getByTestId("confusion-cross_validation")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("confusion-held_out")).toBeVisible();
  await shoot(page, "ex-grouped-plsda");

  // Step 4: a hundred permutations, and p at its floor.
  await page.getByLabel("Permutations").fill("100");
  await page.getByRole("button", { name: "Run permutation test" }).click();
  await expect(page.getByText(/^p = 0\.00990 · 100 permutations/)).toBeVisible({ timeout: 240_000 });

  // Step 6 before 5, while the PLS-DA's tab is open: the nested check.
  await page.getByLabel("Variable importance view").selectOption("selection");
  await page.getByRole("button", { name: "Validate (nested)" }).click();
  await expect(page.getByTestId("nested-result")).toContainText("10 outer × 5 inner folds", {
    timeout: 120_000,
  });

  // Step 5: an SVM off the same centring, scored on the same folds.
  await canvas(page);
  await branch(page, node(page, "Mean centre"), "SVM rbf 5 PC", { dx: 120, dy: 110 });
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(node(page, "SVM rbf 5 PC")).toHaveCount(1);
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.getByTestId("node-complete")).toHaveCount(5, { timeout: 120_000 });
  await outline(page).getByRole("button", { name: /SVM rbf 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("confusion-cross_validation")).toBeVisible({ timeout: 120_000 });
});
