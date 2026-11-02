import path from "node:path";

import { expect, test } from "@playwright/test";

import { apply, branch, canvas, node, outline, ROOT, shoot } from "./docs-helpers";

/** docs/examples/model-comparison.md (#382), done as the page says, on a
 * project that starts empty (8772). The numbers are checked by
 * tests/test_examples.py; this checks the steps work and pictures them.
 * The classifiers' half is the end of docs-classification.spec.ts, on the
 * meat project that walk builds. */

test.describe.configure({ timeout: 600_000 });

const CSV = path.join(ROOT, "docs", "examples", "tecator.csv");

test("the model comparison example, step by step", async ({ page }) => {
  // Step 1: import, and the PLS example's chain through the step list.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await page.getByLabel("Choose file").setInputFiles(CSV);
  await page.getByRole("button", { name: "Import 240 × 100" }).click();
  await expect(page.getByRole("tab", { name: /^tecator/ })).toBeVisible();
  await canvas(page);
  for (const step of ["SNV", "SG d1 w11", "K-fold 10", "Mean centre", "PLS 5 LV"]) {
    await page.getByLabel("Step", { exact: true }).selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(node(page, "PLS 5 LV")).toHaveCount(1);

  // Step 2: the other regressions off the centring, from the Regression heading.
  const centre = node(page, "Mean centre");
  // The saved chain re-renders the canvas; its port is there once it has.
  await expect(centre.locator(".react-flow__handle-right")).toBeVisible();
  await branch(page, centre, "Ridge α1", { dx: -300, dy: 110 });
  await branch(page, centre, "Elastic Net α1", { dx: -120, dy: 140 });
  await branch(page, centre, "SVR rbf 5 PC", { dx: 60, dy: 170 });
  await branch(page, centre, "OPLS 1+1", { dx: 240, dy: 200 });
  await page.getByRole("button", { name: "Save", exact: true }).click();
  for (const label of ["Ridge α1", "Elastic Net α1", "SVR rbf 5 PC", "OPLS 1+1"]) {
    await expect(node(page, label)).toHaveCount(1);
  }
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.getByTestId("node-complete")).toHaveCount(10, { timeout: 300_000 });

  // Each on fat, and its parameter set near its curve's minimum, in the
  // inspector. PLS first: the selection in step 3 is taken from it.
  const header = page.getByTestId("analysis-header");
  for (const [label, fields, expected] of [
    ["PLS 5 LV", { Target: "fat" }, "PLS on fat 5 components"],
    ["Ridge α1", { Target: "fat", Alpha: "3e-5" }, "Ridge on fat"],
    ["Elastic Net α1", { Target: "fat", Alpha: "0.001" }, "Elastic Net on fat"],
    ["OPLS 1+1", { Target: "fat", "N Orthogonal": "4" }, "OPLS on fat 5 components"],
    ["SVR rbf 5 PC", { Target: "fat", C: "50" }, "SVR on fat 5 components"],
  ] as const) {
    await outline(page).getByRole("button", { name: new RegExp(label.replace("+", "\\+")) }).first().dblclick();
    await apply(page, fields);
    await expect(header).toContainText(expected, { timeout: 300_000 });
  }
  await expect(page.getByRole("region", { name: "RMSECV" })).toContainText(/fitted at C = 50/, {
    timeout: 300_000,
  });
  await shoot(page, "ex-compare-svr");

  // Step 3: MLR on the PLS's VIP selection.
  await outline(page).getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  await page.getByLabel("Variable importance view").selectOption("selection");
  await expect(page.getByTestId("selection-count")).toHaveText("22 of 100");
  await page.getByRole("button", { name: "Apply selection" }).click();
  await canvas(page);
  const selection = node(page, "22 variables kept");
  await expect(selection).toHaveCount(1, { timeout: 120_000 });
  await expect(selection.locator(".react-flow__handle-right")).toBeVisible();
  await branch(page, selection, "MLR", { dx: 200, dy: 110 });
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await outline(page).getByRole("button", { name: /^MLR · / }).first().dblclick();
  await apply(page, { Target: "fat" });
  await expect(header).toContainText("MLR on fat 240 × 22", { timeout: 300_000 });
  await canvas(page);
  await shoot(page, "ex-compare-pipeline");
});
