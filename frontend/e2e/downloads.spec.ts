import { readFile } from "node:fs/promises";

import { expect, test, type Page } from "@playwright/test";

/** #247: the files the server serves for export and archiving are reachable
 * from the screens, not only over HTTP with a token pasted into curl. Each
 * button saves what the server serves, under the server's name for it, and a
 * refusal shows the server's sentence. */

async function openNode(page: Page, node: RegExp) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: node }).first().dblclick();
  await expect(page.getByTestId("analysis-header")).toBeVisible();
}

async function saved(page: Page, button: string) {
  const [file] = await Promise.all([page.waitForEvent("download"), page.getByTestId(button).click()]);
  return { name: file.suggestedFilename(), text: await readFile((await file.path())!, "utf8") };
}

test("a PLS result exports its JSON model and its prediction snippet", async ({ page }) => {
  await openNode(page, /PLS 5 LV/);

  const json = await saved(page, "export-json");
  expect(json.name).toBe("pls_d_model.json");
  const served = await page.request.get("/api/results/pls_d/export.json", {
    headers: { Authorization: "Bearer e2e-token" },
  });
  // The same model the server serves, stamped at a different moment.
  const model = JSON.parse(json.text);
  const fresh = await served.json();
  expect({ ...model, created_at: null }).toEqual({ ...fresh, created_at: null });

  const python = await saved(page, "export-python");
  expect(python.name).toBe("pls_d_predict.py");
  expect(python.text).toContain("def predict(X):");
});

test("an export the server refuses says why, in the server's words", async ({ page }) => {
  // The seeded chains all export; a baseline in one is the real refusal, and
  // the envelope below is its shape (docs/model-export.md section 1).
  const sentence = "the chain holds a baseline correction (asls), which an export cannot carry.";
  await page.route("**/api/results/*/export.json", (route) =>
    route.fulfill({
      status: 422,
      contentType: "application/json",
      body: JSON.stringify({ error: { code: "not_exportable", message: sentence, detail: {} } }),
    }),
  );
  await openNode(page, /PLS 5 LV/);
  await page.getByTestId("export-json").click();
  await expect(page.getByTestId("export-json-error")).toHaveText(sentence);
});

test("a PCA result offers no export: there is no regression to carry", async ({ page }) => {
  await openNode(page, /PCA 5 PC/);
  await expect(page.getByTestId("export-json")).toHaveCount(0);
  await expect(page.getByTestId("export-python")).toHaveCount(0);
});

test("an experiment saves its standalone HTML report", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /^Run 1/ }).first().dblclick();
  await expect(page.getByTestId("experiment-view")).toBeVisible();

  const report = await saved(page, "save-report");
  // The server's name for it: the start time, or `unrun` for the seeded
  // experiment, which the seed script records without one.
  expect(report.name).toMatch(/^experiment-(\d{8}-\d{4}|unrun)-[0-9a-f]{8}\.html$/);
  expect(report.text.toLowerCase()).toContain("<!doctype html>");
  expect(report.text).toMatch(/Scores, \d+ samples/);
});
