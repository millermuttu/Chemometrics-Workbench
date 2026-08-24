import { expect, test, type Page } from "@playwright/test";

/** From an empty project to a loaded dataset, with one detection corrected
 * and the failure state reached - all against the stub server's fixtures. */

async function ready(page: Page, query = "") {
  await page.goto(`/?token=e2e-token${query}`);
  await expect(page.getByText("Tecator meat study")).toBeVisible();
}

test("an empty project offers the one action that matters", async ({ page }) => {
  await ready(page, "&empty");
  await expect(page.getByRole("heading", { name: "This project is empty" })).toBeVisible();

  await page.getByRole("button", { name: "Import data" }).click();
  await expect(page.getByRole("heading", { name: "Import data" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Import/ })).toBeVisible();
});

test("the preview shows what was detected, and a correction changes what would be read", async ({
  page,
}) => {
  await ready(page);
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByRole("button", { name: "Use the example file" }).click();

  await expect(page.getByText("tecator.txt")).toBeVisible();
  await expect(page.getByLabel("Delimiter")).toHaveValue("whitespace");
  await expect(page.getByLabel("Decimal")).toHaveValue(".");
  await expect(page.getByRole("button", { name: "Import 240 × 100" })).toBeVisible();

  // The reconstructed axis and the discarded columns are stated, not hidden.
  await expect(page.getByText(/reconstructed as 100 evenly spaced points/)).toBeVisible();
  await expect(page.getByText(/22 principal components/)).toBeVisible();

  // Correcting the orientation swaps what the counts mean, before anything is
  // committed - a transposed file is the common wrong guess.
  await page.getByLabel("Orientation").selectOption("samples_in_columns");
  await expect(page.getByText("corrected")).toBeVisible();
  await expect(page.getByRole("button", { name: "Import 100 × 240" })).toBeVisible();
});

test("confirming the preview opens the dataset, and nothing is committed before that", async ({
  page,
}) => {
  await ready(page);
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByRole("button", { name: "Use the example file" }).click();
  await expect(page.getByRole("tab", { name: /tecator_raw/ })).toHaveCount(0);

  await page.getByRole("button", { name: "Import 240 × 100" }).click();
  await expect(page.getByRole("tab", { name: /tecator_raw/ })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Import/ })).toHaveCount(0);

  // The dataset itself: the artboard's table, the targets as columns.
  await expect(page.getByRole("columnheader", { name: "moisture" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "C001" })).toBeVisible();
  await expect(page.getByText("Showing 60 of 240 samples.")).toBeVisible();
});

test("a failed import names the file and the setting, not a stack trace", async ({ page }) => {
  await ready(page);
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByRole("button", { name: "Import a broken file" }).click();

  const alert = page.getByRole("alert");
  await expect(alert).toContainText("READER_FAILED");
  await expect(alert).toContainText("expected a multiple of 125 values per row");
  await expect(alert).toContainText("row: 87");
  await expect(alert).not.toContainText("Traceback");

  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("button", { name: "Use the example file" })).toBeVisible();
});
