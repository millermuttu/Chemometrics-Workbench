import { expect, test, type Page } from "@playwright/test";

import { spectraCsv } from "./spectra-file";

/** From a project with nothing in it to a loaded dataset, against a real empty
 * project (8766) rather than `?empty`.
 *
 * These run on their own server because they are the tests that *change* the
 * project they run in: an import cannot share a project with the tests that
 * assume a pipeline is already there. They also carry their own file, which is
 * the other half of what `?empty` and "Use the example file" used to fake -
 * the file the user picks is now the file that is read (#99).
 */

/** Six channels, four samples, one target, and a wavelength axis in the header
 * row. Small enough to reason about, real enough for the reader to detect. */
const SPECTRA = spectraCsv(4, 6);

async function choose(page: Page, name: string, body: string) {
  await page.getByLabel("Choose file").setInputFiles({
    name,
    mimeType: "text/csv",
    buffer: Buffer.from(body),
  });
}

test("an empty project offers the one action that matters", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await expect(page.getByRole("heading", { name: "This project is empty" })).toBeVisible();

  await page.getByRole("button", { name: "Import data" }).click();
  await expect(page.getByRole("heading", { name: "Import data" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Import/ })).toBeVisible();
});

test("a failed import names the file and the cause, not a stack trace", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();

  // A file that is genuinely unreadable, rather than a flag that pretends one
  // is. One column is not a spectrum, and the reader says so by name - which
  // is the whole reason `?failrun` could go.
  await choose(page, "one-column.csv", "value\n1\n2\n3\n");

  const alert = page.getByRole("alert");
  await expect(alert).toBeVisible();
  await expect(alert).toContainText("READER_FAILED");
  await expect(alert).toContainText("one-column.csv");
  await expect(alert).not.toContainText("Traceback");

  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByLabel("Choose file")).toBeAttached();
});

test("the preview states what was read, and a correction changes what would be", async ({
  page,
}) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await choose(page, "spectra.csv", SPECTRA);

  // The file the user picked, and the detection stated before anything is
  // committed. The name is the file's own: nothing substitutes an example.
  await expect(page.getByText("spectra.csv")).toBeVisible();
  await expect(page.getByLabel("Delimiter")).toHaveValue(",");
  await expect(page.getByLabel("Decimal")).toHaveValue(".");
  await expect(page.getByLabel("Orientation")).toHaveValue("samples_in_rows");
  await expect(page.getByRole("button", { name: "Import 4 × 6" })).toBeVisible();

  // Correcting the orientation swaps what the counts mean, before anything is
  // committed - a transposed file is the common wrong guess.
  await page.getByLabel("Orientation").selectOption("samples_in_columns");
  await expect(page.getByText("corrected")).toBeVisible();
  await expect(page.getByRole("button", { name: "Import 6 × 4" })).toBeVisible();
});

test("confirming the preview opens the dataset, and nothing is committed before", async ({
  page,
}) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import data" }).click();
  await choose(page, "spectra.csv", SPECTRA);

  await expect(page.getByRole("tab", { name: /spectra/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Import 4 × 6" }).click();

  await expect(page.getByRole("tab", { name: /spectra/ })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Import/ })).toHaveCount(0);

  // The dataset itself: the artboard's table, the target as a column.
  await expect(page.getByRole("columnheader", { name: "moisture" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "A001" })).toBeVisible();

  // An import starts the pipeline it is obviously the beginning of, so the
  // project is no longer empty and the canvas has somewhere to start.
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(1);
});
