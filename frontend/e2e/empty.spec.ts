import path from "node:path";

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

/** #175. A step added from the drop menu had its position only in the tab:
 * Save wrote the recipe and never the layout, so a reload moved the node to a
 * generated place. Runs after the import above, on the source it left. */
test("a step added from the drop menu keeps its position once saved", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(1);

  const port = page.locator('.react-flow__node[data-id="source"] .react-flow__handle-right');
  const start = (await port.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(start.x + 260, start.y + 220, { steps: 12 });
  await page.mouse.up();
  await page.getByTestId("add-step-menu").getByRole("menuitem", { name: "Autoscale" }).click();

  const drawn = await page
    .locator('.react-flow__node[data-id="autoscale"]')
    .evaluate((element) => {
      const [x, y] = getComputedStyle(element)
        .transform.match(/-?\d+\.?\d*/g)!
        .slice(-2)
        .map(Number);
      return { x: Math.round(x), y: Math.round(y) };
    });

  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect
    .poll(async () => {
      const state = await page.request.get("/api/pipelines/current/state", {
        headers: { Authorization: "Bearer e2e-token" },
      });
      const stored = (await state.json()).layout.autoscale;
      return stored && { x: Math.round(stored.x), y: Math.round(stored.y) };
    })
    .toEqual(drawn);
});

test("a zip of OPUS files previews as one dataset with a block to choose", async ({ page }) => {
  // #187. Two soil spectra from opusreader2's sample data, one per file. The
  // preview names the block it read, offers the others, and states the axis
  // in wavenumbers; nothing is committed, so the project stays as the tests
  // after this one expect it.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByLabel("Choose file").setInputFiles(
    path.resolve(import.meta.dirname, "../../tests/fixtures/readers/opus/soil_pair.zip"),
  );
  await expect(page.getByText("soil_pair.zip")).toBeVisible();
  await expect(page.getByLabel("Block")).toHaveValue("AB");
  await expect(page.getByLabel("Block").locator("option")).toHaveText([
    "AB · absorbance",
    "ScSm · sample single channel",
    "ScRf · reference single channel",
  ]);
  await expect(page.getByText("wavenumber_cm-1")).toBeVisible();
  await expect(page.getByRole("button", { name: "Import 2 × 3578" })).toBeVisible();
  await page.getByRole("button", { name: "Cancel" }).click();
});

test("a MAT-file previews its matrix, turned the right way, and another can be chosen", async ({
  page,
}) => {
  // #284. A slice of MLNIRdata (CC-BY-4.0), stored as MATLAB stores it -
  // variables down the rows. The preview turns it, finds the axis vector and
  // the density, and asks again when another matrix is chosen. Nothing is
  // committed.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Import…" }).click();
  await page.getByLabel("Choose file").setInputFiles(
    path.resolve(import.meta.dirname, "../../tests/fixtures/readers/mat/mlnir_slice.mat"),
  );
  await expect(page.getByText("mlnir_slice.mat")).toBeVisible();
  await expect(page.getByLabel("Matrix")).toHaveValue("matrixXNirSpectrumData");
  await expect(page.getByLabel("Orientation")).toHaveValue("samples_in_columns");
  await expect(page.getByLabel("Axis from")).toHaveValue("matrixXNirSpectrumDataAxis");
  await expect(page.getByText("matrixYNirPropertyDensityNormalized")).toBeVisible();
  await expect(page.getByRole("button", { name: "Import 12 × 53" })).toBeVisible();

  await page.getByLabel("Matrix").selectOption("matrixXNirSpectrumDerivative");
  await expect(page.getByRole("button", { name: "Import 12 × 52" })).toBeVisible();
  await expect(page.getByLabel("Axis from")).toHaveValue("matrixXNirSpectrumDerivativeAxis");
  await page.getByRole("button", { name: "Cancel" }).click();
});

/** #258. The canvas fitted the graph only when it mounted, so steps added to a
 * fresh pipeline walked off the right-hand edge. Nothing here is saved. */
test("steps added to a fresh pipeline stay inside the canvas", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  const before = await page.locator(".react-flow__node").count();

  for (const step of ["SNV", "SG d1 w11", "Mean centre", "PCA"]) {
    await page.getByLabel("Step").selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  const nodes = page.locator(".react-flow__node");
  await expect(nodes).toHaveCount(before + 4);

  const canvas = (await page.getByTestId("pipeline-canvas").boundingBox())!;
  await expect
    .poll(async () => {
      const boxes = await Promise.all((await nodes.all()).map((one) => one.boundingBox()));
      return boxes.every(
        (box) =>
          box !== null &&
          box.x >= canvas.x &&
          box.y >= canvas.y &&
          box.x + box.width <= canvas.x + canvas.width &&
          box.y + box.height <= canvas.y + canvas.height,
      );
    })
    .toBe(true);
  await expect(page.locator(".react-flow__controls-fitview")).toBeVisible();
});

/** #258. The add-step menu opened downwards from the drop, so a drop near the
 * bottom of the window put its last entries out of reach. */
test("a connector dropped near the bottom opens a menu that fits", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  const port = page.locator('.react-flow__node[data-id="source"] .react-flow__handle-right');
  const start = (await port.boundingBox())!;
  const bottom = page.viewportSize()!.height - 20;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(start.x + 80, bottom, { steps: 12 });
  await page.mouse.up();

  const menu = page.getByTestId("add-step-menu");
  await expect(menu).toBeVisible();
  const last = menu.getByRole("menuitem").last();
  const box = (await last.boundingBox())!;
  expect(box.y + box.height).toBeLessThanOrEqual(page.viewportSize()!.height);
  await last.click({ trial: true, timeout: 2_000 });
});
