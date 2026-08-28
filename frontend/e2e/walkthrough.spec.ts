import { expect, test, type Page } from "@playwright/test";

import { spectraCsv } from "./spectra-file";

/** Phase 1.1's exit criterion, written as a test.
 *
 * The sub-phase is finished when this passes, not when the screens look done -
 * which is the point of writing it as a test rather than judging by eye at the
 * end. It walks the whole path against the **real** backend: a project with
 * nothing in it, an import of a file the user picks with its detection stated
 * before anything is committed, the dataset loaded, a pipeline assembled
 * through the step list **and saved**, a run watched through its real
 * lifecycle, and the scores read back and compared against the numbers the
 * kernel produced.
 *
 * It was `fixme` between #89 and #108, because the step list built a
 * client-side draft and no endpoint wrote a pipeline back, so the nodes it
 * assembled never reached the server. #108 added `PUT /pipelines/{id}` and the
 * Save that uses it.
 *
 * This runs on the empty project (8766), because it starts by importing.
 */

test.describe.configure({ timeout: 180_000 });

/** What the server says the results are. The walkthrough asserts the screen
 * shows these numbers, not merely that a plot exists. */
async function servedScores(page: Page, node: string) {
  return page.evaluate(async (id) => {
    const response = await fetch(`/api/results/${id}`, {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    const pca = (await response.json()) as {
      scores: number[][];
      samples: { sample_id: string }[];
      explained_variance_ratio: number[];
    };
    return {
      first: pca.scores[0],
      sample: pca.samples[0].sample_id,
      variance: pca.explained_variance_ratio[0],
    };
  }, node);
}

/** What the scores plot is actually drawing. */
async function drawnScores(page: Page) {
  return page.evaluate(() => {
    const plot = document.querySelector("[data-testid=scores-plot]") as HTMLElement & {
      data?: { x?: number[]; y?: number[]; text?: string[]; name?: string }[];
    };
    const points = (plot.data ?? []).find((trace) => trace.text?.length);
    return { x: points?.x?.[0], y: points?.y?.[0], sample: points?.text?.[0] };
  });
}

test("the whole path: empty project to a scores plot the kernel produced", async ({ page }) => {
  // 1. An empty project, with one obvious action.
  await page.goto("/?token=e2e-token");
  await expect(page.getByRole("heading", { name: "This project is empty" })).toBeVisible();
  await page.getByRole("button", { name: "Import data" }).click();

  // 2. Import the file the user picked, with the detection stated before
  //    anything is committed. Thirty samples because a PCA of five components
  //    needs a matrix that has five.
  await page.getByLabel("Choose file").setInputFiles({
    name: "spectra.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(spectraCsv(30, 12)),
  });
  await expect(page.getByText("spectra.csv")).toBeVisible();
  await expect(page.getByLabel("Delimiter")).toHaveValue(",");
  await expect(page.getByLabel("Orientation")).toHaveValue("samples_in_rows");

  // 3. The dataset is loaded only after the preview is confirmed.
  await expect(page.getByRole("tab", { name: /spectra/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Import 30 × 12" }).click();
  await expect(page.getByRole("tab", { name: /spectra/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: "A001" })).toBeVisible();

  // 4. A pipeline assembled through the step list: SNV, then Savitzky-Golay,
  //    then PCA. Direct manipulation is #51; this is 1.1's builder.
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  const before = await page.locator(".react-flow__node").count();

  for (const step of ["SNV", "SG d1 w11", "PCA"]) {
    await page.getByLabel("Step").selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 3);
  await page.getByRole("button", { name: "Validate" }).click();
  await expect(page.getByText("valid · 3 steps")).toBeVisible();

  // Saved, not merely drawn. Until #108 the recipe lived in this tab and
  // nowhere else, so a reload lost it and a run had only the source to execute.
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 3);
  await page.reload();
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 3);

  // 5. The run, through its real lifecycle. The result must not have been
  //    there from the start: no results tab is open yet, and the nodes have no
  //    arrays until this runs.
  await expect(page.getByTestId("scores-plot")).toHaveCount(0);
  await expect(page.getByTestId("node-not_run").first()).toBeVisible();

  // `.status` is the run status bar, not the stale banner that shares its role.
  const status = page.locator(".status");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  // The end state, not a frame of the middle. Thirty samples by twelve
  // channels is over in milliseconds - faster than the job poll - so asserting
  // "Queued" here would be asserting that the machine is slow. `runs.spec.ts`
  // watches progress advance, on a project seeded large enough to see it.
  await expect(status).toContainText("Done", { timeout: 60_000 });
  const width = await page.locator(".status .prog i").first().getAttribute("style");
  expect(Number(/width:\s*([\d.]+)%/.exec(width ?? "")?.[1] ?? "-1")).toBe(100);

  // Every node the walkthrough built now has its arrays, which is the thing
  // that was impossible before #108: a saved recipe is what the run executes.
  await expect(page.getByTestId("node-complete")).toHaveCount(before + 3);
  await expect(page.getByTestId("node-not_run")).toHaveCount(0);

  // 6. The scores, read back and compared against what the kernel produced.
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();

  // The node the walkthrough built, not the fixture's: `withDrafts` names it
  // after the step, so a PCA added to a fresh pipeline is `pca`.
  const served = await servedScores(page, "pca");
  const drawn = await drawnScores(page);
  expect(drawn.sample).toBe(served.sample);
  expect(drawn.x).toBeCloseTo(served.first[0], 12);
  expect(drawn.y).toBeCloseTo(served.first[1], 12);
  await expect(page.getByTestId("analysis-header")).toContainText(
    `${(served.variance * 100).toFixed(1)}%`,
  );

  // 7. Both themes, on the screen the walkthrough ends on.
  const accent = () =>
    page.evaluate(() =>
      getComputedStyle(document.querySelector(".app")!).getPropertyValue("--accent").trim(),
    );
  expect((await accent()).toUpperCase()).toBe("#0B6B62");
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  expect((await accent()).toUpperCase()).toBe("#54BFAB");
  await expect(page.getByTestId("scores-plot")).toBeVisible();
});

/* The cancel and failure paths were here in Phase 1.1, driven by the stub's
 * `?failrun`. They live in `runs.spec.ts` now, against a project seeded with a
 * branch that genuinely cannot be fitted and nothing computed - a real run to
 * cancel, and a real failure to read. Repeating them here would mean asserting
 * them on a project that has just been imported into, where a cached pipeline
 * gives a run no work to do. */
