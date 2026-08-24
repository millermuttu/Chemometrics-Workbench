import { expect, test, type Page } from "@playwright/test";

/** Phase 1.1's exit criterion, written as a test.
 *
 * The sub-phase is finished when this passes, not when the screens look done -
 * which is the point of writing it as a test rather than judging by eye at the
 * end. It walks the whole path against the stub server: an empty project, an
 * import with its detection preview, the dataset loaded, a pipeline assembled
 * through the step list, a run watched through its real lifecycle, and the
 * scores read back and compared against the numbers the kernel produced.
 *
 * Two more paths follow it: one that cancels a running job, one that provokes
 * a failure and reads the cause.
 */

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
  await page.goto("/?token=e2e-token&empty");
  await expect(page.getByRole("heading", { name: "This project is empty" })).toBeVisible();
  await page.getByRole("button", { name: "Import data" }).click();

  // 2. Import, with the detection stated before anything is committed.
  await page.getByRole("button", { name: "Use the example file" }).click();
  await expect(page.getByText("tecator.txt")).toBeVisible();
  await expect(page.getByLabel("Delimiter")).toHaveValue("whitespace");
  await expect(page.getByLabel("Orientation")).toHaveValue("samples_in_rows");
  await expect(page.getByText(/reconstructed as 100 evenly spaced points/)).toBeVisible();

  // 3. The dataset is loaded only after the preview is confirmed.
  await expect(page.getByRole("tab", { name: /tecator_raw/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Import 240 × 100" }).click();
  await expect(page.getByRole("tab", { name: /tecator_raw/ })).toBeVisible();
  await expect(page.getByRole("cell", { name: "C001" })).toBeVisible();

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

  // 5. The run, watched through its real lifecycle. The result must not have
  //    been there from the start: no results tab is open yet.
  await expect(page.getByTestId("scores-plot")).toHaveCount(0);
  const status = page.getByRole("status").first();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(status).toContainText("Queued");

  const seen: number[] = [];
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    const width = await page.locator(".status .prog i").first().getAttribute("style");
    const percent = Number(/width:\s*([\d.]+)%/.exec(width ?? "")?.[1] ?? "-1");
    if (percent >= 0 && percent !== seen.at(-1)) seen.push(percent);
    if (await status.getByText("Complete").count()) break;
    await page.waitForTimeout(150);
  }
  expect(seen.length, `progress advanced through ${seen.join(", ")}`).toBeGreaterThan(2);
  expect(seen).toEqual([...seen].sort((a, b) => a - b));
  expect(seen.at(-1)).toBe(100);

  // 6. The scores, read back and compared against what the kernel produced.
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();

  const served = await servedScores(page, "pca_a");
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

test("the cancel path: a running job stops where it stood", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const status = page.getByRole("status").first();
  await expect(status).toContainText("Preprocessing: SNV", { timeout: 10_000 });
  await expect(page.getByTestId("tab-progress")).toBeVisible();

  await status.getByRole("button", { name: "Cancel" }).click();
  await expect(status).toContainText("Cancelled");

  const frozen = await page.locator(".status .prog i").first().getAttribute("style");
  await page.waitForTimeout(2_000);
  expect(await page.locator(".status .prog i").first().getAttribute("style")).toBe(frozen);
  await expect(page.getByTestId("tab-progress")).toHaveCount(0);
});

test("the failure path: the run names a cause and shows no trace", async ({ page }) => {
  await page.goto("/?token=e2e-token&failrun");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const failure = page.getByTestId("run-failed");
  await expect(failure).toBeVisible({ timeout: 20_000 });
  await expect(failure).toContainText("5 components were asked of a matrix of rank 4");
  await expect(failure).not.toContainText("Traceback");
  await expect(page.getByRole("status").first()).toContainText("rank 4");
});
