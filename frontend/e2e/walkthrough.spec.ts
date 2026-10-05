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
  //    anything is committed. Thirty by twenty-four because a PCA of five
  //    components needs a matrix that really has five: see `spectra-file.ts`,
  //    which had two until #101 made the rank tolerance honest enough to say so.
  await page.getByLabel("Choose file").setInputFiles({
    name: "spectra.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(spectraCsv(30, 24)),
  });
  await expect(page.getByText("spectra.csv")).toBeVisible();
  await expect(page.getByLabel("Delimiter")).toHaveValue(",");
  await expect(page.getByLabel("Orientation")).toHaveValue("samples_in_rows");

  // 3. The dataset is loaded only after the preview is confirmed.
  await expect(page.getByRole("tab", { name: /spectra/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Import 30 × 24" }).click();
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

test("two samples are excluded, the run uses what is left, and v1 comes back", async ({
  page,
}) => {
  // #270. Runs after the walkthrough, on the project it imported into: the
  // only server whose project a test is allowed to change.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /v1 · 30×24/ }).dblclick();

  await page.getByLabel("Select row 2", { exact: true }).check();
  await page.getByLabel("Select row 5", { exact: true }).check();
  await page.getByRole("button", { name: "Exclude 2 selected" }).click();

  // A second version, without them, and the pipeline moved onto it.
  const derived = outline.getByRole("button", { name: /v2 · 28×24/ });
  await expect(derived).toBeVisible();
  await derived.dblclick();
  await expect(page.getByTestId("excluded-pill")).toHaveText("2 excluded from v1");
  await expect(page.getByTestId("excluded-ids")).toHaveText("Left out of v1: A002, A005");
  await expect(page.getByRole("cell", { name: "A002", exact: true })).toHaveCount(0);

  // The run is on the 28 that are left.
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 60_000 });
  await expect
    .poll(async () => {
      const response = await page.request.get("/api/results/pca", {
        headers: { Authorization: "Bearer e2e-token" },
      });
      return response.ok() ? ((await response.json()).samples as unknown[]).length : 0;
    })
    .toBe(28);

  // Undone by putting the source back on v1.
  await page.getByRole("button", { name: "Restore v1" }).click();
  await expect
    .poll(async () => {
      const response = await page.request.get("/api/pipelines/current", {
        headers: { Authorization: "Bearer e2e-token" },
      });
      return (await response.json()).nodes[0].version_id as string;
    })
    .not.toBe(await sourceOfDerived(page));
  await expect(page.getByRole("button", { name: "Restore v1" })).toHaveCount(0);
});

/** The version id the outline's v2 row stands for, read from the server. */
async function sourceOfDerived(page: Page): Promise<string> {
  const projects = await (
    await page.request.get("/api/projects", { headers: { Authorization: "Bearer e2e-token" } })
  ).json();
  const datasets = await (
    await page.request.get(`/api/projects/${projects[0].project_id}/datasets`, {
      headers: { Authorization: "Bearer e2e-token" },
    })
  ).json();
  return datasets[0].versions[1].version_id as string;
}

test("the new smoothers are added from the step list and run", async ({ page }) => {
  // #271, on the walkthrough's project, whose chain ends in its PCA. Until
  // #296 the side list appended below that estimator and the run failed with
  // a KeyError; now it branches from the PCA's input.
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  const before = await page.locator(".react-flow__node").count();
  for (const step of ["Median w5", "Whittaker λ100"]) {
    await page.getByLabel("Step").selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 2);
  await expect(page.getByText("window 5", { exact: true })).toBeVisible();
  await expect(page.getByText("lambda 100", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  // The drafts clear once the server holds them. Running before that ran the
  // old recipe - #291, a race of its own and not what this test is about.
  await expect(page.getByText(/^No steps yet/)).toBeVisible();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 60_000 });
  await expect(page.getByTestId("node-complete")).toHaveCount(before + 2);
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await expect(outline.getByRole("button", { name: /Median w5/ })).toBeVisible();
  await expect(outline.getByRole("button", { name: /Whittaker λ100/ })).toBeVisible();
});

test("a run started while a save is in flight runs what was saved", async ({ page }) => {
  // #291. The save is held back a second, and Run is clicked straight after
  // Save: before the fix the run executed the recipe as it was before the PUT
  // landed, reported "Done", and the new node stayed not run.
  await page.goto("/?token=e2e-token");
  await page.route("**/api/pipelines/current", async (route) => {
    if (route.request().method() === "PUT") await new Promise((resolve) => setTimeout(resolve, 1000));
    await route.continue();
  });
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  const before = await page.locator(".react-flow__node").count();
  await page.getByLabel("Step").selectOption("Autoscale");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 60_000 });
  await expect(page.getByTestId("node-complete")).toHaveCount(before + 1);
});

test("a PCR runs on the dataset's target and reads as a regression without VIP", async ({
  page,
}) => {
  // #272. Written through the pipeline's PUT beside the walkthrough's PCA,
  // because the side list offers no estimator that needs a target.
  await page.goto("/?token=e2e-token");
  const auth = { Authorization: "Bearer e2e-token" };
  const pipeline = await (await page.request.get("/api/pipelines/current", { headers: auth })).json();
  const nodes = pipeline.nodes as { id: string; type: string; inputs: string[] }[];
  const pca = nodes.find((node) => node.type === "estimator")!;
  nodes.push({
    id: "pcr",
    type: "estimator",
    inputs: pca.inputs,
    spec: { kind: "pcr", n_components: 3, target: "moisture" },
  } as (typeof nodes)[number]);
  const saved = await page.request.put("/api/pipelines/current", {
    headers: { ...auth, "Content-Type": "application/json" },
    data: { nodes },
  });
  expect(saved.status()).toBe(200);

  await page.reload();
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 60_000 });
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCR 3 PC · moisture/ }).dblclick();
  await expect(page.getByTestId("analysis-header")).toContainText("PCR on moisture 3 components");
  await expect(page.getByLabel("Variable importance view").locator("option")).toHaveText([
    "Coefficients, raw axis",
  ]);
  await expect(page.getByRole("region", { name: "Predicted vs measured" })).toBeVisible();
});

test("a flagged sample is excluded from the outlier table, rerun, and named in lineage", async ({
  page,
}) => {
  // #279, on the PCR the test above added. Excluding writes a derived version
  // (#270's endpoint), the run starts by itself, and the comparison of the two
  // runs names the source as what changed.
  const auth = { Authorization: "Bearer e2e-token" };
  await page.goto("/?token=e2e-token");
  const original = (await (await page.request.get("/api/pipelines/current", { headers: auth })).json())
    .nodes as { id: string; type: string; version_id?: string }[];
  const source = original.find((node) => node.type === "source")!;
  const served = await (await page.request.get("/api/results/pcr", { headers: auth })).json();
  const n = (served.samples as unknown[]).length;
  const block = await (await page.request.get("/api/results/pcr/outliers", { headers: auth })).json();
  const flags = block.flags as { index: number }[];
  expect(flags.length, "a 30-sample calibration flags something").toBeGreaterThan(0);
  const sample = served.samples[flags[0].index].sample_id as string;

  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCR 3 PC · moisture/ }).dblclick();
  const row = page.getByTestId("outliers-row");
  await row.scrollIntoViewIfNeeded();
  await row.getByLabel(`Exclude ${sample}`, { exact: true }).check();
  await row.getByRole("button", { name: "Exclude 1 and rerun" }).click();

  try {
    // The rerun is on the version without it.
    await expect
      .poll(
        async () => {
          const response = await page.request.get("/api/results/pcr", { headers: auth });
          return response.ok() ? ((await response.json()).samples as unknown[]).length : 0;
        },
        { timeout: 60_000 },
      )
      .toBe(n - 1);
    await expect(page.locator(".status")).toContainText("Done", { timeout: 60_000 });

    // And lineage says what changed between the two runs: the source's version.
    await outline.getByRole("button", { name: /^Run \d+/ }).first().dblclick();
    await expect(page.getByTestId("experiment-view")).toBeVisible();
    await page.getByTestId("compare-with").selectOption({ index: 1 });
    const changed = page.getByTestId("lineage-view").getByTestId("lineage-node-changed");
    await expect(changed).toHaveCount(1);
    await expect(changed).toHaveAttribute("data-node", source.id);
    await expect(changed).toContainText("version_id");
  } finally {
    // Back onto the version the walkthrough imported.
    await page.request.put("/api/pipelines/current", {
      headers: { ...auth, "Content-Type": "application/json" },
      data: { nodes: original },
    });
  }
});
