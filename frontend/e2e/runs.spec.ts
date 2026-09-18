import { expect, test } from "@playwright/test";

/** Runs, against a project whose nodes have never been computed (8767).
 *
 * A run only advances if there is work to do. On the seeded project every
 * array is already on disk, so a run hits the cache and finishes before the
 * screen can render a second frame - which is correct, and useless for
 * watching one. These have their own project for that reason.
 *
 * Ordering matters within the file and Playwright honours it: cancelling
 * leaves the nodes that finished on disk, and the failing run afterwards still
 * fails, because the branch it fails on is not one of them.
 *
 * Real work needs a real budget: 3,000 x 1,200 through four branches and a
 * ten-fold split is about ten seconds cold, and Playwright's default 30 s cap
 * is for tests that only click.
 *
 * **What these do not assert.** Progress counted per node rather than
 * interpolated, cancellation bounded by one node, what a cancelled run keeps,
 * a failure naming the node it stopped at - all of that is run mechanics, and
 * `tests/test_jobs.py` proves it deterministically, driven by
 * `threading.Event` rather than by sleeps. Re-proving it here means racing a
 * live process through a hundred-millisecond DOM poll, and every flake this
 * suite has had lived in exactly that overlap. What only a browser can show is
 * that the screen is *wired* to those mechanics: that the status bar, the tab
 * badge and the node carry the same run, that cancelling from the UI reaches
 * the cancelled state, and that the canvas marks the node the executor named.
 * That is what is left here.
 */

test.describe.configure({ timeout: 180_000 });

test("a node that has never been run says so, rather than loading forever", async ({ page }) => {
  // First in the file on purpose: nothing has run yet, so every node endpoint
  // answers 404. Both screens used to render that as a loading message that
  // never resolved (#181).
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });

  await outline.getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  const results = page.getByTestId("cannot-load");
  await expect(results).toBeVisible();
  await expect(results).toContainText("Nothing to show yet");
  await expect(results).toContainText("has no fitted result yet");
  await expect(page.getByText("Loading results…")).toHaveCount(0);

  await outline.getByRole("button", { name: /^SNV/ }).first().dblclick();
  const spectra = page.getByTestId("cannot-load");
  await expect(spectra).toBeVisible();
  await expect(spectra).toContainText("has no result yet");
  await expect(page.getByText("Loading spectra…")).toHaveCount(0);
});

test("a run shows in all three places, and cancelling stops it", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();

  // `.status` is the run status bar; a second role="status" carries the stale
  // banner, and DOM order puts that one first.
  const status = page.locator(".status");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  // Watched while the run is on, not demanded at an instant. Each of these is
  // present for as long as the thing it describes lasts, which is a fraction
  // of a run: an assertion that arrives a moment late finds a finished run and
  // fails on a machine being quick.
  let sawStatus = false;
  let sawTabProgress = false;
  let sawRunningNode = false;
  let cancellable = false;

  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    // Cancel is offered exactly while the job is queued or running, so it is
    // the application's own answer to "is a run on right now".
    cancellable = Boolean(await status.getByRole("button", { name: "Cancel" }).count());

    if (!sawTabProgress && (await page.getByTestId("tab-progress").count())) sawTabProgress = true;
    if (!sawRunningNode && (await page.getByTestId("node-running").count())) sawRunningNode = true;
    if (!sawStatus && cancellable) {
      const text = (await status.innerText()) || "";
      if (text.trim() && !/^Idle/.test(text)) sawStatus = true;
    }

    if (cancellable && sawStatus && sawRunningNode && sawTabProgress) break;
    // Over means it was on and now is not. Without `sawStatus` as that memory,
    // the first turn of the loop can land in the beat between the job being
    // submitted and Cancel rendering and conclude the run finished before it
    // started - measured, that lost four runs in five.
    if (!cancellable && sawStatus) break;
    await page.waitForTimeout(100);
  }

  const observed = `status=${sawStatus} tab=${sawTabProgress} node=${sawRunningNode}`;
  expect(sawStatus, `the status bar carried the run - ${observed}`).toBe(true);
  expect(sawTabProgress, `the tab carried the run - ${observed}`).toBe(true);
  expect(sawRunningNode, `a node showed as running - ${observed}`).toBe(true);
  expect(cancellable, `the run was still on to be cancelled - ${observed}`).toBe(true);

  await status.getByRole("button", { name: "Cancel" }).click();
  await expect(status).toContainText("Cancelled");

  // Cancelled means stopped: the bar does not move on afterwards, and the tab
  // stops claiming a run. What a cancelled run *keeps* is #85's subject and is
  // asserted where it can be seen exactly - `tests/test_jobs.py`.
  const frozen = await page.locator(".status .prog i").first().getAttribute("style");
  await page.waitForTimeout(2_000);
  await expect(status).toContainText("Cancelled");
  expect(await page.locator(".status .prog i").first().getAttribute("style")).toBe(frozen);
  await expect(page.getByTestId("tab-progress")).toHaveCount(0);
});

test("the failure names its cause, marks the node, and shows no trace", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const failure = page.getByTestId("run-failed");
  // The failing branch is last in topological order, so the banner arrives
  // only after four PCA branches and a ten-fold PLS on 3,000 x 1,200 have
  // run cold. That is about ten seconds here and has taken over sixty on a
  // slow Windows runner (#202's first CI run). The budget is the file's,
  // not a wall-clock guess: this waits on the run's own terminal state.
  await expect(failure).toBeVisible({ timeout: 150_000 });
  await expect(failure).toContainText("RUN FAILED");

  // The kernel's own sentence. `decomposition.py` refuses to return fewer
  // components than asked; the rank is not asserted exactly because a centred
  // matrix read back as float32 reports one more than it has (#101).
  await expect(failure).toContainText(/components were asked of a matrix of rank \d+/);
  await expect(failure).not.toContainText("Traceback");
  await expect(page.locator(".status")).toContainText(/rank \d+/);

  // --fail is semantic and separate from the data palette on purpose: a
  // failing thing must never read as a red spectrum.
  const colours = await page.evaluate(() => {
    const style = getComputedStyle(document.querySelector(".app")!);
    return {
      fail: style.getPropertyValue("--fail").trim(),
      series: ["d1", "d2", "d3", "d4", "d5", "d6"].map((token) =>
        style.getPropertyValue(`--${token}`).trim(),
      ),
    };
  });
  expect(colours.series).not.toContain(colours.fail);
});

test("the canvas marks the node that failed, and the ones that never ran", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();

  // The node the executor named, not the last one to report progress: those
  // are different nodes, and `ExecutorError` carries the id so the canvas does
  // not have to parse it back out of English.
  const failed = page.getByTestId("node-failed").first();
  await expect(failed).toBeVisible();
  await expect(failed).toContainText(/rank \d+/);
  await expect(page.getByTestId("node-not_run").first()).toBeVisible();

  // Form, not only colour: a left stripe for failed, dashed for never-run.
  const border = (state: string) =>
    page
      .getByTestId(`node-${state}`)
      .first()
      .evaluate((element) => {
        const style = getComputedStyle(element);
        return { style: style.borderTopStyle, left: style.borderLeftWidth };
      });
  expect((await border("failed")).left).toBe("3px");
  expect((await border("not_run")).style).toBe("dashed");
});

test("a train/test split runs, and its PLS reports on the held-out set", async ({ page }) => {
  // #183. Last in the file: it rewrites the pipeline, which the tests above
  // read as seeded. The recipe is changed through the API - the split's spec
  // swapped and the failing branch dropped, so the run can succeed - and the
  // screen is what is asserted on: the split's form, the run, and the PLS
  // tab's held-out count and P-suffixed metrics.
  const headers = { Authorization: "Bearer e2e-token" };
  const pipeline = (await (await page.request.get("/api/pipelines/current", { headers })).json()) as {
    nodes: { id: string; spec?: Record<string, unknown> }[];
  };
  const failing = new Set(["range_e", "centre_e", "pca_e"]);
  const nodes = pipeline.nodes
    .filter((node) => !failing.has(node.id))
    .map((node) =>
      node.id === "split_d"
        ? { ...node, spec: { kind: "train_test", test_size: 0.25, seed: 42 } }
        : node,
    );
  const saved = await page.request.put("/api/pipelines/current", { headers, data: { nodes } });
  expect(saved.ok()).toBe(true);

  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /Train\/test 25%/ }).first().dblclick();
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await expect(inspector.getByLabel("Test Size")).toHaveValue("0.25");

  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 150_000 });

  await outline.getByRole("button", { name: /PLS 5 LV/ }).first().dblclick();
  // 3,000 synthetic samples: ceil(0.25 * 3000) held out, the rest calibrated.
  await expect(page.getByText("2250 calibration · 750 held out")).toBeVisible();
  await expect(page.getByTestId("metric-RMSEP")).not.toHaveText("—");
  await expect(page.getByTestId("metric-RMSECV")).toHaveText("—");
});

test("a foldable chain draws its coefficients on the raw axis", async ({ page }) => {
  // #184. The seeded chains all carry an SNV, so the raw-axis vector is
  // refused there by name (analysis.spec.ts). A branch of one mean centre is
  // foldable; added through the API, run from the UI, read off the plot.
  const headers = { Authorization: "Bearer e2e-token" };
  const pipeline = (await (await page.request.get("/api/pipelines/current", { headers })).json()) as {
    nodes: { id: string }[];
  };
  const nodes = [
    ...pipeline.nodes,
    { id: "centre_f", type: "preprocess", inputs: ["source"], step: { kind: "mean_centre" } },
    {
      id: "pls_f",
      type: "estimator",
      inputs: ["centre_f"],
      spec: { kind: "pls", n_components: 5, algorithm: "nipals", target: "fat" },
    },
  ];
  const saved = await page.request.put("/api/pipelines/current", { headers, data: { nodes } });
  expect(saved.ok()).toBe(true);

  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Run pipeline" }).click();
  await expect(page.locator(".status")).toContainText("Done", { timeout: 150_000 });

  // Two nodes are labelled "PLS 5 LV · fat"; the outline follows the
  // pipeline's order and the new one was appended last.
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PLS 5 LV/ }).last().dblclick();
  await page.getByLabel("Variable importance view").selectOption("coefficients");
  await expect(page.getByTestId("coefficients-plot")).toBeVisible();
  await expect(page.getByTestId("coefficients-unavailable")).toHaveCount(0);

  // One coefficient per raw variable: 1,200 on the synthetic dataset, which
  // is the dataset's count and not the node's.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const plot = document.querySelector("[data-testid=coefficients-plot]") as HTMLElement & {
          data?: { x?: number[] }[];
        };
        return plot.data?.[0]?.x?.length ?? 0;
      }),
    )
    .toBe(1200);
});

test("the outline lists every run, and one opens to what it ran", async ({ page }) => {
  // #209. This file's tests have each run the pipeline at least once, so by
  // now the project has a history rather than a single run - which is the
  // thing the outline drew one row for however many there were.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  const runs = outline.getByRole("button", { name: /^Run \d+/ });
  // Polled, not counted once: `count()` does not wait, and the outline draws
  // these from a query that has to resolve first. Counting a frame before it
  // did is what failed on the Windows runner, and it is the flake this suite
  // keeps finding - an assertion about a moment rather than about a state.
  await expect.poll(() => runs.count()).toBeGreaterThan(1);
  const count = await runs.count();

  // The served history is what is drawn: same count, newest first.
  const served = await page.evaluate(async () => {
    const response = await fetch("/api/experiments", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    return (await response.json()) as { experiment_id: string; started_at: string }[];
  });
  expect(served.length).toBe(count);
  expect([...served].sort((a, b) => b.started_at.localeCompare(a.started_at))[0].experiment_id).toBe(
    served[0].experiment_id,
  );

  // Opening the newest shows the record: what it ran, against what, what it
  // scored, where - not the placeholder the experiment tab used to reach.
  await runs.first().dblclick();
  const view = page.getByTestId("experiment-view");
  await expect(view).toBeVisible();
  await expect(view).toContainText("What it ran");
  await expect(view).toContainText("Against what");
  await expect(view).toContainText("What it scored");
  await expect(page.getByTestId("run-pipeline").locator("tbody tr")).not.toHaveCount(0);
  // The placeholder the experiment tab used to reach is gone.
  await expect(page.getByText("view — built in a later issue")).toHaveCount(0);
});

test("two runs compare step by step, and the differing node is named", async ({ page }) => {
  // #215. The tests above each changed the pipeline before running it, so this
  // project's history holds runs of different recipes - which is the case the
  // comparison exists for. The pair is chosen from the served history rather
  // than assumed, because which two differ depends on what ran above.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  const runs = outline.getByRole("button", { name: /^Run \d+/ });
  await expect.poll(() => runs.count()).toBeGreaterThan(1);

  const served = await page.evaluate(async () => {
    const response = await fetch("/api/experiments", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    return (await response.json()) as { experiment_id: string; pipeline_hash: string }[];
  });
  // The outline is the served order, so an index into one is an index into the
  // other. The first run whose recipe differs from the newest is the partner.
  const other = served.findIndex((row) => row.pipeline_hash !== served[0].pipeline_hash);
  expect(other, "this project's history holds runs of more than one recipe").toBeGreaterThan(0);

  await runs.first().dblclick();
  await expect(page.getByTestId("experiment-view")).toBeVisible();
  await page
    .getByTestId("compare-with")
    .selectOption({ value: served[other].experiment_id });

  const view = page.getByTestId("lineage-view");
  await expect(view).toBeVisible();
  // Two different recipes differ by at least one node, and the pill counts the
  // same nodes the table marks - the summary is not a second opinion.
  const summary = page.getByTestId("lineage-summary");
  await expect(summary).toHaveText(/^\d+ nodes? differ$/);
  const counted = Number((await summary.innerText()).split(" ")[0]);
  expect(counted).toBeGreaterThan(0);

  const marked = view.locator(
    '[data-testid="lineage-node-changed"], [data-testid="lineage-node-added"], [data-testid="lineage-node-removed"]',
  );
  await expect(marked).toHaveCount(counted);
  // Named, not merely counted: the row carries the node's own id.
  await expect(marked.first()).toHaveAttribute("data-node", /.+/);

  // The steps the two share are still drawn, because a diff that hides what
  // matched makes the reader reconstruct the recipe to read the difference.
  await expect(view.getByTestId("lineage-node-unchanged").first()).toBeVisible();
  await expect(view).toContainText("What each scored");
});

test("a model saved from the analysis tab appears in the outline", async ({ page }) => {
  // #219. The project has a fitted PLS by now - the tests above ran one under
  // a train/test split and added a second on a foldable chain - so this saves
  // one and reads the registry back, rather than fitting anything of its own.
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });

  // Before: the section says what would put one there.
  await expect(outline).toContainText("No models yet");

  await outline.getByRole("button", { name: /PLS 5 LV/ }).last().dblclick();
  await expect(page.getByTestId("analysis-header")).toBeVisible();

  await page.getByLabel("Model name").fill("Fat, mean centre only");
  await page.getByTestId("save-model").click();
  await expect(page.getByTestId("model-saved")).toBeVisible();

  // The outline lists it, with the one figure a saved regression carries.
  const row = outline.getByRole("button", { name: /^Fat, mean centre only/ });
  await expect(row).toBeVisible();
  await expect(outline).not.toContainText("No models yet");

  // And one opens to its record: what it scored, where it came from, and the
  // file it points at. The artifact is a path, never contents - the database
  // holds the reference (PROPOSAL.md section 11).
  await row.dblclick();
  const view = page.getByTestId("model-view");
  await expect(view).toBeVisible();
  await expect(view).toContainText("What it scored");
  await expect(view).toContainText("Where it came from");
  await expect(view).toContainText(/models\/.+\.cwmodel/);
  await expect(view).toContainText("sha256:");
  await expect(page.getByTestId("model-metric-RMSEC")).not.toHaveText("—");

  // The registry is what the server serves, not what the screen remembers.
  const served = await page.evaluate(async () => {
    const response = await fetch("/api/models", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    return (await response.json()) as { name: string; artifact_path: string }[];
  });
  expect(served.map((model) => model.name)).toEqual(["Fat, mean centre only"]);
  expect(served[0].artifact_path).toMatch(/^models\/.+\.cwmodel$/);
});
