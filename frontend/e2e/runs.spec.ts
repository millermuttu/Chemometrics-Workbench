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
