import { expect, test, type Page } from "@playwright/test";

/** The inspector: a typed form built from the schema, a message that comes
 * from the model, and an edit that recomputes on the press that made it. */

async function selectNode(page: Page, name: RegExp) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name }).first().dblclick();
  return page.getByRole("complementary", { name: "Inspector" });
}

/** How long an `Apply and re-run` gets to reach the server (#222).
 *
 * The press validates, writes the pipeline and starts a run, and nothing on
 * screen has to change before the `PUT` lands - so every check below polls.
 * `expect.poll` defaults to five seconds, which a loaded macOS runner with a
 * re-run in flight misses, and on #221 it did.
 *
 * **A restore that misses its budget is worse than an ordinary failure.** The
 * seeded project is one directory that every spec in this project shares, so a
 * test that edits and restores is borrowing state. When the restore times out
 * the edit stays, and the next test reads a node it never touched and fails
 * for a reason that is not its own. That is the second red test on #221.
 *
 * Not retries: `playwright.config.ts` says why, and a green check should mean
 * the suite passed rather than that it passed on the third attempt.
 */
const APPLIED = { timeout: 30_000 };

/** The window length the server holds for `savgol`, which is the claim an edit
 * makes: what is on disk, not what the form is showing. */
async function savedWindow(page: Page): Promise<number | undefined> {
  const response = await page.request.get("/api/pipelines/current", {
    headers: { Authorization: "Bearer e2e-token" },
  });
  const nodes = (await response.json()).nodes as {
    id: string;
    step?: { window_length?: number };
  }[];
  return nodes.find((node) => node.id === "savgol")?.step?.window_length;
}

test("a preprocessing node gets a typed form with the schema's own bounds", async ({ page }) => {
  const inspector = await selectNode(page, /SG d1 w11/);

  await expect(inspector.getByLabel("Window Length")).toHaveValue("11");
  await expect(inspector.getByLabel("Polyorder")).toHaveValue("2");
  await expect(inspector.getByLabel("Deriv")).toHaveValue("1");

  // The bound comes from models.py: deriv is ge=0, le=2.
  await inspector.getByLabel("Deriv").fill("5");
  await expect(inspector.getByRole("alert")).toHaveText("Deriv must be at most 2");
  await expect(inspector.getByRole("button", { name: "Apply and re-run" })).toBeDisabled();
});

test("a rule the schema cannot express is answered by the model itself", async ({ page }) => {
  const inspector = await selectNode(page, /SG d1 w11/);

  // An even window is legal JSON Schema and illegal chemometrics. The form
  // does not know that; models.py does, and its sentence is what appears.
  await inspector.getByLabel("Window Length").fill("10");
  await expect(inspector.getByRole("button", { name: "Apply and re-run" })).toBeEnabled();
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();
  await expect(inspector.getByRole("alert")).toHaveText("window_length must be odd");
});

test("an accepted edit recomputes on the one press, and nothing is left dimmed", async ({
  page,
}) => {
  const inspector = await selectNode(page, /^MSC/);
  await page.getByRole("tab", { name: /MSC/ }).click();
  await inspector.getByLabel("Reference").selectOption("median");
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();

  // The outcome rather than a frame of the middle: the edited node is the only
  // one whose arrays have to be recomputed and every other node is already in
  // the store, so this run is over in milliseconds - well inside one poll of
  // the job. Asserting "Queued" or even "Done" here asserts that the machine
  // was slow enough to be caught looking, which is why this test was flaky.
  // `runs.spec.ts` watches a run advance, on a project seeded large enough.
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.getByTestId("node-complete")).toHaveCount(16);
  await expect(page.getByTestId("node-stale")).toHaveCount(0);

  // Nothing is left asking to be pressed: the edit ran, so there is no banner
  // and no second button to find.
  await expect(page.getByText("Downstream results are stale.")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Re-run" })).toHaveCount(0);

  // A run does not cost anyone a node. Every one is still on the canvas.
  await expect(page.locator(".react-flow__node")).toHaveCount(16);
});

test("provenance is collapsed until asked for, and hashes are truncated in the middle", async ({
  page,
}) => {
  const inspector = await selectNode(page, /tecator_raw/);
  const toggle = inspector.getByRole("button", { name: /Provenance record/ });
  await expect(toggle).toHaveAttribute("aria-expanded", "false");

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  // The hash is the file's, so it is asserted by shape rather than by value:
  // it belonged to the fixture before #89 and belongs to the imported file now.
  await expect(inspector.getByText(/^sha256:[0-9a-f]{4}…[0-9a-f]{4}$/)).toBeVisible();
});

test("an accepted edit is written to the pipeline, not only to the screen", async ({ page }) => {
  // #157. Apply validated the step, marked the node stale and sent nothing, so
  // the run that followed read the old number back off disk and the edit
  // looked ignored - the same curve, however many times it was re-run.
  //
  // Asserted against the pipeline the server holds rather than against a
  // reopened form. A node's label is built from its parameters, so the moment
  // this edit lands the node stops being called "SG d1 w11" and becomes "SG d1
  // w9" - and `snv_savgol`, which carries the same window, inherits the old
  // label alone. Looking the node up a second time by the name it used to have
  // finds the wrong node and reads 11 off it, which is a passing bug rather
  // than a failing one. "Written to the pipeline" is the claim; ask the
  // pipeline.
  const outline = page.goto("/?token=e2e-token").then(() =>
    page.getByRole("complementary", { name: "Project outline" }),
  );
  await (await outline).getByRole("button", { name: /SG d1 w11/ }).first().dblclick();
  const inspector = page.getByRole("complementary", { name: "Inspector" });

  await expect(inspector.getByLabel("Window Length")).toHaveValue("11");
  await inspector.getByLabel("Window Length").fill("9");
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();

  // Polled, not read once: a single read races the PUT it is checking for.
  await expect.poll(() => savedWindow(page), APPLIED).toBe(9);

  // Put it back: the project outlives this test, and the file above opens by
  // asserting the seeded 11.
  await inspector.getByLabel("Window Length").fill("11");
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();
  await expect.poll(() => savedWindow(page), APPLIED).toBe(11);
});

/** #175. The inspector's metrics came from the experiment, whose numbers are
 * the last estimator's in the graph, so every estimator showed the same ones
 * and a PLS node was labelled "PC1". Asserted as equality with each node's own
 * served result, not as "the two differ". */
test("an estimator's metrics are its own result's", async ({ page }) => {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  const inspector = page.getByRole("complementary", { name: "Inspector" });

  for (const id of ["pca_a", "pca_d"]) {
    const served = await (
      await page.request.get(`/api/results/${id}`, {
        headers: { Authorization: "Bearer e2e-token" },
      })
    ).json();
    await page.locator(`.react-flow__node[data-id="${id}"]`).click();
    await expect(inspector.locator(".kv", { hasText: "PC1 variance" })).toContainText(
      served.explained_variance_ratio[0].toFixed(4),
    );
    await page.getByRole("tab", { name: "Pipeline" }).click();
  }

  const pls = await (
    await page.request.get("/api/results/pls_d", {
      headers: { Authorization: "Bearer e2e-token" },
    })
  ).json();
  await page.locator('.react-flow__node[data-id="pls_d"]').click();
  await expect(inspector.locator(".kv", { hasText: "RMSEC" }).first()).toContainText(
    pls.metrics.rmsec.toFixed(4),
  );
  await expect(inspector.getByText("PC1 variance")).toHaveCount(0);
});

/** The fold count the server holds for `split_d`, which is the claim an edit
 * to a split makes - the same rule `savedWindow` states for a step. */
async function savedSplits(page: Page): Promise<number | undefined> {
  const response = await page.request.get("/api/pipelines/current", {
    headers: { Authorization: "Bearer e2e-token" },
  });
  const nodes = (await response.json()).nodes as { id: string; spec?: { n_splits?: number } }[];
  return nodes.find((node) => node.id === "split_d")?.spec?.n_splits;
}

test("a split is edited like a step, and the edit reaches the pipeline", async ({ page }) => {
  // #182. A split or an estimator carries `spec` rather than `step`, and the
  // inspector built a form only for the latter - so a k-fold's fold count
  // and a PLS node's target could be set only by the menu that added them.
  const inspector = await selectNode(page, /K-fold 10/);
  await expect(inspector.getByLabel("N Splits")).toHaveValue("10");
  await expect(inspector.getByLabel("Shuffle")).toHaveValue("true");

  await inspector.getByLabel("N Splits").fill("5");
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();
  await expect.poll(() => savedSplits(page), APPLIED).toBe(5);

  // Put it back: the seeded project outlives this test.
  await inspector.getByLabel("N Splits").fill("10");
  await inspector.getByRole("button", { name: "Apply and re-run" }).click();
  await expect.poll(() => savedSplits(page), APPLIED).toBe(10);
});

test("a PLS target is chosen from the dataset's own columns", async ({ page }) => {
  const inspector = await selectNode(page, /PLS 5 LV/);
  const target = inspector.getByLabel("Target");
  await expect(target).toHaveValue("fat");
  // A select over the columns, not a free text field: Tecator carries
  // exactly these, and a name that is not one is refused at run time anyway.
  await expect(target.locator("option")).toHaveText(["moisture", "fat", "protein"]);
  await expect(inspector.getByLabel("N Components")).toHaveValue("5");
});

test("a PLS-DA class column is chosen from the dataset's two-valued columns", async ({ page }) => {
  // #185: `fat_class` is the one metadata column with exactly two values.
  const inspector = await selectNode(page, /PLS-DA 5 LV/);
  const column = inspector.getByLabel("Class Column");
  await expect(column).toHaveValue("fat_class");
  await expect(column.locator("option")).toHaveText(["fat_class"]);
});
