import { expect, test, type Page } from "@playwright/test";

/** The inspector: a typed form built from the schema, a message that comes
 * from the model, and an edit that marks downstream results stale. */

async function selectNode(page: Page, name: RegExp) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name }).first().dblclick();
  return page.getByRole("complementary", { name: "Inspector" });
}

test("a preprocessing node gets a typed form with the schema's own bounds", async ({ page }) => {
  const inspector = await selectNode(page, /SG d1 w11/);

  await expect(inspector.getByLabel("Window Length")).toHaveValue("11");
  await expect(inspector.getByLabel("Polyorder")).toHaveValue("2");
  await expect(inspector.getByLabel("Deriv")).toHaveValue("1");

  // The bound comes from models.py: deriv is ge=0, le=2.
  await inspector.getByLabel("Deriv").fill("5");
  await expect(inspector.getByRole("alert")).toHaveText("Deriv must be at most 2");
  await expect(inspector.getByRole("button", { name: "Apply" })).toBeDisabled();
});

test("a rule the schema cannot express is answered by the model itself", async ({ page }) => {
  const inspector = await selectNode(page, /SG d1 w11/);

  // An even window is legal JSON Schema and illegal chemometrics. The form
  // does not know that; models.py does, and its sentence is what appears.
  await inspector.getByLabel("Window Length").fill("10");
  await expect(inspector.getByRole("button", { name: "Apply" })).toBeEnabled();
  await inspector.getByRole("button", { name: "Apply" }).click();
  await expect(inspector.getByRole("alert")).toHaveText("window_length must be odd");
});

test("an accepted edit marks downstream nodes stale and offers a re-run", async ({ page }) => {
  const inspector = await selectNode(page, /^MSC/);
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  const staleBefore = await page.getByTestId("node-stale").count();

  await page.getByRole("tab", { name: /MSC/ }).click();
  await inspector.getByLabel("Reference").selectOption("median");
  await inspector.getByRole("button", { name: "Apply" }).click();

  await expect(page.getByText("Downstream results are stale.")).toBeVisible();

  await page.getByRole("tab", { name: "Pipeline" }).click();
  expect(await page.getByTestId("node-stale").count()).toBeGreaterThan(staleBefore);

  // A stale result dims; it does not vanish. Every node is still on the canvas.
  await expect(page.locator(".react-flow__node")).toHaveCount(15);

  // The re-run is offered, and taking it clears what the edit made stale.
  //
  // The outcome rather than a frame of the middle: the edited node is the only
  // one whose arrays have to be recomputed and every other node is already in
  // the store, so this run is over in milliseconds - well inside one poll of
  // the job. Asserting "Queued" or even "Done" here asserts that the machine
  // was slow enough to be caught looking, which is why this test was flaky.
  // `runs.spec.ts` watches a run advance, on a project seeded large enough.
  await page.getByRole("button", { name: "Re-run" }).click();
  await expect(page.getByText("Downstream results are stale.")).toHaveCount(0);
  await expect(page.getByTestId("node-stale")).toHaveCount(0);
  await expect(page.getByTestId("node-complete")).toHaveCount(15);
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
  await inspector.getByRole("button", { name: "Apply" }).click();
  await expect(page.getByText("Downstream results are stale.")).toBeVisible();

  const saved = await page.request.get("/api/pipelines/current", {
    headers: { Authorization: "Bearer e2e-token" },
  });
  const nodes = (await saved.json()).nodes as { id: string; step?: { window_length?: number } }[];
  expect(nodes.find((node) => node.id === "savgol")?.step?.window_length).toBe(9);

  // Put it back: the project outlives this test, and the file above opens by
  // asserting the seeded 11.
  await inspector.getByLabel("Window Length").fill("11");
  await inspector.getByRole("button", { name: "Apply" }).click();
  await expect
    .poll(async () => {
      const back = await page.request.get("/api/pipelines/current", {
        headers: { Authorization: "Bearer e2e-token" },
      });
      const list = (await back.json()).nodes as { id: string; step?: { window_length?: number } }[];
      return list.find((node) => node.id === "savgol")?.step?.window_length;
    })
    .toBe(11);
});
