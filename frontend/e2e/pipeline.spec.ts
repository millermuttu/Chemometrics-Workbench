import { expect, test, type Page } from "@playwright/test";

/** The pipeline canvas against the real server: the branching graph the
 * executor really ran, and a pipeline assembled through the step list.
 *
 * The stub could show all five node states at once because its fixture said
 * so. A real state is a fact about the project: every node here has its arrays
 * on disk, so every node is `complete`. `running`, `failed` and `not_run` are
 * asserted in `runs.spec.ts`, where a run really runs, and `stale` in
 * `inspector.spec.ts`, where an edit really invalidates one. */

async function openCanvas(page: Page) {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.getByTestId("pipeline-canvas")).toBeVisible();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
}

test("the branching pipeline renders with every node the executor ran", async ({ page }) => {
  await openCanvas(page);
  await expect(page.locator(".react-flow__node")).toHaveCount(14);
  await expect(page.locator(".react-flow__edge")).toHaveCount(13);

  // The node bodies carry their parameters, which is what makes the graph
  // readable without opening anything.
  await expect(page.getByText("window 11 · poly 2 · deriv 1").first()).toBeVisible();
  await expect(page.getByText("10 folds · shuffle · seed 42")).toBeVisible();
});

test("a node that has been run is complete, and says so by form", async ({ page }) => {
  await openCanvas(page);
  await expect(page.getByTestId("node-complete").first()).toBeVisible();
  await expect(page.getByTestId("node-complete")).toHaveCount(14);

  // Form, not only colour: complete is a solid border, which is what the other
  // four states are distinguished *from*.
  const drawn = await page
    .getByTestId("node-complete")
    .first()
    .evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.borderTopStyle, opacity: Number(style.opacity) };
    });
  expect(drawn.style).toBe("solid");
  expect(drawn.opacity).toBe(1);
});

test("selecting a node focuses its tab", async ({ page }) => {
  await openCanvas(page);
  await page.locator(".react-flow__node").filter({ hasText: "SNV" }).first().click();
  await expect(page.getByRole("tab", { name: /SNV/ })).toBeVisible();
});

test("a linear SNV to Savitzky-Golay to PCA pipeline is assembled through the step list", async ({
  page,
}) => {
  await openCanvas(page);
  const before = await page.locator(".react-flow__node").count();

  for (const step of ["SNV", "SG d1 w11", "PCA"]) {
    await page.getByLabel("Step").selectOption(step);
    await page.getByRole("button", { name: "Add", exact: true }).click();
  }
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 3);

  await page.getByRole("button", { name: "Validate" }).click();
  await expect(page.getByText("valid · 3 steps")).toBeVisible();

  await page.getByRole("button", { name: "Remove PCA" }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(before + 2);
});

/** Drag a connection between two node ports.
 *
 * React Flow listens to pointer movement rather than to a drop event, so this
 * is a real mouse path: down on the source port, several moves, up on the
 * target. `dragTo` sends too few moves and the connection never starts.
 */
async function connect(page: Page, from: string, to: string) {
  const port = (id: string, side: "right" | "left") =>
    page.locator(`.react-flow__node[data-id="${id}"] .react-flow__handle-${side}`);
  const source = (await port(from, "right").boundingBox())!;
  const target = (await port(to, "left").boundingBox())!;
  await page.mouse.move(source.x + source.width / 2, source.y + source.height / 2);
  await page.mouse.down();
  await page.mouse.move(target.x + target.width / 2, target.y + target.height / 2, { steps: 12 });
  await page.mouse.up();
}

/** #51's direct manipulation. These edit the canvas and never save, so the
 * seeded project on 8765 is left as the other tests expect to find it. */

test("dragging from an output port moves a branch onto a new parent", async ({ page }) => {
  await openCanvas(page);
  // An edge is an SVG <g> with no box of its own, so it is counted rather
  // than asserted visible - Playwright calls every one of them hidden.
  await expect(page.locator('.react-flow__edge[data-id="snv->centre_a"]')).toHaveCount(1);

  await connect(page, "msc", "centre_a");

  // Exactly one input, so the old edge is replaced rather than added to: the
  // graph keeps its thirteen edges and centre_a now reads from msc.
  await expect(page.locator('.react-flow__edge[data-id="msc->centre_a"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__edge[data-id="snv->centre_a"]')).toHaveCount(0);
  await expect(page.locator(".react-flow__edge")).toHaveCount(13);
});

test("a connection that would make a cycle does not drop, and says why", async ({ page }) => {
  await openCanvas(page);

  // pca_a is downstream of snv, so this would close a loop.
  await connect(page, "pca_a", "snv");

  await expect(page.getByText(/would make a cycle/)).toBeVisible();
  await expect(page.locator('.react-flow__edge[data-id="pca_a->snv"]')).toHaveCount(0);
  await expect(page.locator('.react-flow__edge[data-id="source->snv"]')).toHaveCount(1);
});

test("removing a node reconnects its children to its parent", async ({ page }) => {
  await openCanvas(page);
  // The control is on the node because clicking a node opens its tab: anything
  // in the side panel would be replaced by the tab before it could be used.
  await page.locator('.react-flow__node[data-id="snv"] button[aria-label^="Remove node"]').click();

  // snv fed centre_a and snv_savgol; both now read from snv's own parent.
  await expect(page.locator('.react-flow__node[data-id="snv"]')).toHaveCount(0);
  await expect(page.locator('.react-flow__edge[data-id="source->centre_a"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__edge[data-id="source->snv_savgol"]')).toHaveCount(1);
  await expect(page.locator(".react-flow__node")).toHaveCount(13);
});

test("the source has no remove control, because it is where the data enters", async ({ page }) => {
  await openCanvas(page);
  await expect(
    page.locator('.react-flow__node[data-id="source"] button[aria-label^="Remove node"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('.react-flow__node[data-id="snv"] button[aria-label^="Remove node"]'),
  ).toHaveCount(1);
});
