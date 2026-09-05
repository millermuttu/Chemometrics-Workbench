import { expect, test, type Page } from "@playwright/test";

/** The pipeline canvas against the real server: the branching graph the
 * executor really ran, and a pipeline assembled through the step list.
 *
 * The stub could show all five node states at once because its fixture said
 * so. A real state is a fact about the project: every node here has its arrays
 * on disk, so every node is `complete`. `running`, `failed` and `not_run` are
 * asserted in `runs.spec.ts`, where a run really runs. `stale` is asserted
 * nowhere: the server never reports it (`api.py:955-958`) and #159 removed the
 * client-side marking that used to, so a node edited but not re-run comes back
 * `not_run`. The state stays defined and styled against the day it is
 * derivable. */

async function openCanvas(page: Page) {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.getByTestId("pipeline-canvas")).toBeVisible();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
}

/** `getComputedStyle` reports a border colour as `rgb(r, g, b)`; the token is
 * a hex string. One of them has to be converted, and the hex is the shorter
 * trip. */
function hexOf(rgb: string): string {
  const [r, g, b] = rgb.match(/\d+/g)!.map(Number);
  return `#${[r, g, b].map((part) => part.toString(16).padStart(2, "0")).join("")}`;
}

test("the branching pipeline renders with every node the executor ran", async ({ page }) => {
  await openCanvas(page);
  await expect(page.locator(".react-flow__node")).toHaveCount(15);
  await expect(page.locator(".react-flow__edge")).toHaveCount(14);

  // The node bodies carry their parameters, which is what makes the graph
  // readable without opening anything.
  await expect(page.getByText("window 11 · poly 2 · deriv 1").first()).toBeVisible();
  await expect(page.getByText("10 folds · shuffle · seed 42")).toBeVisible();
});

test("a node that has been run is complete, and says so by form", async ({ page }) => {
  await openCanvas(page);
  await expect(page.getByTestId("node-complete").first()).toBeVisible();
  await expect(page.getByTestId("node-complete")).toHaveCount(15);

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

  // Colour as well as form: a node holding a result reads green, and green is
  // its own token rather than the accent - the accent means *running*, and a
  // finished node must not be the same colour as one still working.
  const paint = await page.evaluate(() => {
    const style = getComputedStyle(document.querySelector(".app")!);
    const complete = document.querySelector('[data-testid="node-complete"]')!;
    return {
      border: getComputedStyle(complete).borderTopColor,
      ok: style.getPropertyValue("--ok").trim(),
      accent: style.getPropertyValue("--accent").trim(),
    };
  });
  expect(hexOf(paint.border)).toBe(paint.ok.toLowerCase());
  expect(paint.ok).not.toBe(paint.accent);
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
  // graph keeps its fourteen edges and centre_a now reads from msc.
  await expect(page.locator('.react-flow__edge[data-id="msc->centre_a"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__edge[data-id="snv->centre_a"]')).toHaveCount(0);
  await expect(page.locator(".react-flow__edge")).toHaveCount(14);
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
  await expect(page.locator(".react-flow__node")).toHaveCount(14);
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

test("duplicating a branch copies it and everything below it", async ({ page }) => {
  await openCanvas(page);
  const before = await page.locator(".react-flow__node").count();

  // snv feeds centre_a and snv_savgol, so duplicating it copies a whole branch
  // rather than a node.
  await page
    .locator('.react-flow__node[data-id="snv"] button[aria-label^="Duplicate node"]')
    .click();

  await expect(page.locator('.react-flow__node[data-id="snv copy"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__node[data-id="centre_a copy"]')).toHaveCount(1);
  await expect(page.locator('.react-flow__node[data-id="pca_a copy"]')).toHaveCount(1);
  // The copy is a sibling branch: it reads from snv's own parent, not from snv.
  await expect(page.locator('.react-flow__edge[data-id="source->snv copy"]')).toHaveCount(1);
  expect(await page.locator(".react-flow__node").count()).toBeGreaterThan(before);
});

test("the source offers no duplicate control, because it cannot be copied", async ({ page }) => {
  await openCanvas(page);
  await expect(
    page.locator('.react-flow__node[data-id="source"] button[aria-label^="Duplicate node"]'),
  ).toHaveCount(0);
});

test("picking two terminal nodes opens a comparison tab", async ({ page }) => {
  await openCanvas(page);

  // Offered on terminal estimators only: pca_a is one, snv is not.
  await expect(
    page.locator('.react-flow__node[data-id="snv"] button[aria-label*="for comparison"]'),
  ).toHaveCount(0);

  const armed = page.locator('.react-flow__node[data-id="pca_a"] button[aria-label^="Pick"]');
  await expect(armed).toHaveCount(1);
  await armed.click();
  // The first pick arms rather than opening anything, and says it is armed.
  await expect(
    page.locator('.react-flow__node[data-id="pca_a"] button[aria-label^="Unpick"]'),
  ).toHaveCount(1);

  await page.locator('.react-flow__node[data-id="pls_d"] button[aria-label^="Pick"]').click();

  await expect(page.getByRole("tab", { name: /pca_a vs pls_d/ })).toBeVisible();
  await expect(page.getByTestId("compare-view")).toBeVisible();
  // Only pls_d carries regression metrics, so every row is one-sided and the
  // difference column is an em dash rather than a number.
  await expect(page.getByRole("region", { name: "Metrics" })).toBeVisible();
  await expect(page.getByTestId("delta-RMSECV")).toHaveText("—");
});

/** Dragging a node writes its position through on the drop, without waiting on
 * Save - a position is not part of the recipe.
 *
 * The survival of a reload is asserted against the layout the server holds,
 * not against a screen coordinate: `fitView` re-fits the viewport on every
 * mount, so a node that has not moved an inch in the graph still lands on a
 * different pixel. Reading the box either side of a reload compares two
 * different zoom levels and fails on a change that did not happen. That the
 * canvas draws what the layout says is `graph.test.ts`'s job, and it has one.
 *
 * Unlike the edits above, this one *does* change the seeded project on 8765.
 * Safe because every other spec there asserts node counts, edges and states,
 * never coordinates - said out loud so the next person to assert a position
 * knows why theirs might move.
 */
test("a dragged node is written through on the drop, and not opened by the drag", async ({
  page,
}) => {
  await openCanvas(page);
  const node = page.locator('.react-flow__node[data-id="snv"]');
  const before = (await node.boundingBox())!;

  // A real mouse path, for the same reason `connect` uses one: React Flow
  // follows pointer movement rather than a drop event.
  await page.mouse.move(before.x + before.width / 2, before.y + 10);
  await page.mouse.down();
  await page.mouse.move(before.x + before.width / 2 + 160, before.y + 130, { steps: 12 });
  await page.mouse.up();

  // Within one mount the viewport is fixed, so the box is a fair comparison.
  const after = (await node.boundingBox())!;
  expect(Math.round(after.x - before.x)).toBeGreaterThan(100);

  // The drag consumed the click that ends it: moving a node must not open it.
  await expect(page.getByRole("tab", { name: /SNV/ })).toHaveCount(0);

  // The claim: the server holds it, so it outlives this page.
  await expect
    .poll(async () => {
      const state = await page.request.get("/api/pipelines/current/state", {
        headers: { Authorization: "Bearer e2e-token" },
      });
      return (await state.json()).layout.snv;
    })
    .not.toEqual({ x: 40, y: 170 });

  const moved = await (
    await page.request.get("/api/pipelines/current/state", {
      headers: { Authorization: "Bearer e2e-token" },
    })
  ).json();

  await openCanvas(page);
  const reloaded = await (
    await page.request.get("/api/pipelines/current/state", {
      headers: { Authorization: "Bearer e2e-token" },
    })
  ).json();
  expect(reloaded.layout.snv).toEqual(moved.layout.snv);
});
