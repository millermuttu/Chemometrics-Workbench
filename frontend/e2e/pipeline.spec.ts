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
