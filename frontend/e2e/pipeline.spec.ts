import { expect, test, type Page } from "@playwright/test";

/** The pipeline canvas against the stub server: the branching graph, the five
 * states side by side, and a pipeline assembled through the step list. */

async function openCanvas(page: Page) {
  await page.goto("/?token=e2e-token");
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.getByTestId("pipeline-canvas")).toBeVisible();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
}

test("the branching pipeline renders with every node the fixture describes", async ({ page }) => {
  await openCanvas(page);
  await expect(page.locator(".react-flow__node")).toHaveCount(14);
  await expect(page.locator(".react-flow__edge")).toHaveCount(13);

  // The node bodies carry their parameters, which is what makes the graph
  // readable without opening anything.
  await expect(page.getByText("window 11 · poly 2 · deriv 1").first()).toBeVisible();
  await expect(page.getByText("10 folds · shuffle · seed 42")).toBeVisible();
});

test("all five states are on screen at once and distinguishable by form", async ({ page }) => {
  await openCanvas(page);
  for (const state of ["complete", "running", "stale", "failed", "not_run"]) {
    await expect(page.getByTestId(`node-${state}`).first()).toBeVisible();
  }

  // Form, not only colour: dashed for stale and not-run, a left stripe for
  // failed, an accent border and progress for running.
  const border = (state: string) =>
    page
      .getByTestId(`node-${state}`)
      .first()
      .evaluate((element) => {
        const style = getComputedStyle(element);
        return {
          style: style.borderTopStyle,
          left: style.borderLeftWidth,
          background: style.backgroundImage,
        };
      });

  expect((await border("stale")).style).toBe("dashed");
  expect((await border("stale")).background).toContain("repeating-linear-gradient");
  expect((await border("not_run")).style).toBe("dashed");
  expect((await border("failed")).left).toBe("3px");
  expect((await border("complete")).style).toBe("solid");

  // The stale node says why, and the failed one says what went wrong.
  await expect(page.getByText("edited - downstream stale")).toBeVisible();
  await expect(page.getByText(/matrix of rank 4/)).toBeVisible();
  await expect(page.getByTestId("node-running").locator(".prog i")).toBeVisible();
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
