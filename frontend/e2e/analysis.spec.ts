import { expect, test, type Page } from "@playwright/test";

/** The analysis tab: the artboard's panel grid, the T² ellipse drawn from the
 * served limit, and an outlier that can be hovered to name its sample. */

async function openResults(page: Page) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: /PCA 5 PC/ }).first().dblclick();
  await expect(page.getByTestId("scores-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
}

test("one tab holds the panel grid that answers question 11.1", async ({ page }) => {
  await openResults(page);
  for (const panel of ["Scores", "Loadings", "Explained variance", "Diagnostics"]) {
    await expect(page.getByText(panel, { exact: true })).toBeVisible();
  }
  // The headline numbers the artboard puts in the header, in the accent ink.
  const header = page.getByTestId("analysis-header");
  await expect(header).toContainText("PCA 5 components · 240 × 100");
  await expect(header).toContainText("PC1");
  await expect(header).toContainText("68.9%");
  await expect(header).toContainText("99.9%");
});

test("the axes carry their components and the loadings axis its real unit", async ({ page }) => {
  await openResults(page);
  await expect(page.getByTestId("scores-plot")).toContainText("PC 1 (68.9%)");
  await expect(page.getByTestId("scores-plot")).toContainText("PC 2 (28.4%)");
  await expect(page.getByTestId("loadings-plot")).toContainText("wavelength_nm (nm)");

  await page.getByLabel("Scores y axis").selectOption("2");
  await expect(page.getByTestId("scores-plot")).toContainText("PC 3 (1.6%)");
});

test("the ellipse comes from the fixture's limit, not from the browser", async ({ page }) => {
  await openResults(page);
  const measured = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=scores-plot]") as HTMLElement & {
      data?: { name?: string; x?: number[]; y?: number[] }[];
    };
    const ellipse = (plot.data ?? []).find((trace) => trace.name === "T² limit");
    return { x: Math.max(...(ellipse?.x ?? [])), y: Math.max(...(ellipse?.y ?? [])) };
  });

  const served = await page.evaluate(async () => {
    const response = await fetch("/api/results/pca_a", {
      headers: { Authorization: `Bearer ${sessionStorage.getItem("token")}` },
    });
    const pca = await response.json();
    return {
      x: Math.sqrt(pca.diagnostics.hotelling_t2_limit * pca.eigenvalues[0]),
      y: Math.sqrt(pca.diagnostics.hotelling_t2_limit * pca.eigenvalues[1]),
    };
  });

  expect(measured.x).toBeCloseTo(served.x, 9);
  expect(measured.y).toBeCloseTo(served.y, 9);
});

test("the diagnostics block lists what the limits put outside, in tabular numerals", async ({
  page,
}) => {
  await openResults(page);
  await expect(page.getByText("Hotelling T² limit")).toBeVisible();
  await expect(page.getByText("SPE limit")).toBeVisible();

  const rows = page.locator("table tbody tr");
  await expect(rows.first()).toBeVisible();
  const alignment = await page
    .locator("table td.n")
    .first()
    .evaluate((cell) => getComputedStyle(cell).fontVariantNumeric);
  expect(alignment).toContain("tabular-nums");
});

test("hovering a score names its sample", async ({ page }) => {
  await openResults(page);
  const plot = page.getByTestId("scores-plot");
  const box = (await plot.boundingBox())!;
  // Sweep the middle of the cloud until Plotly puts a hover label up.
  for (let fraction = 0.3; fraction < 0.7; fraction += 0.02) {
    await page.mouse.move(box.x + box.width * fraction, box.y + box.height * 0.5);
    if (await page.locator(".hovertext").count()) break;
  }
  await expect(page.locator(".hovertext")).toBeVisible();
  await expect(page.locator(".hovertext")).toContainText(/C\d{3}|E\d{3}/);
});
