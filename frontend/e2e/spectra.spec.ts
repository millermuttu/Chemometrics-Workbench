import { expect, test, type Page } from "@playwright/test";

/** The spectra view against the stub server: WebGL traces over a density
 * band, the counts the artboard shows, and a plot that follows the theme. */

async function openSpectra(page: Page, node = /^SNV/) {
  await page.goto("/?token=e2e-token");
  const outline = page.getByRole("complementary", { name: "Project outline" });
  await outline.getByRole("button", { name: node }).first().dblclick();
  await expect(page.getByTestId("spectra-plot")).toBeVisible();
  await expect(page.locator(".gl-container canvas").first()).toBeVisible();
}

test("240 spectra arrive as a band plus the drawn subset, and the counts say so", async ({
  page,
}) => {
  await openSpectra(page);
  await expect(page.getByText("240 spectra · 0 highlighted · 100 variables")).toBeVisible();
  await expect(page.getByText("60 of 240 drawn")).toBeVisible();
  await expect(page.getByText("shaded band = full set (240)")).toBeVisible();

  // Tecator is 100 variables wide, so nothing is dropped along the x axis -
  // the screen says which of the two decimations is in play.
  await expect(page.getByText("no x decimation")).toBeVisible();
});

test("highlighting a sample draws it over the band and names it", async ({ page }) => {
  await openSpectra(page);
  await page.getByLabel("Highlight sample").selectOption({ index: 3 });

  await expect(page.getByText("240 spectra · 1 highlighted · 100 variables")).toBeVisible();
  const named = page.locator("text=Highlighted:");
  await expect(named).toBeVisible();

  // The selected sample is drawn at full strength; the others stay faint.
  const opacities = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=spectra-plot]") as HTMLElement & {
      data?: { opacity?: number; customdata?: number }[];
    };
    return (plot.data ?? []).filter((trace) => typeof trace.customdata === "number").map((t) => t.opacity);
  });
  expect(opacities).toContain(1);
  expect(opacities.filter((value) => value !== 1).length).toBeGreaterThan(50);

  await page.getByRole("button", { name: "Clear highlight" }).click();
  await expect(page.getByText("240 spectra · 0 highlighted · 100 variables")).toBeVisible();
});

test("the band is the token's envelope over the whole set", async ({ page }) => {
  await openSpectra(page);
  const band = await page.evaluate(() => {
    const plot = document.querySelector("[data-testid=spectra-plot]") as HTMLElement & {
      data?: { fill?: string; fillcolor?: string }[];
    };
    const filled = (plot.data ?? []).find((trace) => trace.fill === "tonexty");
    const token = getComputedStyle(document.querySelector(".app")!)
      .getPropertyValue("--band")
      .trim();
    return { fillcolor: filled?.fillcolor, token };
  });
  expect(band.fillcolor).toBe(band.token);
});

test("the axes carry their real units", async ({ page }) => {
  await openSpectra(page);
  const plot = page.getByTestId("spectra-plot");
  await expect(plot).toContainText("wavelength_nm (nm)");
  await expect(plot).toContainText("Absorbance");
});

test("the layers switch, and a source node has only raw", async ({ page }) => {
  await openSpectra(page);
  const group = page.getByRole("group", { name: "Layers" });
  await expect(group.getByRole("button", { name: "Both" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await group.getByRole("button", { name: "Processed" }).click();
  await expect(group.getByRole("button", { name: "Processed" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );

  await openSpectra(page, /^Source/);
  await expect(page.getByRole("group", { name: "Layers" }).getByRole("button", { name: "Processed" })).toBeDisabled();
});

test("the plot is drawn from the tokens, in whichever theme is applied", async ({ page }) => {
  await openSpectra(page);
  // Plotly paints the paper on the first main-svg. Reading it back is the
  // only honest check that the plot took the token rather than its default
  // white, which would look right in the light theme and wrong in the dark.
  const paper = () =>
    page.evaluate(
      () => (document.querySelector(".main-svg") as SVGElement | null)?.style.background ?? "",
    );

  expect(await paper()).toContain("rgb(255, 255, 255)");
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  await expect
    .poll(async () => await paper())
    .toContain("rgb(12, 20, 19)");
});
