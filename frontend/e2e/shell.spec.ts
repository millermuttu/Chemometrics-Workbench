import { expect, test, type Page } from "@playwright/test";

/** The shell against the stub server: the artboard's measurements, the tab
 * rules, and a run that really advances and can really be cancelled. */

async function open(page: Page) {
  await page.goto("/?token=e2e-token");
  await expect(page.getByText("Tecator meat study")).toBeVisible();
}

/** The numbers in design/canvas/Main.dc.html. They are the design of record,
 * so a shell that drifts from them is wrong even if it looks fine. */
test("the regions carry the artboard's measurements", async ({ page }) => {
  await open(page);
  const height = (locator: string) => page.locator(locator).first().boundingBox();

  expect((await height(".tbar"))?.height).toBe(40);
  expect((await height(".side"))?.width).toBe(248);
  expect((await height(".tabs"))?.height).toBe(34);
  expect((await height(".insp"))?.width).toBe(292);
  expect((await height(".status"))?.height).toBe(28);
});

test("the outline previews into one transient tab and pins on a double click", async ({ page }) => {
  await open(page);
  const outline = page.getByRole("complementary", { name: "Project outline" });

  await outline.getByRole("button", { name: /^SNV/ }).first().click();
  const tabs = page.getByRole("tab");
  await expect(tabs).toHaveCount(1);
  await expect(tabs.first()).toHaveClass(/tr/);

  // A second preview replaces the first rather than adding to it.
  await outline.getByRole("button", { name: /MSC/ }).first().click();
  await expect(tabs).toHaveCount(1);

  await outline.getByRole("button", { name: /MSC/ }).first().dblclick();
  await expect(tabs.first()).not.toHaveClass(/tr/);
  await outline.getByRole("button", { name: /^SNV/ }).first().click();
  await expect(tabs).toHaveCount(2);
});

test("tabs close, split and overflow into a searchable menu", async ({ page }) => {
  await open(page);
  const outline = page.getByRole("complementary", { name: "Project outline" });
  const rows = outline.getByRole("button");
  const count = await rows.count();

  // Fifteen preprocessing variants is the normal case; open enough to overflow.
  for (let index = 0; index < count; index += 1) {
    await rows.nth(index).dblclick();
  }
  const overflow = page.getByRole("button", { name: /more tabs/ });
  await expect(overflow).toBeVisible();

  await overflow.click();
  await page.getByRole("textbox", { name: "Search tabs" }).fill("pca_d");
  const match = page.locator(".omenu button");
  await expect(match).toHaveCount(1);
  await match.click();

  await page.getByRole("button", { name: "Split view" }).click();
  await expect(page.locator(".split .pane")).toHaveCount(2);

  const before = await page.getByRole("tab").count();
  await page.getByRole("tab").first().getByRole("button").click();
  await expect(page.getByRole("tab")).toHaveCount(before - 1);
});

test("a run advances, then cancels where it stood", async ({ page }) => {
  await open(page);
  await page.getByRole("button", { name: "Run pipeline" }).click();

  const status = page.getByRole("status");
  await expect(status).toContainText("Queued");
  await expect(status).toContainText("Preprocessing: SNV", { timeout: 5000 });

  await status.getByRole("button", { name: "Cancel" }).click();
  await expect(status).toContainText("Cancelled");

  // Cancelled means stopped: the message does not move on afterwards.
  const width = await page.locator(".prog i").getAttribute("style");
  await page.waitForTimeout(1200);
  await expect(status).toContainText("Cancelled");
  expect(await page.locator(".prog i").getAttribute("style")).toBe(width);
});

test("both themes are the artboard's palettes", async ({ page }) => {
  await open(page);
  const accent = () =>
    page.evaluate(() => getComputedStyle(document.querySelector(".app")!).getPropertyValue("--accent").trim());

  expect((await accent()).toUpperCase()).toBe("#0B6B62");
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  expect((await accent()).toUpperCase()).toBe("#54BFAB");
});
