import { expect, test, type Page } from "@playwright/test";

/** The shell against the real server: the artboard's measurements and the tab
 * rules, over a project whose pipeline has been run. */

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
  const menu = page.locator(".omenu button");
  const hidden = await menu.count();
  expect(hidden).toBeGreaterThan(0);

  // Which tabs are clipped depends on how wide their titles render, so the
  // search term is taken from the menu rather than assumed: what is being
  // tested is that typing narrows it, not which tab happened to overflow.
  const first = (await menu.first().innerText()).trim();
  await page.getByRole("textbox", { name: "Search tabs" }).fill(first);
  await expect(menu).toHaveCount(await page.locator(".omenu button").count());
  expect(await menu.count()).toBeLessThanOrEqual(hidden);
  await expect(menu.first()).toContainText(first);

  await page.getByRole("textbox", { name: "Search tabs" }).fill("no such tab");
  await expect(menu).toHaveCount(0);
  await expect(page.getByText("Nothing matches.")).toBeVisible();

  await page.getByRole("textbox", { name: "Search tabs" }).fill(first);
  await menu.first().click();

  await page.getByRole("button", { name: "Split view" }).click();
  await expect(page.locator(".split .pane")).toHaveCount(2);

  const before = await page.getByRole("tab").count();
  await page.getByRole("tab").first().getByRole("button").click();
  await expect(page.getByRole("tab")).toHaveCount(before - 1);
});

/* A run that advances and cancels needs work to do, and every array in this
 * project is already on disk - a run here hits the cache and is over before a
 * second frame renders. That test lives in `runs.spec.ts`, against the project
 * seeded with nothing computed. */

test("both themes are the artboard's palettes", async ({ page }) => {
  await open(page);
  const accent = () =>
    page.evaluate(() => getComputedStyle(document.querySelector(".app")!).getPropertyValue("--accent").trim());

  expect((await accent()).toUpperCase()).toBe("#0B6B62");
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  expect((await accent()).toUpperCase()).toBe("#54BFAB");
});

test("without a token the shell says so, rather than loading forever", async ({ page }) => {
  // The launch URL carries the token once. Opening the bare address - which a
  // browser's autocomplete does readily - has to say what happened.
  await page.goto("/");

  const notice = page.getByTestId("cannot-load");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("401 · UNAUTHORIZED");
  await expect(notice).toContainText("Not authenticated");
  await expect(notice).toContainText("?token=");
  await expect(notice).not.toContainText("Traceback");

  // The application chrome stays: this is a window, not a failed browser tab.
  await expect(page.locator(".tbar")).toBeVisible();
  await expect(page.getByText("Loading…")).toHaveCount(0);
});

test("a server that is not answering is told apart from one that refuses", async ({ page }) => {
  await page.route("**/api/**", (route) => route.abort());
  await page.goto("/?token=e2e-token");

  const notice = page.getByTestId("cannot-load");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("NO RESPONSE");
  await expect(notice).toContainText("not answering");
});
