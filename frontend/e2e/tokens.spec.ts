import { expect, test } from "@playwright/test";

/** The smoke test the #50 walkthrough grows out of: the built bundle renders,
 * both palettes are on the page, and the type is the bundled IBM Plex rather
 * than a fallback - which is what a Google Fonts @import would leave behind. */
test("the built bundle renders both palettes in IBM Plex", async ({ page }) => {
  // Nothing may leave the machine. A Google Fonts @import would show up here,
  // and this is cheaper than actually pulling the network out from under the
  // test - it fails on the attempt rather than on the fallback that follows.
  const external: string[] = [];
  page.on("request", (request) => {
    if (!request.url().startsWith("http://127.0.0.1")) external.push(request.url());
  });

  await page.goto("/tokens?token=e2e-token");
  await expect(page.getByRole("heading", { name: "Design tokens" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Light" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Dark" })).toBeVisible();

  const font = await page.evaluate(() => getComputedStyle(document.body).fontFamily);
  expect(font).toContain("IBM Plex Sans");

  const accent = await page.evaluate(() => {
    const probe = document.querySelector(".t-light")!;
    return getComputedStyle(probe).getPropertyValue("--accent").trim();
  });
  expect(accent.toUpperCase()).toBe("#0B6B62");

  expect(external).toEqual([]);
});
