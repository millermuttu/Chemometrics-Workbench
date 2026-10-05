import path from "node:path";

import { expect, type Page } from "@playwright/test";

/** What the worked examples' specs share (#237, #288): screenshots, the
 * panels by role, and the canvas gestures a page tells a reader to make. */

export const ROOT = path.join(import.meta.dirname, "..", "..");
export const OUT = path.join(ROOT, "docs", "images", "screens");

export async function shoot(page: Page, name: string) {
  // Plotly draws after its first frame; let it settle before the capture.
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
}

export const outline = (page: Page) => page.getByRole("complementary", { name: "Project outline" });
export const inspector = (page: Page) => page.getByRole("complementary", { name: "Inspector" });
export const node = (page: Page, label: string) =>
  page.locator(".react-flow__node").filter({ hasText: label });

/** Back to the canvas from whichever tab is in front. */
export async function canvas(page: Page) {
  await page.getByRole("button", { name: "Pipeline", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
}

/** Drag from a node's output port onto empty canvas, and pick from the menu. */
export async function branch(
  page: Page,
  from: ReturnType<typeof node>,
  step: string,
  { shot, dx = -60, dy = 110 }: { shot?: string; dx?: number; dy?: number } = {},
) {
  // The canvas refits as it mounts and again once its nodes are measured; a
  // port read before the second fit is no longer under the pointer (#260).
  const handle = from.locator(".react-flow__handle-right");
  let port = (await handle.boundingBox())!;
  await expect
    .poll(async () => {
      await page.waitForTimeout(150);
      const now = (await handle.boundingBox())!;
      const still = now.x === port.x && now.y === port.y;
      port = now;
      return still;
    })
    .toBe(true);
  await page.mouse.move(port.x + port.width / 2, port.y + port.height / 2);
  await page.mouse.down();
  // A node lands where it is dropped. Down and slightly left by default: a
  // chain that walked right would put the next port under the step list.
  await page.mouse.move(port.x + dx, port.y + dy, { steps: 12 });
  await page.mouse.up();
  const menu = page.getByTestId("add-step-menu");
  await expect(menu).toBeVisible();
  if (shot) await shoot(page, shot);
  await menu.getByRole("menuitem", { name: step, exact: true }).click();
}

export async function apply(page: Page, fields: Record<string, string>) {
  for (const [label, value] of Object.entries(fields)) {
    const field = inspector(page).getByLabel(label);
    if ((await field.evaluate((element) => element.tagName)) === "SELECT") {
      await field.selectOption(value);
    } else {
      await field.fill(value);
    }
  }
  await inspector(page).getByRole("button", { name: "Apply and re-run" }).click();
}
