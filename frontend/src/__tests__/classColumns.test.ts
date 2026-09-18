/** #185: a PLS-DA classifies by a column with exactly two values. */
import { expect, it } from "vitest";

import { twoValuedColumns } from "@/shell/classColumns";

it("keeps the two-valued columns and nothing else", () => {
  const columns = {
    fat_class: ["high", "low", "low", "high"],
    batch: ["a", "b", "c", "a"],
    site: ["x", "x", "x", "x"],
    lot: [1, 2, 1, 2],
  };
  expect(twoValuedColumns(columns)).toEqual(["fat_class", "lot"]);
  expect(twoValuedColumns(undefined)).toEqual([]);
});
