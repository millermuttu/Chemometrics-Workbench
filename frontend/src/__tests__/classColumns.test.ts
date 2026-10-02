/** #185: a PLS-DA classifies by a column with exactly two values. */
import { expect, it } from "vitest";

import { stratifiableColumns, twoValuedColumns } from "@/shell/classColumns";

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

it("offers for stratification only columns whose every level has two samples", () => {
  const columns = {
    grade: ["a", "a", "b", "b", "c", "c"],
    lone: ["a", "a", "b", "b", "b", "c"],
    site: ["x", "x", "x", "x", "x", "x"],
  };
  expect(stratifiableColumns(columns)).toEqual(["grade"]);
  expect(stratifiableColumns(undefined)).toEqual([]);
});
