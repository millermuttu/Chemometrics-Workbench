/** #185, #274: a PLS-DA classifies by a column with two or more values. */
import { expect, it } from "vitest";

import { classColumns, stratifiableColumns } from "@/shell/classColumns";

it("keeps the columns with two or more values and nothing else", () => {
  const columns = {
    fat_class: ["high", "low", "low", "high"],
    batch: ["a", "b", "c", "a"],
    site: ["x", "x", "x", "x"],
    lot: [1, 2, 1, 2],
  };
  expect(classColumns(columns)).toEqual(["fat_class", "batch", "lot"]);
  expect(classColumns(undefined)).toEqual([]);
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
