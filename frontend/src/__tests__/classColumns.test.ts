/** #185, #274: a PLS-DA classifies by a column with two or more values. */
import { describe, expect, it } from "vitest";

import { classColumns, roleOf, stratifiableColumns } from "@/shell/classColumns";

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

describe("column roles (#371)", () => {
  const version = {
    targets: { fat: [1, 2] },
    metadata_columns: { batch: ["b0", "b1"], lot: ["7", "7"], variety: ["1", "2"] },
    column_roles: { batch: "group" as const },
  };

  it("reads a role the user chose, and the import's rule for the rest", () => {
    expect(roleOf(version, "fat")).toBe("target");
    expect(roleOf(version, "batch")).toBe("group");
    expect(roleOf(version, "variety")).toBe("class");
    expect(roleOf(version, "lot")).toBe("group");
  });

  it("does not offer a column made a group as a class", () => {
    expect(classColumns(version.metadata_columns, version.column_roles)).toEqual(["variety"]);
    expect(classColumns(version.metadata_columns)).toEqual(["batch", "variety"]);
  });
});
