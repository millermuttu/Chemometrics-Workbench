/** #182: the drop menu's PLS entry models a column the dataset has, or is
 * not offered. It used to write `target: "fat"` into every PLS node. */
import { describe, expect, it } from "vitest";

import { stepMenu } from "@/canvas/catalogue";

describe("the PLS menu entry", () => {
  it("models the dataset's first target", () => {
    const pls = stepMenu(["moisture", "fat"]).find((step) => step.kind === "PLS 5 LV")!;
    expect(pls.payload.spec).toMatchObject({ kind: "pls", target: "moisture" });
    expect(pls.parameters).toBe("5 components · moisture");
  });

  it("offers PLS-DA only for a dataset with a two-valued column, on that column", () => {
    const plsda = stepMenu(["fat"], ["fat_class"]).find((step) => step.kind === "PLS-DA 5 LV")!;
    expect(plsda.payload.spec).toMatchObject({ kind: "plsda", class_column: "fat_class" });
    expect(stepMenu(["fat"]).map((step) => step.kind)).not.toContain("PLS-DA 5 LV");
  });

  it("is absent when the dataset has nothing to model", () => {
    const kinds = stepMenu([]).map((step) => step.kind);
    expect(kinds).not.toContain("PLS 5 LV");
    expect(kinds).toContain("K-fold 10");
    expect(kinds).toContain("Train/test 25%");
  });
});
