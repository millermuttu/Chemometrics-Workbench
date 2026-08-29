/** The inspector's two rules that must not drift.
 *
 * One: the parameter form is generated from `models.py`'s schema, so its
 * bounds are the model's bounds. Two: editing a parameter marks everything
 * computed from it stale - and stale means dimmed, never deleted.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import type { Pipeline } from "@/api/queries";
import { checkBounds, specFor, stepSpecs, type StepSchema } from "@/inspector/schema";
import { downstreamOf } from "@/inspector/stale";

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const read = <T,>(name: string) =>
  JSON.parse(readFileSync(path.join(FIXTURES, `${name}.json`), "utf8")) as T;

const schema = read<StepSchema>("step_schema");
const pipeline = read<Pipeline>("pipeline");

describe("the form is generated, not restated", () => {
  it("covers every preprocessing step the schema can express", () => {
    const kinds = stepSpecs(schema).map((spec) => spec.kind);
    expect(kinds).toEqual(
      expect.arrayContaining([
        "snv",
        "msc",
        "savgol",
        "mean_centre",
        "autoscale",
        "normalise",
        "baseline",
        "range_select",
      ]),
    );
  });

  it("takes each field's bounds from the model", () => {
    const savgol = specFor(schema, "savgol")!;
    const deriv = savgol.fields.find((field) => field.name === "deriv")!;
    expect(deriv.kind).toBe("integer");
    expect([deriv.minimum, deriv.maximum]).toEqual([0, 2]);

    const window = savgol.fields.find((field) => field.name === "window_length")!;
    expect(window.exclusiveMinimum).toBe(2);
  });

  it("reads an enum as a choice and an optional field as optional", () => {
    const msc = specFor(schema, "msc")!;
    expect(msc.fields[0].options).toEqual(["mean", "median"]);

    const baseline = specFor(schema, "baseline")!;
    expect(baseline.fields.find((field) => field.name === "lam")!.optional).toBe(true);
    expect(baseline.fields.find((field) => field.name === "method")!.optional).toBe(false);
  });

  it("refuses a value the model would refuse, with a specific message", () => {
    const savgol = specFor(schema, "savgol")!;
    const deriv = savgol.fields.find((field) => field.name === "deriv")!;
    expect(checkBounds(deriv, 3)).toBe("Deriv must be at most 2");
    expect(checkBounds(deriv, 1.5)).toBe("Deriv must be a whole number");
    expect(checkBounds(deriv, 1)).toBeNull();

    const p = specFor(schema, "baseline")!.fields.find((field) => field.name === "p")!;
    expect(checkBounds(p, 1)).toBe("P must be less than 1");
    expect(checkBounds(p, "")).toBeNull(); // optional
  });

  // The cross-field rules are deliberately absent here: they live in
  // model_validator, have no JSON Schema form, and are checked by the server
  // so the message is the model's own.
  it("does not pretend to know the cross-field rules", () => {
    const savgol = specFor(schema, "savgol")!;
    const window = savgol.fields.find((field) => field.name === "window_length")!;
    expect(checkBounds(window, 10)).toBeNull();
  });
});

describe("editing marks downstream stale", () => {
  it("follows every branch below the edited node", () => {
    expect(downstreamOf(pipeline, "snv").sort()).toEqual(
      ["centre_a", "pca_a", "snv_savgol", "split_d", "centre_d", "pca_d"].sort(),
    );
  });

  it("stops at a leaf and never includes the node itself", () => {
    expect(downstreamOf(pipeline, "pca_a")).toEqual([]);
    expect(downstreamOf(pipeline, "msc")).toEqual(["centre_b", "pca_b"]);
  });

  it("marks the whole graph when the source is edited", () => {
    expect(downstreamOf(pipeline, "source")).toHaveLength(pipeline.nodes.length - 1);
  });
});
