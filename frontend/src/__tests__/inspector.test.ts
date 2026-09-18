/** The inspector's rule that must not drift: the parameter form is generated
 * from `models.py`'s schema, so its bounds are the model's bounds.
 *
 * Staleness is not the inspector's to compute. The server derives a node's
 * state from whether its result exists under its cache key, so an edited
 * node's descendants come back `not_run` (#83, #181).
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { checkBounds, specFor, stepSpecs, type StepSchema } from "@/inspector/schema";

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const read = <T,>(name: string) =>
  JSON.parse(readFileSync(path.join(FIXTURES, `${name}.json`), "utf8")) as T;

const schema = read<StepSchema>("step_schema");

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

/** #182: the same form serves estimators and splits, whose schema carries
 * things a preprocessing step's never did - a referenced enum, a boolean, and
 * a bare string for the PLS target. Built inline rather than read from the
 * contract fixture, which is the 1.1 preprocessing-only shape and stays so. */
describe("an estimator's or a split's spec", () => {
  const withSpecs: StepSchema = {
    $defs: {
      PLSAlgorithm: { title: "PLSAlgorithm", enum: ["nipals", "simpls"] },
      PLSRegressionSpec: {
        title: "PLSRegressionSpec",
        properties: {
          kind: { const: "pls", type: "string" },
          n_components: { type: "integer", minimum: 1, title: "N Components" },
          algorithm: { $ref: "#/$defs/PLSAlgorithm", default: "nipals" },
          target: { type: "string", title: "Target" },
        },
      },
      KFoldSplit: {
        title: "KFoldSplit",
        properties: {
          kind: { const: "kfold", type: "string" },
          n_splits: { type: "integer", minimum: 2, title: "N Splits" },
          shuffle: { type: "boolean", default: true, title: "Shuffle" },
        },
      },
    },
  };

  it("is a kind of node only when it carries a kind - a bare enum is not one", () => {
    expect(stepSpecs(withSpecs).map((spec) => spec.kind)).toEqual(["pls", "kfold"]);
  });

  it("resolves a referenced enum into a choice", () => {
    const algorithm = specFor(withSpecs, "pls")!.fields.find((f) => f.name === "algorithm")!;
    expect(algorithm.kind).toBe("enum");
    expect(algorithm.options).toEqual(["nipals", "simpls"]);
  });

  it("reads a string and a boolean as what they are, and bounds neither by number", () => {
    const target = specFor(withSpecs, "pls")!.fields.find((f) => f.name === "target")!;
    expect(target.kind).toBe("string");
    expect(checkBounds(target, "fat")).toBeNull();
    expect(checkBounds(target, "")).toBe("Target is required");

    const shuffle = specFor(withSpecs, "kfold")!.fields.find((f) => f.name === "shuffle")!;
    expect(shuffle.kind).toBe("boolean");
    expect(checkBounds(shuffle, "true")).toBeNull();
    expect(checkBounds(shuffle, "yes")).toBe("Shuffle must be true or false");
  });
});
