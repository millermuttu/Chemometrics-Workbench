/** #215: what two pipeline snapshots differ by.
 *
 * `PROPOSAL.md` section 8.3's picture is one dataset forked into four
 * preprocessing paths; the thing worth seeing is that two of them differ only
 * in their scatter correction. These are the rules that make that sentence
 * true, checked without a browser.
 */
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import { changedFields, diffPipelines } from "@/lineage/diff";

const source: PipelineNode = {
  id: "source",
  type: "source",
  inputs: [],
  version_id: "1a2b3c4d-0000-4000-8000-000000000000",
};
const snv: PipelineNode = { id: "scatter", type: "preprocess", inputs: ["source"], step: { kind: "snv" } };
const msc: PipelineNode = {
  id: "scatter",
  type: "preprocess",
  inputs: ["source"],
  step: { kind: "msc", reference: "mean" },
};
const centre: PipelineNode = {
  id: "centre",
  type: "preprocess",
  inputs: ["scatter"],
  step: { kind: "mean_centre" },
};
const pls = (components: number): PipelineNode => ({
  id: "pls",
  type: "estimator",
  inputs: ["centre"],
  spec: { kind: "pls", n_components: components, algorithm: "nipals", target: "fat" },
});

describe("two recipes compared", () => {
  it("reports the same recipe as identical, whatever order it is given in", () => {
    const left = [source, snv, centre, pls(5)];
    const right = [source, snv, centre, pls(5)];
    const diff = diffPipelines(left, right);

    expect(diff.identical).toBe(true);
    expect(diff.differing).toBe(0);
    expect(diff.nodes.every((node) => node.change === "unchanged")).toBe(true);
    // The node list is the left run's order, so a reader follows the recipe
    // they started from.
    expect(diff.nodes.map((node) => node.id)).toEqual(["source", "scatter", "centre", "pls"]);
  });

  it("names the one node two otherwise identical models differ by", () => {
    // Section 8.3's case: SNV against MSC, everything else the same.
    const diff = diffPipelines([source, snv, centre, pls(5)], [source, msc, centre, pls(5)]);

    expect(diff.differing).toBe(1);
    const changed = diff.nodes.filter((node) => node.change !== "unchanged");
    expect(changed).toHaveLength(1);
    expect(changed[0].id).toBe("scatter");
    expect(changed[0].fields).toEqual(["kind", "reference"]);
  });

  it("tells a parameter change from an added node and from a removed one", () => {
    const edited = diffPipelines([source, snv, centre, pls(5)], [source, snv, centre, pls(7)]);
    const [changed] = edited.nodes.filter((node) => node.change === "changed");
    expect(changed.id).toBe("pls");
    expect(changed.fields).toEqual(["n_components"]);
    expect(changed.left?.spec?.n_components).toBe(5);
    expect(changed.right?.spec?.n_components).toBe(7);

    const added = diffPipelines([source, snv], [source, snv, centre]);
    expect(added.nodes.map((node) => [node.id, node.change])).toEqual([
      ["source", "unchanged"],
      ["scatter", "unchanged"],
      ["centre", "added"],
    ]);
    // A node only the right run has is appended, and carries no left side.
    expect(added.nodes[2].left).toBeUndefined();
    expect(added.nodes[2].right?.id).toBe("centre");

    const removed = diffPipelines([source, snv, centre], [source, snv]);
    expect(removed.nodes[2].change).toBe("removed");
    expect(removed.nodes[2].right).toBeUndefined();
  });

  it("reads a renamed node as one removed and one added, because that is what it is", () => {
    const renamed: PipelineNode = { ...snv, id: "snv_1" };
    const diff = diffPipelines([source, snv], [source, renamed]);

    expect(diff.differing).toBe(2);
    expect(diff.nodes.map((node) => [node.id, node.change])).toEqual([
      ["source", "unchanged"],
      ["scatter", "removed"],
      ["snv_1", "added"],
    ]);
  });

  it("notices a rewired input, which changes the science without changing a parameter", () => {
    const rewired: PipelineNode = { ...centre, inputs: ["source"] };
    const diff = diffPipelines([source, snv, centre], [source, snv, rewired]);
    const [changed] = diff.nodes.filter((node) => node.change === "changed");
    expect(changed.id).toBe("centre");
    expect(changed.fields).toEqual(["inputs"]);
  });

  it("compares the recipe and nothing else, which is why layout cannot move it", () => {
    // Canvas coordinates are not in a snapshot at all - they live outside
    // Pipeline.content_hash() so that moving a node changes neither the
    // science nor this diff (design/data-model.md).
    expect(Object.keys(snv)).not.toContain("position");
    expect(changedFields(snv, { ...snv })).toEqual([]);
  });
});
