/** Editing the graph by hand - the rules, without a browser.
 *
 * These are `models.py`'s DAG rules stated on the client so a drag that would
 * be refused at `PUT` time never drops. Each test that names a refusal is the
 * client half of a server validator; if one of those moves, this is where the
 * two stop agreeing.
 */
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import {
  add,
  connect,
  connectionRefusal,
  duplicate,
  duplicationRefusal,
  remove,
  removalRefusal,
  terminals,
} from "@/canvas/edits";

/** source → snv → pca, the shape the walkthrough builds. */
const chain: PipelineNode[] = [
  { id: "source", type: "source", inputs: [], version_id: "v1" },
  { id: "snv", type: "preprocess", inputs: ["source"], step: { kind: "snv" } },
  { id: "pca", type: "estimator", inputs: ["snv"], spec: { kind: "pca", n_components: 5 } },
];

describe("connecting", () => {
  it("forks one node into two paths, which is what a branch is", () => {
    // Every non-source node holds exactly one input, so a branch is a node
    // with two children - not a node with two parents.
    const forked = connect([...chain, { id: "msc", type: "preprocess", inputs: ["pca"], step: { kind: "msc", reference: "mean" } }], "source", "msc");
    expect(forked.filter((node) => node.inputs[0] === "source").map((node) => node.id)).toEqual([
      "snv",
      "msc",
    ]);
  });

  it("replaces the target's input rather than adding to it", () => {
    const rewired = connect(chain, "source", "pca");
    expect(rewired.find((node) => node.id === "pca")!.inputs).toEqual(["source"]);
  });

  it("leaves every other node untouched", () => {
    const rewired = connect(chain, "source", "pca");
    expect(rewired.filter((node) => node.id !== "pca")).toEqual(
      chain.filter((node) => node.id !== "pca"),
    );
  });

  it("refuses a cycle, naming which way round it would go", () => {
    expect(connectionRefusal(chain, "pca", "snv")).toContain("cycle");
    expect(() => connect(chain, "pca", "snv")).toThrow(/cycle/);
  });

  it("refuses a node feeding itself", () => {
    expect(connectionRefusal(chain, "snv", "snv")).toBe("A node cannot feed itself.");
  });

  it("refuses an input on the source, because that is where the data enters", () => {
    expect(connectionRefusal(chain, "snv", "source")).toContain("takes no input");
  });

  it("refuses an edge that already exists rather than redrawing it", () => {
    expect(connectionRefusal(chain, "source", "snv")).toBe("snv already reads from source.");
  });

  it("refuses a node that is not in the pipeline", () => {
    expect(connectionRefusal(chain, "source", "pls")).toContain("not in this pipeline");
  });

  it("allows a legal connection", () => {
    expect(connectionRefusal(chain, "source", "pca")).toBeNull();
  });
});

describe("removing", () => {
  it("reconnects the children to the parent, leaving nothing dangling", () => {
    // source → snv → pca, minus snv, is the pipeline the user meant.
    expect(remove(chain, "snv")).toEqual([
      chain[0],
      { id: "pca", type: "estimator", inputs: ["source"], spec: { kind: "pca", n_components: 5 } },
    ]);
  });

  it("reconnects every child of a branch point, not just the first", () => {
    const branched: PipelineNode[] = [
      ...chain,
      { id: "msc", type: "preprocess", inputs: ["snv"], step: { kind: "msc", reference: "mean" } },
    ];
    expect(remove(branched, "snv").map((node) => [node.id, node.inputs[0]])).toEqual([
      ["source", undefined],
      ["pca", "source"],
      ["msc", "source"],
    ]);
  });

  it("drops a terminal node without touching anything else", () => {
    expect(remove(chain, "pca")).toEqual(chain.slice(0, 2));
  });

  it("refuses to remove the source", () => {
    expect(removalRefusal(chain, "source")).toContain("cannot be removed");
    expect(() => remove(chain, "source")).toThrow(/cannot be removed/);
  });

  it("refuses a node that is not in the pipeline", () => {
    expect(removalRefusal(chain, "pls")).toContain("not in this pipeline");
  });
});

describe("duplicating a subgraph", () => {
  /** source → snv → centre → pca, plus a second branch, so a copy has to take
   * only what is below its root. */
  const forked: PipelineNode[] = [
    { id: "source", type: "source", inputs: [], version_id: "v1" },
    { id: "snv", type: "preprocess", inputs: ["source"], step: { kind: "snv" } },
    { id: "centre", type: "preprocess", inputs: ["snv"], step: { kind: "mean_centre" } },
    { id: "pca", type: "estimator", inputs: ["centre"], spec: { kind: "pca", n_components: 5 } },
    { id: "msc", type: "preprocess", inputs: ["source"], step: { kind: "msc" } },
  ];

  it("copies the root and everything below it, and nothing beside it", () => {
    const after = duplicate(forked, "snv");
    const added = after.filter((node) => !forked.some((original) => original.id === node.id));

    expect(added.map((node) => node.id)).toEqual(["snv copy", "centre copy", "pca copy"]);
    // msc is a sibling of snv, not a descendant, so it is untouched.
    expect(added.some((node) => node.id.startsWith("msc"))).toBe(false);
  });

  it("hangs the copy off the original's parent, making a branch not a chain", () => {
    const after = duplicate(forked, "snv");

    expect(after.find((node) => node.id === "snv copy")!.inputs).toEqual(["source"]);
    // Everything below points at its own copy, never back at the original.
    expect(after.find((node) => node.id === "centre copy")!.inputs).toEqual(["snv copy"]);
    expect(after.find((node) => node.id === "pca copy")!.inputs).toEqual(["centre copy"]);
  });

  it("leaves the original branch exactly as it was", () => {
    const after = duplicate(forked, "snv");
    for (const original of forked) {
      expect(after.find((node) => node.id === original.id)).toEqual(original);
    }
  });

  it("numbers a second copy rather than colliding with the first", () => {
    const ids = duplicate(duplicate(forked, "snv"), "snv").map((node) => node.id);

    expect(ids).toContain("snv copy");
    expect(ids).toContain("snv copy 2");
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("copies a leaf on its own", () => {
    const after = duplicate(forked, "pca");
    expect(after.find((node) => node.id === "pca copy")!.inputs).toEqual(["centre"]);
    expect(after).toHaveLength(forked.length + 1);
  });

  it("refuses the source, and says why", () => {
    expect(duplicationRefusal(forked, "source")).toMatch(/data enters/);
    expect(() => duplicate(forked, "source")).toThrow(/data enters/);
  });

  it("refuses a node that is not in the pipeline", () => {
    expect(duplicationRefusal(forked, "ghost")).toMatch(/not in this pipeline/);
  });

  it("allows every other node", () => {
    for (const id of ["snv", "centre", "pca", "msc"]) {
      expect(duplicationRefusal(forked, id)).toBeNull();
    }
  });
});

describe("terminal nodes", () => {
  it("are the ones nothing reads from", () => {
    const forked: PipelineNode[] = [
      { id: "source", type: "source", inputs: [], version_id: "v1" },
      { id: "snv", type: "preprocess", inputs: ["source"], step: { kind: "snv" } },
      { id: "pca_a", type: "estimator", inputs: ["snv"], spec: { kind: "pca", n_components: 5 } },
      { id: "pca_b", type: "estimator", inputs: ["snv"], spec: { kind: "pca", n_components: 3 } },
    ];
    expect(terminals(forked).map((node) => node.id)).toEqual(["pca_a", "pca_b"]);
  });

  it("is the last node of a plain chain", () => {
    expect(terminals(chain).map((node) => node.id)).toEqual(["pca"]);
  });
});

describe("adding a step", () => {
  // The parent is the node whose connector was dragged, so it is never the
  // "first terminal node" guess `withDrafts` makes and admits to.
  it("hangs the new node off the parent it was given, not off an end", () => {
    const grown = add(chain, "source", { type: "preprocess", step: { kind: "msc" } }, "MSC");
    expect(grown).toHaveLength(4);
    expect(grown.at(-1)!.inputs).toEqual(["source"]);
    // The branch it was added to is untouched: this is a fork, not an insert.
    expect(grown.find((node) => node.id === "snv")!.inputs).toEqual(["source"]);
  });

  it("derives an id from the step, and does not collide with itself", () => {
    const once = add(chain, "snv", { type: "preprocess", step: { kind: "msc" } }, "MSC");
    const twice = add(once, "snv", { type: "preprocess", step: { kind: "msc" } }, "MSC");
    const added = twice.slice(chain.length).map((node) => node.id);
    expect(added).toEqual(["msc", "msc_2"]);
    expect(new Set(twice.map((node) => node.id)).size).toBe(twice.length);
  });

  it("refuses a parent that is not in the pipeline", () => {
    expect(() => add(chain, "ghost", { type: "preprocess", step: { kind: "msc" } }, "MSC")).toThrow(
      "not in this pipeline",
    );
  });

  it("can add a split, which the side list cannot draft", () => {
    const split = add(
      chain,
      "snv",
      { type: "split", spec: { kind: "kfold", n_splits: 10, shuffle: true, seed: 42 } },
      "K-fold 10",
    );
    expect(split.at(-1)!.type).toBe("split");
    expect(split.at(-1)!.id).toBe("k_fold_10");
  });
});
