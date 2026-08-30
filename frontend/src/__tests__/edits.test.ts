/** Editing the graph by hand - the rules, without a browser.
 *
 * These are `models.py`'s DAG rules stated on the client so a drag that would
 * be refused at `PUT` time never drops. Each test that names a refusal is the
 * client half of a server validator; if one of those moves, this is where the
 * two stop agreeing.
 */
import { describe, expect, it } from "vitest";

import type { PipelineNode } from "@/api/queries";
import { connect, connectionRefusal, remove, removalRefusal } from "@/canvas/edits";

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
