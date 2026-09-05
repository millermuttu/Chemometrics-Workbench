/** The canvas's encodings: what each node says, and what each edge means.
 *
 * These are the artboard's rules, checked against the committed fixture. The
 * screen is the signature one and its states carry meaning - a stale result
 * must stay visible rather than vanish - so the rules are tested here rather
 * than only looked at.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import type { Pipeline, PipelineNode, PipelineState } from "@/api/queries";
import { withDrafts } from "@/canvas/PipelineCanvas";
import {
  draftGraph,
  nodeStateOf,
  parameterLine,
  placements,
  toEdges,
  toNodes,
  type DraftStep,
} from "@/canvas/graph";

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const read = <T,>(name: string) =>
  JSON.parse(readFileSync(path.join(FIXTURES, `${name}.json`), "utf8")) as T;

const pipeline = read<Pipeline>("pipeline");
const state = read<PipelineState>("pipeline_state");
const colours = { rule: "#D2DAD8", accent: "#0B6B62", stale: "#9A6206" };

it("carries all five states at once, which is what the artboard shows", () => {
  const states = new Set(Object.values(state.nodes).map((node) => node.state));
  for (const required of ["complete", "running", "stale", "failed", "not_run"]) {
    expect(states, required).toContain(required);
  }
});

describe("nodes", () => {
  it("read as what they do, with their parameters underneath", () => {
    const savgol = pipeline.nodes.find((node) => node.id === "savgol")!;
    expect(parameterLine(savgol)).toBe("window 11 · poly 2 · deriv 1");
    const split = pipeline.nodes.find((node) => node.id === "split_d")!;
    expect(parameterLine(split)).toBe("10 folds · shuffle · seed 42");
  });

  it("carry why they are stale and what failed, because that is the useful part", () => {
    expect(nodeStateOf("savgol", state).footer).toBe("edited - downstream stale");
    expect(nodeStateOf("pca_d", state).footer).toContain("rank 4");
  });

  it("take their position from the layout the server sends, not from the recipe", () => {
    const nodes = toNodes(pipeline, state, (node) => node.id);
    for (const node of nodes) {
      expect(node.position).toEqual(state.layout[node.id]);
    }
    // The layout is not part of what the pipeline hashes: moving a node must
    // not change the science (design/data-model.md).
    expect(Object.keys(pipeline)).not.toContain("layout");
  });
});

describe("edges", () => {
  const edges = toEdges(pipeline, state, colours, true);
  const edge = (id: string) => edges.find((candidate) => candidate.id === id)!;

  it("run accent and animated into a node a run is working on", () => {
    expect(edge("source->msc").style.stroke).toBe(colours.accent);
    expect(edge("source->msc").style.strokeWidth).toBe(1.8);
    expect(edge("source->msc").animated).toBe(true);
    expect(edge("centre_b->pca_b").style.stroke).toBe(colours.accent);
  });

  it("run dashed stale wherever either end is stale", () => {
    expect(edge("source->savgol").style).toMatchObject({
      stroke: colours.stale,
      strokeDasharray: "4 3",
    });
    expect(edge("savgol->autoscale_c").style.stroke).toBe(colours.stale);
  });

  it("run rule-coloured and still at rest", () => {
    expect(edge("snv->centre_a").style).toEqual({ stroke: colours.rule, strokeWidth: 1.4 });
    expect(edge("snv->centre_a").animated).toBe(false);
  });

  it("never animate when motion is not wanted", () => {
    expect(toEdges(pipeline, state, colours, false).every((item) => !item.animated)).toBe(true);
  });
});

it("lays a drafted pipeline out as a chain below the committed one", () => {
  const { nodes, edges } = draftGraph(
    [
      { kind: "SNV", type: "preprocess", parameters: "", payload: { step: { kind: "snv" } } },
      {
        kind: "SG d1 w11",
        type: "preprocess",
        parameters: "",
        payload: { step: { kind: "savgol", window_length: 11, polyorder: 2, deriv: 1 } },
      },
      {
        kind: "PCA",
        type: "estimator",
        parameters: "5 components",
        payload: { spec: { kind: "pca", n_components: 5 } },
      },
    ],
    { x: 40, y: 500 },
  );
  expect(nodes.map((node) => node.data.label)).toEqual(["SNV", "SG d1 w11", "PCA"]);
  expect(nodes.every((node) => node.data.state === "not_run")).toBe(true);
  expect(nodes.map((node) => node.position.x)).toEqual([40, 210, 380]);
  expect(edges).toHaveLength(2);
});

describe("withDrafts", () => {
  const source: PipelineNode = {
    id: "source",
    type: "source",
    inputs: [],
    version_id: "1a2b3c4d-0000-4000-8000-000000000000",
  };
  const snv: DraftStep = {
    kind: "SNV",
    type: "preprocess",
    parameters: "",
    payload: { step: { kind: "snv" } },
  };
  const pca: DraftStep = {
    kind: "PCA",
    type: "estimator",
    parameters: "",
    payload: { spec: { kind: "pca", n_components: 5 } },
  };

  it("chains the drafts onto the terminal node, carrying their payloads", () => {
    const nodes = withDrafts([source], [snv, pca]);

    expect(nodes.map((node) => node.id)).toEqual(["source", "snv", "pca"]);
    expect(nodes[1].inputs).toEqual(["source"]);
    expect(nodes[2].inputs).toEqual(["snv"]);
    // The payload travels: what is drawn is what is sent.
    expect(nodes[1].step).toEqual({ kind: "snv" });
    expect(nodes[2].spec).toEqual({ kind: "pca", n_components: 5 });
    expect(nodes[0]).toBe(source);
  });

  it("appends to the end that nothing consumes, not to the last in the list", () => {
    // A source and a branch off it: the terminal node is the branch's tip.
    const centre: PipelineNode = {
      id: "centre",
      type: "preprocess",
      inputs: ["source"],
      step: { kind: "mean_centre" },
    };
    const nodes = withDrafts([source, centre], [snv]);
    expect(nodes[2].inputs).toEqual(["centre"]);
  });

  it("does not collide ids when the same step is added twice", () => {
    const nodes = withDrafts([source], [snv, snv]);
    expect(nodes.map((node) => node.id)).toEqual(["source", "snv", "snv_2"]);
    expect(nodes[2].inputs).toEqual(["snv"]);
  });

  it("does not collide with an id the saved pipeline already uses", () => {
    const saved: PipelineNode[] = [
      source,
      { id: "snv", type: "preprocess", inputs: ["source"], step: { kind: "snv" } },
    ];
    expect(withDrafts(saved, [snv]).map((node) => node.id)).toEqual(["source", "snv", "snv_2"]);
  });

  it("returns the saved nodes untouched when there is nothing drafted", () => {
    const saved = [source];
    expect(withDrafts(saved, [])).toBe(saved);
  });
});

describe("a node the server has never placed", () => {
  // A duplicate, or a step added to a branch, exists on the canvas before the
  // server has seen it. It used to fall back to the origin, so every new node
  // appeared in the same spot and copies stacked on top of each other.
  const added: PipelineNode = { id: "msc_2", type: "preprocess", inputs: ["snv"], step: { kind: "msc" } };
  const extended: Pipeline = { ...pipeline, nodes: [...pipeline.nodes, added] };

  it("lands beside its parent, not at the origin", () => {
    const placed = placements(extended, state.layout);
    expect(placed.msc_2).not.toEqual({ x: 0, y: 0 });
    expect(placed.msc_2.x).toBe(placed.snv.x + 170);
  });

  it("leaves every placed node exactly where the server put it", () => {
    const placed = placements(extended, state.layout);
    for (const [id, position] of Object.entries(state.layout)) {
      expect(placed[id]).toEqual(position);
    }
  });

  it("does not stack two new children of the same parent", () => {
    const second: PipelineNode = { id: "msc_3", type: "preprocess", inputs: ["snv"], step: { kind: "msc" } };
    const placed = placements({ ...extended, nodes: [...extended.nodes, second] }, state.layout);
    expect(placed.msc_2).not.toEqual(placed.msc_3);
  });
});
