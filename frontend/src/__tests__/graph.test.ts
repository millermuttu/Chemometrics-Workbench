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

import type { Pipeline, PipelineState } from "@/api/queries";
import { draftGraph, nodeStateOf, parameterLine, toEdges, toNodes } from "@/canvas/graph";

const FIXTURES = path.resolve(import.meta.dirname, "../../../stub/fixtures");
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
      { kind: "SNV", type: "preprocess", parameters: "" },
      { kind: "SG d1 w11", type: "preprocess", parameters: "" },
      { kind: "PCA", type: "estimator", parameters: "5 components" },
    ],
    { x: 40, y: 500 },
  );
  expect(nodes.map((node) => node.data.label)).toEqual(["SNV", "SG d1 w11", "PCA"]);
  expect(nodes.every((node) => node.data.state === "not_run")).toBe(true);
  expect(nodes.map((node) => node.position.x)).toEqual([40, 210, 380]);
  expect(edges).toHaveLength(2);
});
