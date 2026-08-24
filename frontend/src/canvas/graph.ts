import type { Pipeline, PipelineNode, PipelineState } from "@/api/queries";

/** Pipeline plus run state plus layout, turned into what React Flow draws.
 *
 * Pure, and tested: the state and edge encodings are the artboard's rules and
 * they should not need a browser to check. Layout coordinates come from the
 * server alongside the pipeline and never from inside `content_hash()` -
 * design/data-model.md is explicit that moving a node must not change the
 * science.
 */

export type NodeState = "complete" | "running" | "queued" | "stale" | "failed" | "not_run";

export interface NodeData extends Record<string, unknown> {
  label: string;
  type: string;
  parameters: string;
  footer?: string;
  state: NodeState;
  progress?: number;
  draft?: boolean;
}

export interface FlowNode {
  id: string;
  type: "workbench";
  position: { x: number; y: number };
  data: NodeData;
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  animated: boolean;
  style: { stroke: string; strokeWidth: number; strokeDasharray?: string };
}

/** "window 11 · poly 2 · deriv 1" - what the artboard's node bodies carry. */
export function parameterLine(node: PipelineNode): string {
  const step = node.step as Record<string, unknown> | undefined;
  const spec = node.spec as Record<string, unknown> | undefined;
  const kind = step?.kind ?? spec?.kind ?? node.type;
  switch (kind) {
    case "savgol":
      return `window ${step!.window_length} · poly ${step!.polyorder} · deriv ${step!.deriv}`;
    case "msc":
      return `reference: ${step!.reference}`;
    case "autoscale":
      return `ddof ${step!.ddof}`;
    case "kfold":
      return `${spec!.n_splits} folds · ${spec!.shuffle ? "shuffle · " : ""}seed ${spec!.seed}`;
    case "pca":
      return `${spec!.n_components} components`;
    case "snv":
      return "population statistics per row";
    case "mean_centre":
      return "column means";
    default:
      return node.id;
  }
}

export function nodeStateOf(
  id: string,
  state: PipelineState | undefined,
): { state: NodeState; progress?: number; footer?: string } {
  const entry = state?.nodes[id];
  if (!entry) return { state: "not_run" };
  return {
    state: entry.state as NodeState,
    progress: entry.progress,
    // Stale carries why it is stale; failed carries what went wrong. Both are
    // footers in the artboard, and both matter more than the state's name.
    footer: entry.reason ?? entry.message,
  };
}

export function toNodes(
  pipeline: Pipeline,
  state: PipelineState | undefined,
  labelOf: (node: PipelineNode) => string,
): FlowNode[] {
  return pipeline.nodes.map((node) => {
    const status = nodeStateOf(node.id, state);
    return {
      id: node.id,
      type: "workbench" as const,
      position: state?.layout?.[node.id] ?? { x: 0, y: 0 },
      data: {
        label: labelOf(node),
        type: node.type,
        parameters: parameterLine(node),
        state: status.state,
        progress: status.progress,
        footer: status.footer,
      },
    };
  });
}

/** The edge encoding, from the artboard:
 *
 * - 1.8px `--accent` along the path a run is currently on - an edge feeding a
 *   running or queued node
 * - dashed 4 3 in `--stale` wherever either end is stale, because a stale
 *   result must stay visible rather than vanish
 * - 1.4px `--rule` at rest
 *
 * Animation rides on the accent path only, and the caller turns it off under
 * prefers-reduced-motion.
 */
export function toEdges(
  pipeline: Pipeline,
  state: PipelineState | undefined,
  colours: { rule: string; accent: string; stale: string },
  animate: boolean,
): FlowEdge[] {
  const stateOf = (id: string) => state?.nodes[id]?.state;
  return pipeline.nodes.flatMap((node) =>
    node.inputs.map((input) => {
      const source = stateOf(input);
      const target = stateOf(node.id);
      const stale = source === "stale" || target === "stale";
      const active = target === "running" || target === "queued";
      return {
        id: `${input}->${node.id}`,
        source: input,
        target: node.id,
        animated: active && animate,
        style: stale
          ? { stroke: colours.stale, strokeWidth: 1.4, strokeDasharray: "4 3" }
          : active
            ? { stroke: colours.accent, strokeWidth: 1.8 }
            : { stroke: colours.rule, strokeWidth: 1.4 },
      };
    }),
  );
}

/** The step list's draft, as nodes and edges below the committed graph.
 *
 * 1.1 builds a pipeline through this list rather than by dragging; direct
 * manipulation is #51. A draft has run no steps, so every node is not_run.
 */
export interface DraftStep {
  kind: string;
  parameters: string;
  type: "preprocess" | "estimator";
}

export function draftGraph(
  steps: DraftStep[],
  origin: { x: number; y: number },
): { nodes: FlowNode[]; edges: FlowEdge[] } {
  const nodes = steps.map((step, index) => ({
    id: `draft-${index}`,
    type: "workbench" as const,
    position: { x: origin.x + 170 * index, y: origin.y },
    data: {
      label: step.kind,
      type: step.type,
      parameters: step.parameters,
      state: "not_run" as const,
      draft: true,
    },
  }));
  const edges = nodes.slice(1).map((node, index) => ({
    id: `draft-${index}->${index + 1}`,
    source: nodes[index].id,
    target: node.id,
    animated: false,
    style: { stroke: "currentColor", strokeWidth: 1.4, strokeDasharray: "4 3" },
  }));
  return { nodes, edges };
}
