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
  /** Removing a node happens on the node (#51): clicking one opens its tab,
   * so a control in the side panel is replaced by the tab before it can be
   * used. Absent on a draft and on the source, which cannot be removed. */
  onRemove?: () => void;
  /** Copy this node and everything below it into a sibling branch (#51).
   * Absent on the source, which cannot be duplicated. */
  onDuplicate?: () => void;
  /** Pick this node for a comparison, or unpick it. Offered only on terminal
   * estimator nodes: comparing anything else has nothing to put side by side. */
  onCompare?: () => void;
  /** Already picked, so the control reads as a toggle rather than a verb. */
  comparing?: boolean;
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

/** The layout the server sent, with somewhere sensible for anything it has
 * not placed.
 *
 * A node created on the canvas - a duplicate, or a step added to a branch -
 * exists before the server has ever seen it, so it has no stored position. It
 * used to fall back to the origin, which put every new node in the same place
 * and stacked copies on top of each other. Beside its parent is the answer the
 * server already gives for a node it generates a position for, at the same
 * 170/130 step, so a node that has just been added and one that has been
 * reloaded land in the same kind of place.
 */
export function placements(
  pipeline: Pipeline,
  layout: PipelineState["layout"] | undefined,
): Record<string, { x: number; y: number }> {
  const placed: Record<string, { x: number; y: number }> = { ...(layout ?? {}) };
  const taken = new Set(Object.keys(placed));

  // Parents before children: a node placed beside its parent needs the parent
  // placed first, and a chain of new nodes needs each link in turn.
  const order = [...pipeline.nodes].sort(
    (left, right) => depthOf(left.id, pipeline) - depthOf(right.id, pipeline),
  );
  for (const node of order) {
    if (placed[node.id]) continue;
    const parent = node.inputs[0] ? placed[node.inputs[0]] : undefined;
    const siblings = [...taken].filter((id) => id !== node.id).length;
    placed[node.id] = parent
      ? { x: parent.x + 170, y: parent.y + 130 * pending(node.id, pipeline, taken) }
      : { x: 40, y: 40 + 130 * siblings };
    taken.add(node.id);
  }
  return placed;
}

/** How many already-placed nodes share this one's parent, so a second new
 * child sits below the first rather than on it. */
function pending(id: string, pipeline: Pipeline, taken: Set<string>): number {
  const parent = pipeline.nodes.find((node) => node.id === id)?.inputs[0];
  if (!parent) return 0;
  return pipeline.nodes.filter(
    (node) => node.inputs[0] === parent && node.id !== id && taken.has(node.id),
  ).length;
}

function depthOf(id: string, pipeline: Pipeline, seen = new Set<string>()): number {
  if (seen.has(id)) return 0;
  seen.add(id);
  const parent = pipeline.nodes.find((node) => node.id === id)?.inputs[0];
  return parent ? 1 + depthOf(parent, pipeline, seen) : 0;
}

export function toNodes(
  pipeline: Pipeline,
  state: PipelineState | undefined,
  labelOf: (node: PipelineNode) => string,
  onRemove?: (id: string) => void,
  compare?: { ids: string[]; terminals: Set<string>; onCompare: (id: string) => void },
  onDuplicate?: (id: string) => void,
): FlowNode[] {
  const placed = placements(pipeline, state?.layout);
  return pipeline.nodes.map((node) => {
    const status = nodeStateOf(node.id, state);
    return {
      id: node.id,
      type: "workbench" as const,
      position: placed[node.id],
      data: {
        label: labelOf(node),
        type: node.type,
        parameters: parameterLine(node),
        state: status.state,
        progress: status.progress,
        footer: status.footer,
        onRemove:
          onRemove && node.type !== "source" ? () => onRemove(node.id) : undefined,
        onDuplicate:
          onDuplicate && node.type !== "source" ? () => onDuplicate(node.id) : undefined,
        onCompare:
          compare && node.type === "estimator" && compare.terminals.has(node.id)
            ? () => compare.onCompare(node.id)
            : undefined,
        comparing: compare?.ids.includes(node.id) ?? false,
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
  /** What the node actually is, in the shape `PUT /pipelines/{id}` takes.
   *
   * Until #108 a draft carried only the two display strings above, because
   * nothing was ever sent anywhere - the step list drew a picture. Saving one
   * means the draft has to *be* the node, so the payload travels with it and
   * the server's schema is what decides whether it is legal. */
  payload: {
    step?: { kind: string; [key: string]: unknown };
    spec?: { kind: string; [key: string]: unknown };
  };
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
