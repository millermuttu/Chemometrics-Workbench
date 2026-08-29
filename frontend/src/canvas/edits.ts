import type { PipelineNode } from "@/api/queries";

/** Editing the pipeline by hand: the graph operations, without React Flow.
 *
 * Pure and tested for the same reason `graph.ts` is - these are the rules
 * `models.py` enforces at `PUT` time, and a rule that is only checked by the
 * server is a rule the user meets as a rejected save rather than as a drag
 * that will not drop. DESIGN_BRIEF.md section 5 asks for the rules to be
 * enforced "during the drag rather than after it", which is what the two
 * refusal functions are for.
 *
 * **Every non-source node takes exactly one input** - `models.py` types it as
 * `tuple[NodeId]`, not a list. Two things follow, and they shape everything
 * here:
 *
 * - Connecting **replaces** the target's input rather than adding to it. A
 *   branch is one node with several *children*, which is what the artboard's
 *   fork of corn_raw into four preprocessing paths actually is.
 * - An edge cannot be deleted on its own: the child would be left with no
 *   input, which is not a pipeline. Edges are rewired, never cut.
 */

const parentOf = (node: PipelineNode): string | undefined => node.inputs[0];

/** Everything reachable downstream of `id`, `id` included. */
function descendants(nodes: PipelineNode[], id: string): Set<string> {
  const found = new Set([id]);
  // The list is small and a pipeline is shallow, so this repeats until it
  // stops growing rather than sorting the graph first.
  for (let grew = true; grew; ) {
    grew = false;
    for (const node of nodes) {
      const parent = parentOf(node);
      if (parent && found.has(parent) && !found.has(node.id)) {
        found.add(node.id);
        grew = true;
      }
    }
  }
  return found;
}

/** Why this connection is refused, in the words the canvas shows - or null.
 *
 * Written as a reason rather than a boolean because the canvas has somewhere
 * to put it: a refused drag that says nothing is indistinguishable from one
 * the application failed to notice.
 */
export function connectionRefusal(
  nodes: PipelineNode[],
  source: string,
  target: string,
): string | null {
  const from = nodes.find((node) => node.id === source);
  const to = nodes.find((node) => node.id === target);
  if (!from || !to) return "That node is not in this pipeline.";
  if (source === target) return "A node cannot feed itself.";
  if (to.type === "source") return "The source is where the data enters — it takes no input.";
  if (parentOf(to) === source) return `${target} already reads from ${source}.`;
  if (descendants(nodes, target).has(source)) {
    return `${source} is downstream of ${target}, so this would make a cycle.`;
  }
  return null;
}

/** Point `target` at `source`, replacing whatever it read before.
 *
 * Refusals are the caller's to check first - `isValidConnection` runs during
 * the drag, so by the time this is called the drop has already been allowed.
 * It throws rather than returning the list unchanged, because an edit that
 * silently does nothing is how a canvas starts disagreeing with what was saved.
 */
export function connect(
  nodes: PipelineNode[],
  source: string,
  target: string,
): PipelineNode[] {
  const refusal = connectionRefusal(nodes, source, target);
  if (refusal) throw new Error(refusal);
  return nodes.map((node) => (node.id === target ? { ...node, inputs: [source] } : node));
}

/** Why this node cannot be removed, or null. */
export function removalRefusal(nodes: PipelineNode[], id: string): string | null {
  const node = nodes.find((candidate) => candidate.id === id);
  if (!node) return "That node is not in this pipeline.";
  if (node.type === "source") {
    return "The source is where the data enters — it cannot be removed.";
  }
  return null;
}

/** Drop a node and reconnect its children to its parent.
 *
 * Deleting SNV from source → SNV → PCA leaves source → PCA, which is the
 * pipeline the user meant. Leaving PCA dangling instead would be a graph the
 * server refuses, discovered at save time.
 */
export function remove(nodes: PipelineNode[], id: string): PipelineNode[] {
  const refusal = removalRefusal(nodes, id);
  if (refusal) throw new Error(refusal);
  const orphaned = nodes.find((node) => node.id === id)!;
  const inherited = parentOf(orphaned)!;
  return nodes
    .filter((node) => node.id !== id)
    .map((node) => (parentOf(node) === id ? { ...node, inputs: [inherited] } : node));
}
