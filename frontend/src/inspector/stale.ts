import type { Pipeline } from "@/api/queries";

/** Editing a parameter invalidates what was computed from it.
 *
 * Everything downstream of the edited node becomes stale - not deleted, not
 * hidden. A stale result stays readable; that is the whole reason the state
 * exists and why the canvas draws it dimmed rather than dropping it.
 */
export function downstreamOf(pipeline: Pipeline, nodeId: string): string[] {
  const stale = new Set<string>();
  let frontier = [nodeId];
  while (frontier.length > 0) {
    const next: string[] = [];
    for (const node of pipeline.nodes) {
      if (stale.has(node.id)) continue;
      if (node.inputs.some((input) => frontier.includes(input))) {
        stale.add(node.id);
        next.push(node.id);
      }
    }
    frontier = next;
  }
  return [...stale];
}
