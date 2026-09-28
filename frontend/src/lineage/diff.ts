/** What two pipeline snapshots differ by (#215).
 *
 * `PROPOSAL.md` section 8.3 wants a comparison that shows *exactly which steps
 * differ* between two models, and says it is free once the data model is
 * correct. It is: an `Experiment` carries its pipeline by value, so comparing
 * two runs is comparing two recipes rather than reconstructing what the canvas
 * used to look like.
 *
 * Pure, and tested without a browser. **Nothing here is drawn and nothing here
 * is a score** - what each run scored comes from its own record.
 */
import type { PipelineNode } from "@/api/queries";

export type NodeChange = "added" | "removed" | "changed" | "unchanged";

export interface NodeDiff {
  id: string;
  change: NodeChange;
  /** The node as the left run ran it, absent when the left run had none. */
  left?: PipelineNode;
  right?: PipelineNode;
  /** For a changed node, which parts differ: `type`, `inputs`, or the name of
   * a parameter. Empty for every other change. */
  fields: string[];
}

export interface PipelineDiff {
  nodes: NodeDiff[];
  identical: boolean;
  /** How many nodes are not `unchanged` - what a header counts. */
  differing: number;
}

/** A node's parameters, whichever field carries them.
 *
 * Preprocessing nodes carry `step`; estimators and splits carry `spec`. The
 * discriminator is not a parameter: it names the kind, and a node whose kind
 * changed is reported on `kind` rather than on every field at once.
 */
function parameters(node: PipelineNode): Record<string, unknown> {
  return (node.step ?? node.spec ?? {}) as Record<string, unknown>;
}

/** Which parts of one node differ from the other's.
 *
 * Deliberately shallow-compares each parameter with `JSON.stringify`: a step's
 * parameters are numbers, strings and booleans in `models.py`, and a deep
 * comparison would be machinery for a case the schema does not have.
 */
export function changedFields(left: PipelineNode, right: PipelineNode): string[] {
  const fields: string[] = [];
  if (left.type !== right.type) fields.push("type");
  if (JSON.stringify(left.inputs) !== JSON.stringify(right.inputs)) fields.push("inputs");

  const before = parameters(left);
  const after = parameters(right);
  for (const name of [...new Set([...Object.keys(before), ...Object.keys(after)])].sort()) {
    if (JSON.stringify(before[name]) !== JSON.stringify(after[name])) fields.push(name);
  }
  return fields;
}

/** The two recipes, node by node.
 *
 * **Matched by id.** A node is the same node when it has the same id; a
 * renamed one reads as one removed and one added, which is what it is. The
 * order is the left run's, with nodes only the right run has appended - so a
 * reader follows the recipe they started from and sees what was added to it.
 */
export function diffPipelines(
  left: PipelineNode[],
  right: PipelineNode[],
): PipelineDiff {
  const byIdLeft = new Map(left.map((node) => [node.id, node]));
  const byIdRight = new Map(right.map((node) => [node.id, node]));

  const order = [...left.map((node) => node.id), ...right.map((node) => node.id).filter((id) => !byIdLeft.has(id))];

  const nodes: NodeDiff[] = order.map((id) => {
    const before = byIdLeft.get(id);
    const after = byIdRight.get(id);
    if (before && !after) return { id, change: "removed", left: before, fields: [] };
    if (!before && after) return { id, change: "added", right: after, fields: [] };

    const fields = changedFields(before!, after!);
    return {
      id,
      change: fields.length > 0 ? "changed" : "unchanged",
      left: before,
      right: after,
      fields,
    };
  });

  const differing = nodes.filter((node) => node.change !== "unchanged").length;
  return { nodes, identical: differing === 0, differing };
}
