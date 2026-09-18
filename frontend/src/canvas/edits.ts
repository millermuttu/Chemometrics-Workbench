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

/** A new node hung off `parentId`.
 *
 * The parent is not guessed. `withDrafts` appends to the first terminal node
 * and its own docstring admits that choosing which branch to extend was left
 * undone; here the parent is the node whose connector was dragged, so the
 * question never arises.
 *
 * The id is derived from the step rather than random - `snv`, then `snv 2` -
 * because a pipeline is read by a person, and `msc` beside `snv` says what
 * happened where a uuid says only that something did.
 */
export function add(
  nodes: PipelineNode[],
  parentId: string,
  step: Pick<PipelineNode, "type"> & { step?: unknown; spec?: unknown },
  stem: string,
): PipelineNode[] {
  if (!nodes.some((node) => node.id === parentId)) {
    throw new Error("That node is not in this pipeline.");
  }
  const id = numberedId(new Set(nodes.map((node) => node.id)), stem);
  return [...nodes, { ...step, id, inputs: [parentId] } as PipelineNode];
}

/** An unused id derived from `stem`: `msc`, then `msc_2`. Distinct from
 * `freeId`, which reads as a copy of something; a node added from the menu is
 * not a copy of anything. */
export function numberedId(taken: Set<string>, stem: string): string {
  const base = stem.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  if (!taken.has(base)) return base;
  for (let n = 2; ; n += 1) {
    const candidate = `${base}_${n}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** Why this subgraph cannot be duplicated, or null. */
export function duplicationRefusal(nodes: PipelineNode[], id: string): string | null {
  const node = nodes.find((candidate) => candidate.id === id);
  if (!node) return "That node is not in this pipeline.";
  if (node.type === "source") {
    return "The source is where the data enters — duplicating it would copy the whole pipeline.";
  }
  return null;
}

/** An unused id derived from `base`: `snv copy`, then `snv copy 2`, and so on. */
function freeId(taken: Set<string>, base: string): string {
  const first = `${base} copy`;
  if (!taken.has(first)) return first;
  for (let n = 2; ; n += 1) {
    const candidate = `${first} ${n}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** Copy `id` and everything below it, hanging the copy off the same parent.
 *
 * **This is what the canvas is for.** DESIGN_BRIEF.md section 5 draws one
 * dataset forked into four competing preprocessing paths, and building the
 * fourth by hand when it differs from the third by one parameter is the work
 * this removes: duplicate the branch, then edit the copy.
 *
 * The copy reads from the original's parent, so it is a sibling branch rather
 * than a continuation - attaching it below the original would make a chain,
 * which is not what "duplicate" means on a graph whose point is comparison.
 *
 * Ids are derived rather than random. A pipeline is read by a person, and
 * `snv copy` beside `snv` says what happened; a uuid says only that something
 * did. `models.py` types `NodeId` as a plain string, so a space is legal.
 */
export function duplicate(nodes: PipelineNode[], id: string): PipelineNode[] {
  const refusal = duplicationRefusal(nodes, id);
  if (refusal) throw new Error(refusal);

  const copied = descendants(nodes, id);
  const taken = new Set(nodes.map((node) => node.id));
  const renamed = new Map<string, string>();
  // In the pipeline's own order, so a parent is always renamed before the
  // child that has to point at its new name.
  for (const node of nodes) {
    if (!copied.has(node.id)) continue;
    const fresh = freeId(taken, node.id);
    taken.add(fresh);
    renamed.set(node.id, fresh);
  }

  const copies = nodes
    .filter((node) => copied.has(node.id))
    .map((node) => {
      const parent = parentOf(node);
      return {
        ...node,
        id: renamed.get(node.id)!,
        // The root of the copy keeps the original's parent; everything below
        // it points at its own copied parent.
        inputs: parent ? [renamed.get(parent) ?? parent] : [],
      };
    });

  return [...nodes, ...copies];
}

/** The nodes nothing reads from — what DESIGN_BRIEF.md section 5 calls
 * terminal, and what a comparison is offered between. */
export function terminals(nodes: PipelineNode[]): PipelineNode[] {
  const consumed = new Set(nodes.flatMap((node) => node.inputs));
  return nodes.filter((node) => !consumed.has(node.id));
}
