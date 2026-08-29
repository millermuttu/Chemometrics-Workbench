import { Background, BackgroundVariant, ReactFlow, type Node } from "@xyflow/react";
import { useMemo, useRef, useState } from "react";

import { ApiError, api } from "@/api/client";
import type { PipelineNode } from "@/api/queries";
import { usePipeline, usePipelineState, useSavePipeline } from "@/api/queries";
import { NodeCard } from "@/canvas/NodeCard";
import { StepList } from "@/canvas/StepList";
import { connect, connectionRefusal, remove } from "@/canvas/edits";
import { draftGraph, toEdges, toNodes, type DraftStep } from "@/canvas/graph";
import { nodeLabel } from "@/shell/Sidebar";

import "@xyflow/react/dist/style.css";

/** The signature screen: the pipeline as a graph.
 *
 * Editable since #51: a branch is dragged from a node's output port, and a
 * node is removed with its children reconnected to its parent. The step list
 * beside it stays - appending a linear chain is what it is good at, and
 * DESIGN_BRIEF.md section 5 wants the *branch* to be the direct-manipulation
 * case.
 *
 * The rules live in `edits.ts` and are enforced **during the drag**: a
 * connection that would make a cycle will not drop, rather than being refused
 * by the server after a save. Nothing here computes a graph rule of its own.
 *
 * Edits are held locally until Save, which is the same whole-list `PUT` the
 * step list uses (#108). Node positions are still not draggable: layout lives
 * in `pipeline_state.json` and nothing writes it back yet.
 */

const NODE_TYPES = { workbench: NodeCard };

/** Motion is an accent on a running path, never decoration, and never at all
 * for someone who asked the system not to move things. */
function usesMotion(): boolean {
  return !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** The saved nodes with the drafts chained onto the end of the graph.
 *
 * Drafts attach to a **terminal node** - one nothing else consumes - because
 * that is the only end a list can be appended to. A freshly imported project
 * has exactly one, its source, which is the shape the step list exists for. A
 * pipeline with several branches has several ends, and this takes the first;
 * choosing which branch to extend needs somewhere to say so, and that is #51's
 * direct manipulation rather than a guess made here.
 *
 * Ids are derived from the step and the position so that adding a second SNV
 * does not collide with the first. The server refuses duplicates anyway - the
 * point is to not send it something it has to refuse.
 */
export function withDrafts(saved: PipelineNode[], drafts: DraftStep[]): PipelineNode[] {
  if (drafts.length === 0) return saved;
  const consumed = new Set(saved.flatMap((node) => node.inputs));
  const taken = new Set(saved.map((node) => node.id));
  let parent = (saved.find((node) => !consumed.has(node.id)) ?? saved[saved.length - 1]).id;

  const added = drafts.map((draft) => {
    const stem = draft.kind.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
    let id = stem;
    for (let suffix = 2; taken.has(id); suffix += 1) id = `${stem}_${suffix}`;
    taken.add(id);

    const node: PipelineNode = { id, type: draft.type, inputs: [parent], ...draft.payload };
    parent = id;
    return node;
  });
  return [...saved, ...added];
}


export function PipelineCanvas({ onOpenNode }: { onOpenNode: (id: string, label: string) => void }) {
  const pipeline = usePipeline();
  const state = usePipelineState();
  const save = useSavePipeline();
  const [steps, setSteps] = useState<DraftStep[]>([]);
  const [validation, setValidation] = useState<string | null>(null);
  /** The edited graph, or null while it still matches what the server holds. */
  const [edited, setEdited] = useState<PipelineNode[] | null>(null);
  const [selected, setSelected] = useState<{ id: string; label: string } | null>(null);
  // React Flow asks whether a connection is valid many times during one drag
  // and never reports the drop it refused. The reason for the last refusal is
  // kept here so the end of the drag can say why nothing happened.
  const refusal = useRef<string | null>(null);

  // Memoised because it feeds the graph's useMemo: a fresh [] every render
  // would rebuild the whole graph on every keystroke elsewhere in the tab.
  const nodes = useMemo(
    () => edited ?? pipeline.data?.nodes ?? [],
    [edited, pipeline.data],
  );

  const graph = useMemo(() => {
    if (!pipeline.data) return { nodes: [], edges: [] };
    const style = getComputedStyle(document.documentElement);
    const token = (name: string) => style.getPropertyValue(`--${name}`).trim() || "currentColor";
    const committed = {
      nodes: toNodes({ ...pipeline.data, nodes }, state.data, nodeLabel),
      edges: toEdges(
        { ...pipeline.data, nodes },
        state.data,
        { rule: token("rule"), accent: token("accent"), stale: token("stale") },
        usesMotion(),
      ),
    };
    const lowest = Math.max(...committed.nodes.map((node) => node.position.y), 0);
    const draft = draftGraph(steps, { x: 40, y: lowest + 150 });
    return { nodes: [...committed.nodes, ...draft.nodes], edges: [...committed.edges, ...draft.edges] };
  }, [pipeline.data, state.data, steps, nodes]);

  /** An edit that a rule refuses says so, rather than throwing into the void. */
  const edit = (apply: () => PipelineNode[]) => {
    try {
      setEdited(apply());
      setValidation(null);
    } catch (error) {
      setValidation(error instanceof Error ? error.message : "That edit is not allowed.");
    }
  };

  return (
    <div className="pane" style={{ flexDirection: "row" }}>
      <div style={{ flex: 1, minWidth: 0, background: "var(--surface)" }} data-testid="pipeline-canvas">
        <ReactFlow
          nodes={graph.nodes}
          edges={graph.edges}
          nodeTypes={NODE_TYPES}
          fitView
          nodesDraggable={false}
          nodesConnectable
          proOptions={{ hideAttribution: true }}
          isValidConnection={(connection) => {
            const { source, target } = connection;
            if (!source || !target) return false;
            // A draft belongs to the step list, which is where it is edited.
            if (source.startsWith("draft-") || target.startsWith("draft-")) {
              refusal.current = "A draft step is edited in the list until it is saved.";
              return false;
            }
            refusal.current = connectionRefusal(nodes, source, target);
            return refusal.current === null;
          }}
          onConnect={({ source, target }) => {
            if (source && target) edit(() => connect(nodes, source, target));
          }}
          onConnectEnd={() => {
            if (refusal.current) setValidation(refusal.current);
            refusal.current = null;
          }}
          onNodeClick={(_, node: Node) => {
            // Selecting a node focuses its tab - the mechanism that ties the
            // graph to the pages.
            if (String(node.id).startsWith("draft-")) return;
            const label = String((node.data as { label: string }).label);
            setSelected({ id: node.id, label });
            onOpenNode(node.id, label);
          }}
        >
          {/* The artboard's ground: a 22px dot grid in --grid. */}
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--grid)" />
        </ReactFlow>
      </div>

      <StepList
        steps={steps}
        saving={save.isPending}
        edited={edited !== null}
        selected={selected && nodes.some((node) => node.id === selected.id) ? selected : null}
        onRemove={(id) => {
          edit(() => remove(nodes, id));
          setSelected(null);
        }}
        onSave={async () => {
          if (!pipeline.data) return;
          try {
            await save.mutateAsync(withDrafts(nodes, steps));
            // The drafts are nodes now; keeping them would draw each one twice,
            // and the edits are what the server holds.
            setSteps([]);
            setEdited(null);
            setValidation(null);
          } catch (error) {
            setValidation(error instanceof ApiError ? error.message : "Could not save.");
          }
        }}
        onChange={(next) => {
          setSteps(next);
          setValidation(null);
        }}
        onValidate={async () => {
          const result = await api<{ valid: boolean; problems: string[] }>(
            "/pipelines/current/validate",
            { method: "POST" },
          );
          setValidation(result.valid ? `valid · ${steps.length} steps` : result.problems.join(" · "));
        }}
        validation={validation}
      />
    </div>
  );
}
