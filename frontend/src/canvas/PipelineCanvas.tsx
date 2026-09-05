import {
  Background,
  BackgroundVariant,
  ReactFlow,
  type Node,
  type NodeChange,
  type ReactFlowInstance,
} from "@xyflow/react";
import { useCallback, useMemo, useRef, useState } from "react";

import { ApiError, api } from "@/api/client";
import type { PipelineNode } from "@/api/queries";
import { usePipeline, usePipelineState, useSaveLayout, useSavePipeline } from "@/api/queries";
import { STEP_MENU } from "@/canvas/catalogue";
import { NodeCard } from "@/canvas/NodeCard";
import { StepList } from "@/canvas/StepList";
import {
  add,
  connect,
  connectionRefusal,
  duplicate,
  numberedId,
  remove,
  terminals,
} from "@/canvas/edits";
import {
  draftGraph,
  toEdges,
  toNodes,
  type DraftStep,
  type FlowEdge,
  type FlowNode,
} from "@/canvas/graph";
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
    // `numberedId` rather than a third copy of the same loop: `edits.ts`
    // already mints ids for duplicates and for nodes added from the menu.
    const id = numberedId(taken, draft.kind);
    taken.add(id);

    const node: PipelineNode = { id, type: draft.type, inputs: [parent], ...draft.payload };
    parent = id;
    return node;
  });
  return [...saved, ...added];
}


export function PipelineCanvas({
  onOpenNode,
  onCompare,
}: {
  onOpenNode: (id: string, label: string) => void;
  /** Opens the comparison tab once two terminal estimators are picked (#51). */
  onCompare?: (left: string, right: string) => void;
}) {
  const pipeline = usePipeline();
  const state = usePipelineState();
  const save = useSavePipeline();
  const [steps, setSteps] = useState<DraftStep[]>([]);
  const [validation, setValidation] = useState<string | null>(null);
  /** The edited graph, or null while it still matches what the server holds. */
  const [edited, setEdited] = useState<PipelineNode[] | null>(null);
  // React Flow asks whether a connection is valid many times during one drag
  // and never reports the drop it refused. The reason for the last refusal is
  // kept here so the end of the drag can say why nothing happened.
  const refusal = useRef<string | null>(null);
  /** The nodes picked for a comparison. Two opens the tab and clears it, so
   * the control is a toggle with a very short memory rather than a mode. */
  const [picked, setPicked] = useState<string[]>([]);
  /** Where the user has dragged nodes since the last fetch, laid over the
   * layout the server sent. Local because a position is not part of the
   * recipe: it is written by its own request and must not wait on Save. */
  const [moved, setMoved] = useState<Record<string, { x: number; y: number }>>({});
  const saveLayout = useSaveLayout();
  // A drag ends with a click on the same node, and a click opens its tab. The
  // drag is recorded here so the click that follows it can be ignored - one
  // gesture should not both move a node and open it.
  const dragged = useRef(false);
  /** Where a connector was dropped on empty canvas, and which node it came
   * from. Non-null while the menu of steps to add is open. */
  const [dropped, setDropped] = useState<
    { parent: string; at: { x: number; y: number }; screen: { x: number; y: number } } | null
  >(null);
  // Captured from onInit rather than useReactFlow(), which would need this
  // component wrapped in a provider it does not currently have.
  const flow = useRef<ReactFlowInstance<FlowNode, FlowEdge> | null>(null);

  // Memoised because it feeds the graph's useMemo: a fresh [] every render
  // would rebuild the whole graph on every keystroke elsewhere in the tab.
  const nodes = useMemo(
    () => edited ?? pipeline.data?.nodes ?? [],
    [edited, pipeline.data],
  );

  /** Pick a node for comparison; the second pick opens the tab.
   *
   * Two clicks rather than a multi-select because a click on a node body
   * already means "open this", so a selection would need a modifier nobody is
   * told about. The armed node saying so is the affordance instead.
   *
   * Memoised for the same reason `nodes` is: it feeds the graph's useMemo, and
   * a fresh function each render would rebuild the whole graph.
   */
  const pick = useCallback(
    (id: string) => {
      setPicked((current) => {
        if (current.includes(id)) return current.filter((other) => other !== id);
        const next = [...current, id];
        if (next.length < 2) return next;
        onCompare?.(next[0], next[1]);
        return [];
      });
    },
    [onCompare],
  );

  const graph = useMemo(() => {
    if (!pipeline.data) return { nodes: [], edges: [] };
    const style = getComputedStyle(document.documentElement);
    const token = (name: string) => style.getPropertyValue(`--${name}`).trim() || "currentColor";
    const committed = {
      nodes: toNodes(
        { ...pipeline.data, nodes },
        state.data,
        nodeLabel,
        (id) => edit(() => remove(nodes, id)),
        onCompare && {
          ids: picked,
          terminals: new Set(terminals(nodes).map((node) => node.id)),
          onCompare: pick,
        },
        (id) => edit(() => duplicate(nodes, id)),
      ),
      edges: toEdges(
        { ...pipeline.data, nodes },
        state.data,
        { rule: token("rule"), accent: token("accent"), stale: token("stale") },
        usesMotion(),
      ),
    };
    committed.nodes = committed.nodes.map((node) =>
      moved[node.id] ? { ...node, position: moved[node.id] } : node,
    );
    const lowest = Math.max(...committed.nodes.map((node) => node.position.y), 0);
    const draft = draftGraph(steps, { x: 40, y: lowest + 150 });
    return { nodes: [...committed.nodes, ...draft.nodes], edges: [...committed.edges, ...draft.edges] };
  }, [pipeline.data, state.data, steps, nodes, picked, onCompare, pick, moved]);

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
          nodesConnectable
          onNodesChange={(changes: NodeChange[]) => {
            // Only positions. React Flow also reports selection, dimensions
            // and removal here, and this canvas derives all three from the
            // pipeline rather than from React Flow's own node state.
            const positions = changes.filter(
              (change): change is Extract<NodeChange, { type: "position" }> =>
                change.type === "position" && change.position !== undefined,
            );
            if (positions.length === 0) return;
            setMoved((current) => {
              const next = { ...current };
              for (const change of positions) next[change.id] = change.position!;
              return next;
            });
          }}
          onNodeDragStop={(_event, node) => {
            dragged.current = true;
            // A draft has no node on the server to hang a position on; it gets
            // one when Save turns it into a real node.
            if (String(node.id).startsWith("draft-")) return;
            saveLayout.mutate({ ...moved, [node.id]: node.position });
          }}
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
          onInit={(instance) => {
            flow.current = instance;
          }}
          onConnectEnd={(event, state) => {
            if (refusal.current) setValidation(refusal.current);
            refusal.current = null;
            // A connector dropped on empty canvas is an offer to add a step
            // there. The parent is the node it was dragged from and the
            // position is where it was let go, so neither has to be guessed -
            // which is what `withDrafts` could not do and says so.
            if (state.isValid !== null) return;
            const parent = state.fromNode?.id;
            if (!parent || !flow.current || String(parent).startsWith("draft-")) return;
            const point =
              "clientX" in event
                ? { x: event.clientX, y: event.clientY }
                : { x: event.changedTouches[0].clientX, y: event.changedTouches[0].clientY };
            setDropped({
              parent,
              at: flow.current.screenToFlowPosition(point),
              screen: point,
            });
          }}
          onNodeClick={(_, node: Node) => {
            // A drag ends with a click on the node it moved. Opening its tab
            // there would mean no node could be moved without also being
            // opened, so the drag consumes the click that follows it.
            if (dragged.current) {
              dragged.current = false;
              return;
            }
            // Selecting a node focuses its tab - the mechanism that ties the
            // graph to the pages.
            if (String(node.id).startsWith("draft-")) return;
            onOpenNode(node.id, String((node.data as { label: string }).label));
          }}
        >
          {/* The artboard's ground: a 22px dot grid in --grid. */}
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--grid)" />
        </ReactFlow>
      </div>

      {dropped ? (
        <div
          role="menu"
          aria-label="Add a step"
          data-testid="add-step-menu"
          style={{
            position: "fixed",
            left: dropped.screen.x,
            top: dropped.screen.y,
            zIndex: 10,
            background: "var(--surface)",
            border: "1px solid var(--rule)",
            borderRadius: 3,
            padding: 4,
            boxShadow: "0 6px 18px rgb(0 0 0 / 0.16)",
            minWidth: 150,
          }}
        >
          <div className="ilabel" style={{ padding: "2px 8px 4px" }}>
            Add after {dropped.parent}
          </div>
          {STEP_MENU.map((step) => (
            <button
              key={step.kind}
              role="menuitem"
              className="srow"
              style={{ display: "block", width: "100%", textAlign: "left", padding: "3px 8px" }}
              onClick={() => {
                // Minted once and used twice: `add` derives the same id from
                // the same set, and the position has to be filed under it.
                const id = numberedId(new Set(nodes.map((node) => node.id)), step.kind);
                edit(() =>
                  add(nodes, dropped.parent, { type: step.type, ...step.payload }, step.kind),
                );
                // Placed where the connector was let go. #162's layout write
                // carries it to the server when the pipeline is saved.
                setMoved((current) => ({ ...current, [id]: dropped.at }));
                setDropped(null);
              }}
            >
              {step.kind}
            </button>
          ))}
        </div>
      ) : null}

      <StepList
        steps={steps}
        saving={save.isPending}
        edited={edited !== null}
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
