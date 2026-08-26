import { Background, BackgroundVariant, ReactFlow, type Node } from "@xyflow/react";
import { useMemo, useState } from "react";

import { api } from "@/api/client";
import { usePipeline, usePipelineState } from "@/api/queries";
import { NodeCard } from "@/canvas/NodeCard";
import { StepList } from "@/canvas/StepList";
import { draftGraph, toEdges, toNodes, type DraftStep } from "@/canvas/graph";
import { nodeLabel } from "@/shell/Sidebar";

import "@xyflow/react/dist/style.css";

/** The signature screen: the pipeline as a graph.
 *
 * Read-only in 1.1 - it renders and it selects, and the pipeline is built
 * through the step list beside it. Dragging nodes around is #51.
 */

const NODE_TYPES = { workbench: NodeCard };

/** Motion is an accent on a running path, never decoration, and never at all
 * for someone who asked the system not to move things. */
function usesMotion(): boolean {
  return !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

export function PipelineCanvas({ onOpenNode }: { onOpenNode: (id: string, label: string) => void }) {
  const pipeline = usePipeline();
  const state = usePipelineState();
  const [steps, setSteps] = useState<DraftStep[]>([]);
  const [validation, setValidation] = useState<string | null>(null);

  const { nodes, edges } = useMemo(() => {
    if (!pipeline.data) return { nodes: [], edges: [] };
    const style = getComputedStyle(document.documentElement);
    const token = (name: string) => style.getPropertyValue(`--${name}`).trim() || "currentColor";
    const committed = {
      nodes: toNodes(pipeline.data, state.data, nodeLabel),
      edges: toEdges(
        pipeline.data,
        state.data,
        { rule: token("rule"), accent: token("accent"), stale: token("stale") },
        usesMotion(),
      ),
    };
    const lowest = Math.max(...committed.nodes.map((node) => node.position.y), 0);
    const draft = draftGraph(steps, { x: 40, y: lowest + 150 });
    return { nodes: [...committed.nodes, ...draft.nodes], edges: [...committed.edges, ...draft.edges] };
  }, [pipeline.data, state.data, steps]);

  return (
    <div className="pane" style={{ flexDirection: "row" }}>
      <div style={{ flex: 1, minWidth: 0, background: "var(--surface)" }} data-testid="pipeline-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          fitView
          nodesDraggable={false}
          nodesConnectable={false}
          proOptions={{ hideAttribution: true }}
          onNodeClick={(_, node: Node) => {
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

      <StepList
        steps={steps}
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
