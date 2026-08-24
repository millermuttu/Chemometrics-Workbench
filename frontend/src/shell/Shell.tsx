import { useCallback, useReducer, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";

import type { DatasetEntry, PipelineState } from "@/api/queries";
import {
  useCancelJob,
  useDatasets,
  useExperiment,
  useJob,
  usePipeline,
  usePipelineState,
  useProjects,
  useRunExperiment,
} from "@/api/queries";
import { DatasetView } from "@/screens/DatasetView";
import { EmptyProject } from "@/screens/EmptyProject";
import { Import } from "@/screens/Import";
import { PipelineCanvas } from "@/canvas/PipelineCanvas";
import { SpectraView } from "@/screens/SpectraView";
import { Inspector } from "@/shell/Inspector";
import { Sidebar } from "@/shell/Sidebar";
import { StatusBar } from "@/shell/StatusBar";
import { TabStrip } from "@/shell/TabStrip";
import { FlaskIcon, KIND_ICONS } from "@/shell/icons";
import { downstreamOf } from "@/inspector/stale";
import { emptyTabs, tabsReducer, type Tab } from "@/shell/tabs";

/** The frame every screen opens inside. The measurements are the artboard's -
 * see src/styles/shell.css, which is ported from design/canvas/_base.css. */

type Theme = "t-light" | "t-dark";

/** A region the user can drag wider or collapse. The handle straddles the rule
 * so the pointer target is bigger than the line it moves. */
function useResizable(initial: number, min: number, max: number, side: "left" | "right") {
  const [width, setWidth] = useState(initial);
  const dragging = useRef(false);

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      dragging.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      const startX = event.clientX;
      const startWidth = width;
      const move = (moveEvent: PointerEvent) => {
        if (!dragging.current) return;
        const delta = side === "left" ? moveEvent.clientX - startX : startX - moveEvent.clientX;
        setWidth(Math.min(max, Math.max(min, startWidth + delta)));
      };
      const up = () => {
        dragging.current = false;
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    },
    [width, min, max, side],
  );

  return { width, onPointerDown };
}

function Pane({
  tab,
  datasets,
  onImported,
  onCloseImport,
  onOpenNode,
}: {
  tab: Tab | undefined;
  datasets: DatasetEntry[] | undefined;
  onOpenNode: (id: string, label: string) => void;
  onImported: (versionId: string, name: string) => void;
  onCloseImport: () => void;
}) {
  if (tab?.kind === "import") {
    return <Import onImported={onImported} onCancel={onCloseImport} />;
  }

  if (tab?.kind === "pipeline") return <PipelineCanvas onOpenNode={onOpenNode} />;
  if (tab?.kind === "spectra") return <SpectraView nodeId={tab.id} title={tab.title} />;

  const found = datasets
    ?.flatMap((entry) => entry.versions.map((version) => ({ entry, version })))
    .find((pair) => pair.version.version_id === tab?.id);
  if (found) return <DatasetView entry={found.entry} version={found.version} />;

  if (!tab) {
    return (
      <div className="pane">
        <div className="empty" style={{ padding: 16 }}>
          Nothing open. Select something in the outline.
        </div>
      </div>
    );
  }
  const KindIcon = KIND_ICONS[tab.kind];
  return (
    <div className="pane">
      <div
        style={{
          height: 40,
          flex: "none",
          display: "flex",
          alignItems: "center",
          gap: 9,
          padding: "0 14px",
          borderBottom: "1px solid var(--rule2)",
        }}
      >
        <KindIcon />
        <span style={{ fontWeight: 600, fontSize: 13.5 }}>{tab.title}</span>
        <span className="pill mono">{tab.kind}</span>
      </div>
      {/* The documents themselves are #44 to #48. The shell's job is to open,
          close, reorder, split and focus them. */}
      <div className="empty" style={{ padding: 16 }}>
        {tab.kind} view — built in a later issue.
      </div>
    </div>
  );
}

export function Shell() {
  const [theme, setTheme] = useState<Theme>("t-light");
  const [state, dispatch] = useReducer(tabsReducer, emptyTabs);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const sidebar = useResizable(248, 180, 420, "left");
  const inspector = useResizable(292, 220, 460, "right");

  const queryClient = useQueryClient();
  const projects = useProjects();
  const project = projects.data?.[0];
  const datasets = useDatasets(project?.project_id);
  const pipeline = usePipeline();
  const pipelineState = usePipelineState();
  const experiment = useExperiment();

  const [staleFrom, setStaleFrom] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const job = useJob(jobId);
  const run = useRunExperiment();
  const cancel = useCancelJob();

  const open = useCallback(
    (tab: Omit<Tab, "transient">, transient: boolean) =>
      dispatch({ type: "open", tab, transient }),
    [],
  );

  const openImport = useCallback(
    () => open({ id: "import", kind: "import", title: "Import" }, false),
    [open],
  );
  const imported = useCallback(
    (versionId: string, name: string) => {
      dispatch({ type: "close", id: "import" });
      dispatch({ type: "open", tab: { id: versionId, kind: "dataset", title: name }, transient: false });
    },
    [],
  );

  const openNode = useCallback(
    (id: string, label: string) => {
      const node = pipeline.data?.nodes.find((candidate) => candidate.id === id);
      open(
        { id, kind: node?.type === "estimator" ? "results" : "spectra", title: label },
        false,
      );
    },
    [open, pipeline.data],
  );

  /** Editing a parameter invalidates everything computed from it. The results
   * stay on screen, dimmed - a stale result must not vanish. */
  const markStale = useCallback(
    (nodeId: string) => {
      if (!pipeline.data) return;
      const affected = [nodeId, ...downstreamOf(pipeline.data, nodeId)];
      queryClient.setQueryData<PipelineState>(["pipeline-state"], (current) =>
        current
          ? {
              ...current,
              nodes: {
                ...current.nodes,
                ...Object.fromEntries(
                  affected.map((id) => [
                    id,
                    {
                      ...current.nodes[id],
                      state: "stale",
                      reason: id === nodeId ? "edited - downstream stale" : "upstream changed",
                    },
                  ]),
                ),
              },
            }
          : current,
      );
      setStaleFrom(nodeId);
    },
    [pipeline.data, queryClient],
  );

  const activeTab = state.tabs.find((tab) => tab.id === state.activeId);
  const splitTab = state.tabs.find((tab) => tab.id === state.splitId);
  const samples = datasets.data?.[0]?.versions.at(-1);
  const noDatasets = datasets.isSuccess && datasets.data.length === 0;

  /** An estimator node's headline numbers, in .kv form with tabular numerals.
   * The full results table is #48; this is what fits in 292px. */
  const metricsFor = (tab: Tab | undefined) => {
    const node = pipeline.data?.nodes.find((candidate) => candidate.id === tab?.id);
    if (node?.type !== "estimator" || !experiment.data) return undefined;
    const variance = experiment.data.metrics.explained_variance ?? [];
    return {
      "PC1 variance": variance[0] ?? null,
      "PC1-5 cumulative": variance.slice(0, 5).reduce((total, item) => total + item, 0) || null,
      components: (node.spec?.n_components as number) ?? null,
    };
  };

  return (
    <div className={`app ${theme}`} style={{ position: "relative" }}>
      <div className="tbar">
        <div className="tb-l">
          <FlaskIcon />
          <span className="proj">{project?.name ?? "Loading…"}</span>
          <span className="crumb mono">{project?.directory ?? ""}</span>
        </div>
        <div className="tb-r">
          {samples ? (
            <span className="pill mono">
              {samples.n_samples} × {samples.n_variables}
            </span>
          ) : null}
          <button
            className="btn"
            onClick={() => open({ id: "pipeline", kind: "pipeline", title: "Pipeline" }, false)}
          >
            Pipeline
          </button>
          <button className="btn" onClick={openImport}>
            Import…
          </button>
          <button className="btn" onClick={() => setTheme(theme === "t-light" ? "t-dark" : "t-light")}>
            {theme === "t-light" ? "Dark" : "Light"}
          </button>
          <button className="btn" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
            Outline
          </button>
          <button className="btn" onClick={() => setInspectorCollapsed(!inspectorCollapsed)}>
            Inspector
          </button>
          <button
            className="btn btn-p"
            onClick={async () => {
              const started = await run.mutateAsync({});
              setJobId(started.job_id);
              setStartedAt(Date.now());
            }}
          >
            Run pipeline
          </button>
        </div>
      </div>

      <div className="body">
        <div style={{ display: "flex", width: sidebarCollapsed ? undefined : sidebar.width }}>
          <div style={{ flex: 1, display: "flex", minWidth: 0 }}>
            <Sidebar
              projectId={project?.project_id}
              activeId={state.activeId}
              collapsed={sidebarCollapsed}
              onOpen={open}
            />
          </div>
        </div>
        {sidebarCollapsed ? null : (
          <div className="grip" onPointerDown={sidebar.onPointerDown} role="separator" aria-label="Resize outline" />
        )}

        <main className="doc">
          <TabStrip
            tabs={state.tabs}
            activeId={state.activeId}
            splitId={state.splitId}
            onActivate={(id) => dispatch({ type: "activate", id })}
            onPin={(id) => dispatch({ type: "pin", id })}
            onClose={(id) => dispatch({ type: "close", id })}
            onMove={(from, to) => dispatch({ type: "move", from, to })}
            onSplit={(id) => dispatch({ type: "split", id })}
          />
          {noDatasets && !activeTab ? (
            <EmptyProject onImport={openImport} />
          ) : state.splitId ? (
            <div className="split">
              <Pane tab={activeTab} datasets={datasets.data} onImported={imported} onCloseImport={() => dispatch({ type: "close", id: "import" })} onOpenNode={openNode} />
              <Pane tab={splitTab} datasets={datasets.data} onImported={imported} onCloseImport={() => dispatch({ type: "close", id: "import" })} onOpenNode={openNode} />
            </div>
          ) : (
            <Pane tab={activeTab} datasets={datasets.data} onImported={imported} onCloseImport={() => dispatch({ type: "close", id: "import" })} onOpenNode={openNode} />
          )}
        </main>

        {inspectorCollapsed ? null : (
          <div className="grip" onPointerDown={inspector.onPointerDown} role="separator" aria-label="Resize inspector" />
        )}
        <div style={{ display: "flex", width: inspectorCollapsed ? undefined : inspector.width }}>
          <div style={{ flex: 1, display: "flex", minWidth: 0 }}>
            <Inspector
              tab={activeTab}
              project={project}
              datasets={datasets.data}
              pipeline={pipeline.data}
              state={pipelineState.data}
              metrics={metricsFor(activeTab)}
              collapsed={inspectorCollapsed}
              onEdit={markStale}
            />
          </div>
        </div>
      </div>

      {staleFrom ? (
        <div
          role="status"
          style={{
            position: "absolute",
            right: 304,
            bottom: 36,
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 10px",
            borderRadius: 3,
            border: "1px solid var(--stale)",
            background: "var(--staleSoft)",
            fontSize: 11.5,
          }}
        >
          <span style={{ color: "var(--stale)" }}>Downstream results are stale.</span>
          <button
            className="btn"
            style={{ height: 22 }}
            onClick={async () => {
              const started = await run.mutateAsync({});
              setJobId(started.job_id);
              setStartedAt(Date.now());
              setStaleFrom(null);
            }}
          >
            Re-run
          </button>
          <button className="tabx" aria-label="Dismiss" onClick={() => setStaleFrom(null)}>
            ×
          </button>
        </div>
      ) : null}

      <StatusBar
        job={job.data}
        startedAt={startedAt}
        onCancel={() => {
          if (jobId) cancel.mutate(jobId);
        }}
      />
      {/* experiment is read for the outline's Experiments section; keeping the
          query here means one fetch shared by both regions. */}
      <span hidden>{experiment.data?.status}</span>
    </div>
  );
}
