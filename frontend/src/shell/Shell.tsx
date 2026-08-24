import { useCallback, useReducer, useRef, useState } from "react";

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
import { Inspector } from "@/shell/Inspector";
import { Sidebar } from "@/shell/Sidebar";
import { StatusBar } from "@/shell/StatusBar";
import { TabStrip } from "@/shell/TabStrip";
import { FlaskIcon, KIND_ICONS } from "@/shell/icons";
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

function Pane({ tab }: { tab: Tab | undefined }) {
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

  const projects = useProjects();
  const project = projects.data?.[0];
  const datasets = useDatasets(project?.project_id);
  const pipeline = usePipeline();
  const pipelineState = usePipelineState();
  const experiment = useExperiment();

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

  const activeTab = state.tabs.find((tab) => tab.id === state.activeId);
  const splitTab = state.tabs.find((tab) => tab.id === state.splitId);
  const samples = datasets.data?.[0]?.versions.at(-1);

  return (
    <div className={`app ${theme}`}>
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
          {state.splitId ? (
            <div className="split">
              <Pane tab={activeTab} />
              <Pane tab={splitTab} />
            </div>
          ) : (
            <Pane tab={activeTab} />
          )}
        </main>

        {inspectorCollapsed ? null : (
          <div className="grip" onPointerDown={inspector.onPointerDown} role="separator" aria-label="Resize inspector" />
        )}
        <div style={{ display: "flex", width: inspectorCollapsed ? undefined : inspector.width }}>
          <div style={{ flex: 1, display: "flex", minWidth: 0 }}>
            <Inspector
              tab={activeTab}
              datasets={datasets.data}
              nodes={pipeline.data?.nodes}
              state={pipelineState.data}
              collapsed={inspectorCollapsed}
            />
          </div>
        </div>
      </div>

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
