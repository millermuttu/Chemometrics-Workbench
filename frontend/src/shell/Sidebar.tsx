import {
  useDatasets,
  useExperiment,
  usePipeline,
  usePipelineState,
  type PipelineNode,
} from "@/api/queries";
import { DatasetIcon, FlaskIcon, ModelIcon, NodeIcon } from "@/shell/icons";
import type { Tab } from "@/shell/tabs";

/** The left outline: datasets, the pipeline as a node list, experiments and
 * models. Selecting a row opens it as a preview; a double click pins it. That
 * is the mechanism tying the outline to the pages, so it lives here and the
 * shell owns nothing but the handler. */

interface Props {
  projectId: string | undefined;
  activeId: string | null;
  collapsed: boolean;
  onOpen: (tab: Omit<Tab, "transient">, transient: boolean) => void;
}

function Row({
  icon,
  label,
  dim,
  depth,
  selected,
  onOpen,
}: {
  icon: React.ReactNode;
  label: string;
  dim?: string;
  depth: number;
  selected: boolean;
  onOpen: (transient: boolean) => void;
}) {
  return (
    <button
      className={`srow${selected ? " sel" : ""}`}
      style={{ paddingLeft: depth }}
      onClick={() => onOpen(true)}
      onDoubleClick={() => onOpen(false)}
      title={label}
    >
      {icon}
      <span>{label}</span>
      {dim ? <span className="sdim mono">{dim}</span> : null}
    </button>
  );
}

function Head({ label, note }: { label: string; note?: string }) {
  return (
    <div className="shead">
      <span>{label}</span>
      {note ? (
        <span className="mono" style={{ fontSize: 10 }}>
          {note}
        </span>
      ) : null}
    </div>
  );
}

/** "SNV", "SG d1 w11", "K-fold 10 · seed 42" - what the artboard's outline
 * shows. A node reads as what it does, not as its id or its node type. */
export function nodeLabel(node: PipelineNode): string {
  const step = node.step as Record<string, number | string> | undefined;
  const spec = node.spec as Record<string, number | string> | undefined;
  switch (step?.kind ?? spec?.kind ?? node.type) {
    case "snv":
      return "SNV";
    case "msc":
      return `MSC · ${step?.reference}`;
    case "mean_centre":
      return "Mean centre";
    case "autoscale":
      return "Autoscale";
    case "savgol":
      return `SG d${step?.deriv} w${step?.window_length}`;
    case "kfold":
      return `K-fold ${spec?.n_splits} · seed ${spec?.seed}`;
    case "pca":
      return `PCA ${spec?.n_components} PC`;
    case "pls":
      // Latent variables rather than PCs, and the response, because two PLS
      // nodes on one branch differ by what they model rather than by their
      // component count.
      return `PLS ${spec?.n_components} LV · ${spec?.target}`;
    case "source":
      return "Source";
    default:
      return node.id;
  }
}

export function Sidebar({ projectId, activeId, collapsed, onOpen }: Props) {
  const datasets = useDatasets(projectId);
  const pipeline = usePipeline();
  const pipelineState = usePipelineState();
  const experiment = useExperiment();

  return (
    <aside className={`side${collapsed ? " rail" : ""}`} aria-label="Project outline">
      <div style={{ overflowY: "auto" }}>
        <Head label="Datasets" note={datasets.data ? String(datasets.data.length) : undefined} />
        {datasets.data?.map((entry) =>
          entry.versions.map((version) => (
            <Row
              key={version.version_id}
              icon={<DatasetIcon />}
              label={entry.dataset.name}
              dim={`v${version.version} · ${version.n_samples}×${version.n_variables}`}
              depth={16}
              selected={activeId === version.version_id}
              onOpen={(transient) =>
                onOpen(
                  { id: version.version_id, kind: "dataset", title: entry.dataset.name },
                  transient,
                )
              }
            />
          )),
        )}

        <Head
          label="Pipeline"
          note={pipeline.data ? `${pipeline.data.nodes.length} nodes` : undefined}
        />
        {pipeline.data?.nodes.map((node) => {
          const state = pipelineState.data?.nodes[node.id]?.state;
          return (
            <Row
              key={node.id}
              icon={<NodeIcon />}
              label={nodeLabel(node)}
              dim={state && state !== "complete" ? state : node.id}
              depth={node.type === "source" ? 26 : 34}
              selected={activeId === node.id}
              onOpen={(transient) =>
                onOpen(
                  {
                    id: node.id,
                    kind: node.type === "estimator" ? "results" : "spectra",
                    title: nodeLabel(node),
                  },
                  transient,
                )
              }
            />
          );
        })}

        <Head label="Experiments" note={experiment.data ? "1" : undefined} />
        {experiment.data ? (
          <Row
            icon={<FlaskIcon />}
            label="Runs"
            dim={experiment.data.status}
            depth={26}
            selected={activeId === experiment.data.experiment_id}
            onOpen={(transient) =>
              onOpen(
                { id: experiment.data.experiment_id, kind: "experiment", title: "Runs" },
                transient,
              )
            }
          />
        ) : null}

        <Head label="Models" note="0" />
        {/* Models are Phase 2 (#51) and no fixture describes one, so the
            section is here with its empty state rather than invented. */}
        <div className="empty">
          {collapsed ? <ModelIcon /> : "No models yet — a run produces the first."}
        </div>
      </div>
    </aside>
  );
}
