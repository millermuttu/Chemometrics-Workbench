import {
  useDatasets,
  useExperiments,
  useModels,
  usePipeline,
  usePipelineState,
  type ExperimentRow,
  type ModelRow,
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
    case "train_test":
      return `Train/test ${Math.round(Number(spec?.test_size) * 100)}% · seed ${spec?.seed}`;
    case "pca":
      return `PCA ${spec?.n_components} PC`;
    case "pls":
      // Latent variables rather than PCs, and the response, because two PLS
      // nodes on one branch differ by what they model rather than by their
      // component count.
      return `PLS ${spec?.n_components} LV · ${spec?.target}`;
    case "plsda":
      return `PLS-DA ${spec?.n_components} LV · ${spec?.class_column}`;
    case "source":
      return "Source";
    default:
      return node.id;
  }
}

/** "Run 12", counted from the project's first rather than from the top of the
 * list, so a run keeps its number as later ones arrive. */
export function runLabel(run: ExperimentRow, index: number, total: number): string {
  void run;
  return `Run ${total - index}`;
}

/** The one figure a row carries: whichever headline the run has. A run that
 * failed says so instead, because a failed experiment is a result (section 8.2)
 * and its status is the thing to read. */
export function runFigure(run: ExperimentRow): string {
  if (run.status !== "succeeded") return run.status;
  const m = run.metrics;
  if (m?.rmsecv != null) return `RMSECV ${m.rmsecv.toFixed(4)}`;
  if (m?.accuracy != null) return `accuracy ${m.accuracy.toFixed(3)}`;
  if (m?.explained_variance != null) return `PC1 ${(m.explained_variance * 100).toFixed(1)}%`;
  return run.status;
}

/** The one figure a saved model carries: whichever headline its metrics hold.
 * A decomposition has no error to quote and reads as what it is, because a
 * metric it does not have is absent rather than zero (section 11). */
export function modelFigure(model: ModelRow): string {
  const m = model.metrics;
  if (m.rmsecv != null) return `RMSECV ${m.rmsecv.toFixed(4)}`;
  if (m.rmsec != null) return `RMSEC ${m.rmsec.toFixed(4)}`;
  if (m.accuracy != null) return `accuracy ${m.accuracy.toFixed(3)}`;
  if (m.explained_variance != null) return `PC1 ${(m.explained_variance * 100).toFixed(1)}%`;
  return model.task;
}

export function Sidebar({ projectId, activeId, collapsed, onOpen }: Props) {
  const datasets = useDatasets(projectId);
  const pipeline = usePipeline();
  const pipelineState = usePipelineState();
  const experiments = useExperiments();
  const models = useModels();

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

        <Head
          label="Experiments"
          note={experiments.data ? String(experiments.data.length) : undefined}
        />
        {/* Every run this project has recorded, newest first (#209). The
            table has kept them since #121; until then the outline drew one
            row labelled "Runs" however many there were. */}
        {experiments.data?.length === 0 ? (
          <div className="empty">{collapsed ? <FlaskIcon /> : "Nothing run yet."}</div>
        ) : null}
        {experiments.data?.map((run, index) => (
          <Row
            key={run.experiment_id}
            icon={<FlaskIcon />}
            label={runLabel(run, index, experiments.data!.length)}
            dim={runFigure(run)}
            depth={26}
            selected={activeId === run.experiment_id}
            onOpen={(transient) =>
              onOpen(
                {
                  id: run.experiment_id,
                  kind: "experiment",
                  title: runLabel(run, index, experiments.data!.length),
                },
                transient,
              )
            }
          />
        ))}

        <Head label="Models" note={models.data ? String(models.data.length) : undefined} />
        {/* What this project holds, newest first (#219). Saving a fitted
            estimator from the analysis tab is what puts one here; a run on its
            own does not, because which fitted node is worth keeping is the
            user's judgement rather than the executor's. */}
        {models.data?.length === 0 ? (
          <div className="empty">
            {collapsed ? <ModelIcon /> : "No models yet — save one from a fitted estimator."}
          </div>
        ) : null}
        {models.data?.map((model) => (
          <Row
            key={model.model_id}
            icon={<ModelIcon />}
            label={model.name}
            dim={modelFigure(model)}
            depth={26}
            selected={activeId === model.model_id}
            onOpen={(transient) =>
              onOpen({ id: model.model_id, kind: "model", title: model.name }, transient)
            }
          />
        ))}
      </div>
    </aside>
  );
}
