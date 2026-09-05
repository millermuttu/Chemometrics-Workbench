import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import type { StepSchema } from "@/inspector/schema";

import { api } from "./client";

/** The payload shapes are the stub server's, generated from the kernels by
 * stub/generate_fixtures.py. Only the fields the shell reads are typed. */

export interface Project {
  project_id: string;
  name: string;
  description: string;
  directory: string;
  created_at: string;
}

export interface DatasetVersion {
  version_id: string;
  version: number;
  content_hash: string;
  n_samples: number;
  n_variables: number;
  axis: { kind: string; unit: string | null; values?: number[] };
  sample_ids: string[];
  targets: Record<string, number[]>;
  metadata_columns: Record<string, (string | number)[]>;
  excluded_samples: number[];
  excluded_variables: number[];
  source: {
    filename: string;
    reader: string;
    reader_version: string;
    file_hash: string;
    imported_at?: string;
  } | null;
  derived_from: string | null;
  array_path: string;
  created_at: string;
}

/** What a reader says it found, before anything is committed. Every value the
 * user can correct arrives with the alternatives the reader considered. */
export interface Detected<T> {
  value: T;
  alternatives: T[];
}

export interface ImportPreview {
  source: {
    filename: string;
    file_hash: string;
    reader: string;
    reader_version: string;
    size_bytes: number;
  };
  detected: {
    delimiter: Detected<string>;
    decimal: Detected<string>;
    orientation: Detected<string>;
    n_samples: number;
    n_variables: number;
    axis: {
      kind: string;
      unit: string | null;
      start: number;
      end: number;
      reconstructed: boolean;
      note?: string;
    };
    metadata_columns: string[];
    targets: string[];
    discarded: { what: string; why: string }[];
  };
  head: { sample_ids: string[]; rows: number[][] };
}

export interface DatasetEntry {
  dataset: { dataset_id: string; name: string; description: string };
  versions: DatasetVersion[];
}

export interface PipelineNode {
  id: string;
  type: string;
  inputs: string[];
  /** Preprocessing nodes carry `step`; estimators and splits carry `spec`. */
  step?: { kind: string; [key: string]: unknown };
  spec?: { kind: string; [key: string]: unknown };
  version_id?: string;
}

export interface Pipeline {
  pipeline_id: string;
  name: string;
  nodes: PipelineNode[];
}

export interface PipelineState {
  pipeline_id: string;
  nodes: Record<string, { state: string; progress?: number; reason?: string; message?: string }>;
  /** Presentation only, and deliberately outside Pipeline.content_hash():
   * moving a node must not change the science (design/data-model.md). */
  layout: Record<string, { x: number; y: number }>;
}

export interface Experiment {
  experiment_id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  metrics: { explained_variance: number[] | null };
}

/** What the spectra endpoint returns for one node. Decimation is consumed,
 * never computed here: the fixture carries the decimated and band forms and
 * 1.2 computes them server-side, where PROPOSAL.md section 13 puts them. */
export interface SpectraPayload {
  node_id: string;
  label: string;
  axis: { kind: string; unit: string | null; values: number[] };
  ordinate: { label: string };
  n_spectra: number;
  decimation: {
    variables_total: number;
    variables_kept: number;
    traces_total: number;
    traces_drawn: number;
    banded: boolean;
  };
  traces: { index: number; sample_id: string; y: number[] }[];
  band: { n_spectra: number; y_lower: number[]; y_median: number[]; y_upper: number[] };
}

/** One estimator node's results. Every number is the kernel's: scores,
 * loadings, variances, T², SPE and both limits arrive as data. */
export interface PcaPayload {
  node_id: string;
  task: string;
  n_components: number;
  n_samples: number;
  n_variables: number;
  rank: number;
  samples: { index: number; sample_id: string }[];
  scores: number[][];
  loadings: {
    axis: { kind: string; unit: string | null; values: number[] };
    components: number[][];
  };
  eigenvalues: number[];
  explained_variance_ratio: number[];
  cumulative_explained_variance: number[];
  diagnostics: {
    hotelling_t2: number[];
    hotelling_t2_limit: number;
    spe: number[];
    spe_limit: number;
    alpha: number;
  };
  /** The held-out rows of the fitted fold, present only below a split. Its
   * `observed` and `predicted` are there only for a regression. */
  validation?: {
    fold: number;
    samples: { index: number; sample_id: string }[];
    scores: number[][];
    hotelling_t2: number[];
    spe: number[];
    observed?: number[];
    predicted?: number[];
  };
  /** Present only when `task === "regression"`. The half of a PLS result that
   * has no counterpart on a decomposition; everything above is shared. */
  regression?: {
    target: string | null;
    observed: number[];
    predicted: number[];
    coefficients: number[];
    vip: number[];
    y_loadings: number[];
    y_explained_variance_ratio: number[];
  };
  /** `metrics-and-validation.md` section 11's table, flat. **A metric that
   * could not be computed is absent** - never zero, never NaN - so a reader
   * must render `undefined` as an em dash rather than a number. */
  metrics?: Partial<Record<string, number>>;
  /** RMSECV against component count, one point per `A`. Empty above a split. */
  rmsecv_curve?: number[];
}

export interface Job {
  job_id: string;
  experiment_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
  /** Which node the run is on, or the one a failure stopped at. Added by #85;
   * the five fields above are Phase 1.1's and did not change. */
  node_id: string | null;
}

export function useProjects() {
  return useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
}

/** A new project is empty because nothing has been imported into it, and a
 * dataset is past the envelope because of its own shape. Both states used to
 * need a query parameter; #89 removed the parameters and the states are
 * reached by being in them. */
export function useDatasets(projectId: string | undefined) {
  return useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api<DatasetEntry[]>(`/projects/${projectId}/datasets`),
    enabled: Boolean(projectId),
  });
}

/** Nothing is committed by a preview: it reports what the reader found and the
 * user confirms or corrects it. The corrections ride along on the import, so
 * the reader parses with the ones the user actually made.
 *
 * The file goes as multipart, because the request has to carry a file *and*
 * fields. Phase 1.1 sent neither — the screen's file input discarded what was
 * picked and posted an empty body, which the stub answered from a fixture
 * whatever it was sent (#99). The URL is the one it always was. */
export function useImportPreview() {
  return useMutation({
    mutationFn: (file: File) => {
      const body = new FormData();
      body.append("file", file);
      return api<ImportPreview>("/import/preview", { method: "POST", body });
    },
  });
}

export function useImportDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ file, corrections }: { file: File; corrections: Record<string, string> }) => {
      const body = new FormData();
      body.append("file", file);
      body.append("corrections", JSON.stringify(corrections));
      return api<DatasetEntry>("/import", { method: "POST", body });
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["datasets"] });
      void client.invalidateQueries({ queryKey: ["pipeline"] });
      void client.invalidateQueries({ queryKey: ["pipeline-state"] });
    },
  });
}

/** Phase 1.1 has one pipeline and one experiment, and the stub server returns
 * them whatever id it is given. 1.2 keeps the URLs and stops ignoring the id. */
export function usePipeline() {
  return useQuery({ queryKey: ["pipeline"], queryFn: () => api<Pipeline>("/pipelines/current") });
}

export function usePipelineState() {
  return useQuery({
    queryKey: ["pipeline-state"],
    queryFn: () => api<PipelineState>("/pipelines/current/state"),
    // Run state changes because something in this session changed it - an edit
    // marking downstream nodes stale, or a run advancing. Refetching on every
    // remount would throw those away, and 1.1 has nothing else writing it.
    staleTime: Infinity,
  });
}

/** Save the recipe. The whole node list, not a patch (#108).
 *
 * One project holds one pipeline and one user edits it, so last-write-wins
 * needs no conflict rules, and the canvas already holds the entire graph it is
 * drawing - sending a diff would mean inventing an operation language for a
 * problem nobody has yet.
 *
 * Both queries are invalidated because both change: the pipeline is the new
 * recipe, and the state is derived from it - a node whose key changed has no
 * arrays under the new key and comes back `not_run`. Nothing here writes
 * staleness; it is read back out of the store.
 */
/** Where the canvas put its nodes.
 *
 * Its own mutation against its own endpoint, mirroring the split the server
 * makes: a position lives outside `Pipeline.content_hash()`, so writing one
 * must not go through the body that carries the recipe. Nothing is
 * invalidated afterwards - the canvas already holds the position it just
 * sent, and the next `pipeline-state` fetch will echo the same value back.
 */
export function useSaveLayout() {
  return useMutation({
    mutationFn: (layout: Record<string, { x: number; y: number }>) =>
      api<{ layout: Record<string, { x: number; y: number }> }>("/pipelines/current/layout", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ layout }),
      }),
  });
}

export function useSavePipeline() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (nodes: PipelineNode[]) =>
      api<Pipeline>("/pipelines/current", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodes }),
      }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["pipeline"] });
      void client.invalidateQueries({ queryKey: ["pipeline-state"] });
      void client.invalidateQueries({ queryKey: ["experiment"] });
    },
  });
}

export function useExperiment() {
  return useQuery({
    queryKey: ["experiment"],
    queryFn: () => api<Experiment>("/experiments/current"),
  });
}

/** A run is a real job: it advances, it can be cancelled and it can fail. The
 * poll stops when the job reaches a terminal status. */
export function useSpectra(nodeId: string | undefined) {
  return useQuery({
    queryKey: ["spectra", nodeId],
    queryFn: () => api<SpectraPayload>(`/spectra/${nodeId}`),
    enabled: Boolean(nodeId),
    // The decimated payload is the largest thing crossing the wire; holding it
    // means switching between two nodes does not refetch either.
    staleTime: Infinity,
  });
}

export function useResults(nodeId: string | undefined) {
  return useQuery({
    queryKey: ["results", nodeId],
    queryFn: () => api<PcaPayload>(`/results/${nodeId}`),
    enabled: Boolean(nodeId),
    staleTime: Infinity,
  });
}

export function useStepSchema() {
  return useQuery({
    queryKey: ["step-schema"],
    queryFn: () => api<StepSchema>("/schema/steps"),
    staleTime: Infinity,
  });
}

export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api<Job>(`/jobs/${jobId}`),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running" ? 400 : false;
    },
  });
}

/** A run fails when it fails. #89 removed the parameter that used to make one. */
export function useRunExperiment() {
  return useMutation({
    mutationFn: () => api<Job>("/experiments/current/run", { method: "POST" }),
  });
}

export function useCancelJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<Job>(`/jobs/${jobId}/cancel`, { method: "POST" }),
    onSuccess: (job) => client.setQueryData(["job", job.job_id], job),
  });
}
