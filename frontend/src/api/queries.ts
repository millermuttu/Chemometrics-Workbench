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

export interface Job {
  job_id: string;
  experiment_id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
}

export function useProjects() {
  return useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
}

/** `?empty` in the application's own URL asks the stub server for a project
 * with no datasets, which is how the empty-project state (#44) is reached
 * without editing code. In 1.2 a new project is simply empty. */
export function useDatasets(projectId: string | undefined) {
  const empty = new URLSearchParams(window.location.search).has("empty");
  return useQuery({
    queryKey: ["datasets", projectId, empty],
    queryFn: () => api<DatasetEntry[]>(`/projects/${projectId}/datasets${empty ? "?empty=true" : ""}`),
    enabled: Boolean(projectId),
  });
}

/** Nothing is committed by a preview: it reports what the reader found and
 * the user confirms or corrects it. The corrections ride along on the import
 * so 1.2's reader has them; the stub server ignores them. */
export function useImportPreview() {
  return useMutation({
    mutationFn: (options: { fail?: boolean } = {}) =>
      api<ImportPreview>("/import/preview", {
        method: "POST",
        headers: options.fail ? { "X-Stub-Fail": "1" } : {},
      }),
  });
}

export function useImportDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (corrections: Record<string, string>) =>
      api<DatasetEntry>("/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrections }),
      }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["datasets"] }),
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

export function useRunExperiment() {
  return useMutation({
    mutationFn: (options: { fail?: boolean } = {}) =>
      api<Job>(`/experiments/current/run${options.fail ? "?fail=true" : ""}`, { method: "POST" }),
  });
}

export function useCancelJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<Job>(`/jobs/${jobId}/cancel`, { method: "POST" }),
    onSuccess: (job) => client.setQueryData(["job", job.job_id], job),
  });
}
