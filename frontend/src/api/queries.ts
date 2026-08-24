import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
  source: { filename: string; reader: string; reader_version: string; file_hash: string } | null;
  derived_from: string | null;
  created_at: string;
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
  nodes: Record<string, { state: string; message?: string }>;
}

export interface Experiment {
  experiment_id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  metrics: { explained_variance: number[] | null };
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

export function useDatasets(projectId: string | undefined) {
  return useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api<DatasetEntry[]>(`/projects/${projectId}/datasets`),
    enabled: Boolean(projectId),
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
