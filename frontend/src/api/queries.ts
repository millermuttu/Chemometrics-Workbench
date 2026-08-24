import { useQuery } from "@tanstack/react-query";

import { api } from "./client";

/** The payload shapes are the stub server's, which are generated from the
 * kernels - see stub/generate_fixtures.py. Only the fields used are typed. */
export interface Project {
  project_id: string;
  name: string;
  description: string;
  directory: string;
  created_at: string;
}

export function useProjects() {
  return useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/projects") });
}
