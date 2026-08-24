/** The gl2d bundle ships no types. Only what this project calls is declared:
 * a fuller @types/plotly.js would pull in the whole API surface for two
 * functions, and would then describe a bundle that is not the one loaded. */
declare module "plotly.js-gl2d-dist-min" {
  export interface PlotData {
    [key: string]: unknown;
  }
  export function react(
    root: HTMLElement,
    data: PlotData[],
    layout: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<void>;
  export function purge(root: HTMLElement): void;
  const Plotly: { react: typeof react; purge: typeof purge };
  export default Plotly;
}
