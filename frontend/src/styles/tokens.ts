/** The token names, grouped the way the issue and the brief group them.
 *
 * Names only: the values live in tokens.css, ported from _base.css, and
 * src/__tests__/tokens.test.ts holds all three in step.
 */
export const TOKEN_GROUPS = [
  { label: "Ground", names: ["ground", "panel", "surface", "sunken"] },
  { label: "Ink", names: ["ink", "ink2", "ink3"] },
  { label: "Rules", names: ["rule", "rule2"] },
  { label: "Accent", names: ["accent", "accentSoft", "accentInk"] },
  { label: "Run state", names: ["stale", "staleSoft", "fail", "failSoft"] },
  { label: "Data series", names: ["d1", "d2", "d3", "d4", "d5", "d6"] },
  { label: "Plot furniture", names: ["grid", "band"] },
] as const;

export const TOKEN_NAMES = TOKEN_GROUPS.flatMap((group) => group.names);

export type Theme = "t-light" | "t-dark";
