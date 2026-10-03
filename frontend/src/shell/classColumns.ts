/** Which metadata columns a PLS-DA can classify by (#185, #274).
 *
 * `pls-da.md` section 2: two or more distinct values. A column with one value
 * has nothing to separate, and the executor refuses it by name - so it is not
 * offered, rather than offered and refused. Pure, so the rule is tested
 * without a browser.
 */
export function classColumns(
  columns: Record<string, (string | number)[]> | undefined,
): string[] {
  return Object.entries(columns ?? {})
    .filter(([, values]) => new Set(values.map(String)).size >= 2)
    .map(([name]) => name);
}

/** Which metadata columns a split can stratify by (#268).
 *
 * `metrics-and-validation.md` section 8.7: at least two levels, and every
 * level with at least two samples - one to fit on and one to hold out. A
 * column that breaks either is refused by the executor, so it is not offered.
 */
export function stratifiableColumns(
  columns: Record<string, (string | number)[]> | undefined,
): string[] {
  return Object.entries(columns ?? {})
    .filter(([, values]) => {
      const counts = new Map<string, number>();
      for (const value of values) counts.set(String(value), (counts.get(String(value)) ?? 0) + 1);
      return counts.size >= 2 && [...counts.values()].every((count) => count >= 2);
    })
    .map(([name]) => name);
}
