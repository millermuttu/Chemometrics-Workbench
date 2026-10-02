/** Which metadata columns a two-class PLS-DA can classify by (#185).
 *
 * `pls-da.md` section 2: exactly two distinct values. A column with one value
 * has nothing to separate and one with three is PLS2, which the executor
 * refuses by name - so neither is offered, rather than offered and refused.
 * Pure, so the rule is tested without a browser.
 */
export function twoValuedColumns(
  columns: Record<string, (string | number)[]> | undefined,
): string[] {
  return Object.entries(columns ?? {})
    .filter(([, values]) => new Set(values.map(String)).size === 2)
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
