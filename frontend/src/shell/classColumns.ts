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
