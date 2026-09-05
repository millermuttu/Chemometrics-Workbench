/** The comparison table's arithmetic, without a browser.
 *
 * The rule from #48 holds here too: nothing is computed in the frontend. The
 * difference column is a subtraction of two numbers the server sent, which is
 * arithmetic on displayed values rather than a statistic - and this is where
 * that distinction is pinned.
 */
import { describe, expect, it } from "vitest";

import { compareRows } from "@/screens/analysis/CompareResults";

describe("comparing two results", () => {
  it("subtracts the left from the right, and says which is better", () => {
    const rows = compareRows({ rmsecv: 2.4 }, { rmsecv: 2.1 });
    expect(rows).toHaveLength(1);
    expect(rows[0].delta).toBeCloseTo(-0.3, 10);
    // Lower is better for an error, so the right-hand model wins.
    expect(rows[0].better).toBe(true);
  });

  it("reverses that for a metric where higher is better", () => {
    expect(compareRows({ q2: 0.9 }, { q2: 0.95 })[0].better).toBe(true);
    expect(compareRows({ q2: 0.95 }, { q2: 0.9 })[0].better).toBe(false);
    expect(compareRows({ rmsecv: 1.0 }, { rmsecv: 2.0 })[0].better).toBe(false);
  });

  it("calls a tie neither, rather than picking one", () => {
    expect(
      compareRows({ rmsec: 1.5 }, { rmsec: 1.5 })[0].better,
    ).toBeUndefined();
  });

  it("keeps a metric only one of them carries, with no difference", () => {
    const rows = compareRows({ rmsecv: 2.4 }, {});
    expect(rows).toHaveLength(1);
    expect(rows[0].a).toBe(2.4);
    expect(rows[0].b).toBeUndefined();
    expect(rows[0].delta).toBeUndefined();
    expect(rows[0].better).toBeUndefined();
  });

  it("drops a metric neither carries rather than showing an empty row", () => {
    expect(
      compareRows({ rmsec: 1.0 }, { rmsec: 1.1 }).map((row) => row.metric),
    ).toEqual(["rmsec"]);
  });

  it("has nothing to say about two decompositions", () => {
    expect(compareRows(undefined, undefined)).toEqual([]);
    expect(compareRows({}, {})).toEqual([]);
  });

  it("lists the metrics in the order section 11 does", () => {
    const both = {
      rmsec: 1,
      rmsecv: 1,
      rmsep: 1,
      r2: 1,
      q2: 1,
      bias: 1,
      sec: 1,
      sep: 1,
    };
    expect(compareRows(both, both).map((row) => row.metric)).toEqual([
      "rmsec",
      "rmsecv",
      "rmsep",
      "r2",
      "q2",
      "bias",
      "sec",
      "sep",
    ]);
  });
});
