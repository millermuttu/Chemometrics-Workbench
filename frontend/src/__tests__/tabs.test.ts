/** The tab rules. The point of the transient tab is that clicking through
 * fifteen preprocessing variants leaves one tab behind, not fifteen. */
import { describe, expect, it } from "vitest";

import { emptyTabs, tabsReducer, type TabState } from "@/shell/tabs";

const preview = (id: string): Parameters<typeof tabsReducer>[1] => ({
  type: "open",
  tab: { id, kind: "spectra", title: id.toUpperCase() },
  transient: true,
});

const pinned = (id: string): Parameters<typeof tabsReducer>[1] => ({
  type: "open",
  tab: { id, kind: "spectra", title: id.toUpperCase() },
  transient: false,
});

function run(actions: Parameters<typeof tabsReducer>[1][], from: TabState = emptyTabs): TabState {
  return actions.reduce(tabsReducer, from);
}

describe("previewing", () => {
  it("replaces the transient tab instead of adding one", () => {
    const state = run([preview("snv"), preview("msc"), preview("savgol")]);
    expect(state.tabs.map((tab) => tab.id)).toEqual(["savgol"]);
    expect(state.activeId).toBe("savgol");
  });

  it("keeps pinned tabs and previews beside them, in place", () => {
    const state = run([pinned("snv"), preview("msc"), preview("savgol")]);
    expect(state.tabs.map((tab) => tab.id)).toEqual(["snv", "savgol"]);
  });

  it("pins on a double click, and the next preview no longer replaces it", () => {
    const state = run([preview("snv"), { type: "pin", id: "snv" }, preview("msc")]);
    expect(state.tabs.map((tab) => tab.id)).toEqual(["snv", "msc"]);
    expect(state.tabs[0].transient).toBe(false);
  });

  it("opening an open tab focuses it rather than duplicating it", () => {
    const state = run([pinned("snv"), pinned("msc"), preview("snv")]);
    expect(state.tabs.map((tab) => tab.id)).toEqual(["snv", "msc"]);
    expect(state.activeId).toBe("snv");
  });

  it("opening a preview deliberately a second time makes it permanent", () => {
    const state = run([preview("snv"), pinned("snv"), preview("msc")]);
    expect(state.tabs.map((tab) => tab.id)).toEqual(["snv", "msc"]);
  });
});

describe("closing", () => {
  it("falls to the neighbour on the right, then the left", () => {
    const three = run([pinned("a"), pinned("b"), pinned("c")]);
    expect(tabsReducer(three, { type: "close", id: "c" }).activeId).toBe("b");
    const middle = tabsReducer({ ...three, activeId: "b" }, { type: "close", id: "b" });
    expect(middle.activeId).toBe("c");
  });

  it("leaves nothing active when the last tab closes", () => {
    const state = run([pinned("a"), { type: "close", id: "a" }]);
    expect(state).toEqual({ tabs: [], activeId: null, splitId: null });
  });

  it("closes the split when the tab it showed goes", () => {
    const state = run([pinned("a"), pinned("b"), { type: "split", id: "a" }, { type: "close", id: "a" }]);
    expect(state.splitId).toBeNull();
  });

  it("does not disturb the active tab when another closes", () => {
    const state = run([pinned("a"), pinned("b"), { type: "activate", id: "a" }, { type: "close", id: "b" }]);
    expect(state.activeId).toBe("a");
  });
});

it("reorders by moving one tab to another index", () => {
  const state = run([pinned("a"), pinned("b"), pinned("c"), { type: "move", from: 2, to: 0 }]);
  expect(state.tabs.map((tab) => tab.id)).toEqual(["c", "a", "b"]);
});
