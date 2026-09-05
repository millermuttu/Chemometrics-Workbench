/** The tab model. Pure, so the rules can be tested without a browser.
 *
 * The one rule worth stating: a single click previews into *the* transient
 * tab - there is only ever one, and opening another preview replaces it. A
 * double click, or an edit, pins it. This is what stops a session of clicking
 * through fifteen preprocessing variants from leaving fifteen tabs behind.
 */

export type TabKind =
  | "dataset"
  | "import"
  | "pipeline"
  | "spectra"
  | "results"
  | "compare"
  | "experiment"
  | "model";

export interface Tab {
  /** Stable across opens: the node, dataset or experiment id it shows. */
  id: string;
  kind: TabKind;
  title: string;
  transient: boolean;
}

export interface TabState {
  tabs: Tab[];
  activeId: string | null;
  /** The document shown beside the active one, or null when not split. */
  splitId: string | null;
}

export const emptyTabs: TabState = { tabs: [], activeId: null, splitId: null };

export type TabAction =
  | { type: "open"; tab: Omit<Tab, "transient">; transient: boolean }
  | { type: "pin"; id: string }
  | { type: "close"; id: string }
  | { type: "activate"; id: string }
  | { type: "move"; from: number; to: number }
  | { type: "split"; id: string | null };

export function tabsReducer(state: TabState, action: TabAction): TabState {
  switch (action.type) {
    case "open": {
      const existing = state.tabs.find((tab) => tab.id === action.tab.id);
      if (existing) {
        // Opening what is already open focuses it. A pinned tab stays pinned;
        // a preview opened again deliberately becomes permanent.
        const tabs = action.transient
          ? state.tabs
          : state.tabs.map((tab) =>
              tab.id === action.tab.id ? { ...tab, transient: false } : tab,
            );
        return { ...state, tabs, activeId: action.tab.id };
      }
      const opened: Tab = { ...action.tab, transient: action.transient };
      const transientIndex = state.tabs.findIndex((tab) => tab.transient);
      const tabs = [...state.tabs];
      if (action.transient && transientIndex >= 0) tabs.splice(transientIndex, 1, opened);
      else tabs.push(opened);
      return { ...state, tabs, activeId: opened.id };
    }
    case "pin":
      return {
        ...state,
        tabs: state.tabs.map((tab) => (tab.id === action.id ? { ...tab, transient: false } : tab)),
        activeId: action.id,
      };
    case "close": {
      const index = state.tabs.findIndex((tab) => tab.id === action.id);
      if (index < 0) return state;
      const tabs = state.tabs.filter((tab) => tab.id !== action.id);
      // Closing the active tab falls to its neighbour, the way an editor does.
      const activeId =
        state.activeId === action.id
          ? (tabs[index]?.id ?? tabs[index - 1]?.id ?? null)
          : state.activeId;
      const splitId = state.splitId === action.id ? null : state.splitId;
      return { tabs, activeId, splitId };
    }
    case "activate":
      return { ...state, activeId: action.id };
    case "move": {
      const tabs = [...state.tabs];
      const [moved] = tabs.splice(action.from, 1);
      if (!moved) return state;
      tabs.splice(action.to, 0, moved);
      return { ...state, tabs };
    }
    case "split":
      return { ...state, splitId: action.id };
  }
}
