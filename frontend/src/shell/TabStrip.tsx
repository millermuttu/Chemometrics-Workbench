import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { KIND_ICONS, SplitIcon } from "@/shell/icons";
import type { Tab } from "@/shell/tabs";

/** The tab strip: reorderable, closable, with the overflow the artboard draws
 * as a mono "+3" at the end. Fifteen preprocessing variants is the normal
 * case, so the overflow is searchable rather than a plain list.
 *
 * Every tab stays in the DOM and the strip clips them. The first version
 * rendered only the tabs that fit, which cannot work: a tab that is not
 * rendered has no width, so the measurement that decides what fits could only
 * ever confirm what it already showed, and the strip stuck at one tab.
 *
 * Reordering is the platform's own drag and drop. A drag library would be
 * three dependencies for what four handlers do.
 */

interface Props {
  tabs: Tab[];
  activeId: string | null;
  splitId: string | null;
  onActivate: (id: string) => void;
  onPin: (id: string) => void;
  onClose: (id: string) => void;
  onMove: (from: number, to: number) => void;
  onSplit: (id: string | null) => void;
}

export function TabStrip({
  tabs,
  activeId,
  splitId,
  onActivate,
  onPin,
  onClose,
  onMove,
  onSplit,
}: Props) {
  const stripRef = useRef<HTMLDivElement>(null);
  const [clipped, setClipped] = useState<string[]>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [search, setSearch] = useState("");
  const dragFrom = useRef<number | null>(null);

  // Which tabs the strip has run out of room for is a measurement, not a
  // guess. Reserve room for the "+N" and the split control at the right end.
  useLayoutEffect(() => {
    const strip = stripRef.current;
    if (!strip) return;
    const measure = () => {
      const available = strip.clientWidth - 96;
      const out: string[] = [];
      for (const child of Array.from(strip.querySelectorAll<HTMLElement>("[data-tab]"))) {
        if (child.offsetLeft + child.offsetWidth > available) out.push(child.dataset.tab!);
      }
      setClipped(out);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(strip);
    return () => observer.disconnect();
  }, [tabs]);

  useEffect(() => {
    if (!menuOpen) setSearch("");
  }, [menuOpen]);

  const hidden = tabs.filter((tab) => clipped.includes(tab.id));
  // Searching matches the id as well as the title: a pipeline with four PCA
  // branches gives four tabs reading "PCA 5 PC", and the id is what tells
  // them apart.
  const matches = hidden.filter((tab) =>
    `${tab.title} ${tab.id}`.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="tabs" ref={stripRef} style={{ position: "relative" }} role="tablist">
      {tabs.map((tab, index) => {
        const KindIcon = KIND_ICONS[tab.kind];
        return (
          <div
            key={tab.id}
            data-tab={tab.id}
            draggable
            onDragStart={() => (dragFrom.current = index)}
            onDragOver={(event) => event.preventDefault()}
            onDrop={() => {
              if (dragFrom.current !== null) onMove(dragFrom.current, index);
              dragFrom.current = null;
            }}
            className={`tab${tab.id === activeId ? " act" : ""}${tab.transient ? " tr" : ""}`}
            role="tab"
            aria-selected={tab.id === activeId}
            onClick={() => onActivate(tab.id)}
            onDoubleClick={() => onPin(tab.id)}
          >
            <KindIcon />
            <span>{tab.title}</span>
            <button
              className="tabx"
              aria-label={`Close ${tab.title}`}
              onClick={(event) => {
                event.stopPropagation();
                onClose(tab.id);
              }}
            >
              ×
            </button>
          </div>
        );
      })}

      {hidden.length > 0 ? (
        <button
          className="tab mono"
          style={{
            color: "var(--ink3)",
            borderRight: "none",
            position: "absolute",
            right: 38,
            top: 0,
            bottom: 0,
            background: "var(--panel)",
          }}
          aria-label={`${hidden.length} more tabs`}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          +{hidden.length}
        </button>
      ) : null}

      <div
        style={{
          position: "absolute",
          right: 0,
          top: 0,
          bottom: 0,
          display: "flex",
          alignItems: "center",
          padding: "0 8px",
          background: "var(--panel)",
        }}
      >
        <button
          className="iconbtn"
          aria-label={splitId ? "Close split" : "Split view"}
          aria-pressed={Boolean(splitId)}
          onClick={() => {
            const other = tabs.find((tab) => tab.id !== activeId);
            onSplit(splitId ? null : (other?.id ?? null));
          }}
        >
          <SplitIcon />
        </button>
      </div>

      {menuOpen ? (
        <div className="omenu">
          <input
            autoFocus
            placeholder="Search tabs…"
            value={search}
            aria-label="Search tabs"
            onChange={(event) => setSearch(event.target.value)}
          />
          {matches.map((tab) => {
            const KindIcon = KIND_ICONS[tab.kind];
            return (
              <button
                key={tab.id}
                onClick={() => {
                  onActivate(tab.id);
                  setMenuOpen(false);
                }}
              >
                <KindIcon />
                {tab.title}
              </button>
            );
          })}
          {matches.length === 0 ? <div className="empty">Nothing matches.</div> : null}
        </div>
      ) : null}
    </div>
  );
}
