import type { ReactNode } from "react";

/** The artboard's panel: a 24px title bar with the name left and a mono note
 * right, over a bordered surface. The grid of these is the answer to open
 * design question 11.1 - one analysis tab, several titled panels. */
export function Panel({
  title,
  note,
  width,
  height,
  children,
}: {
  title: string;
  note?: ReactNode;
  width?: number | string;
  height?: number;
  children: ReactNode;
}) {
  return (
    <section
      // A `section` is only a landmark when it is named, so this is the markup
      // the element already implied - and it makes a panel addressable by what
      // it is called rather than by a chain of styles. Several panels repeat a
      // word: "RMSECV" is a title, an axis label, a metric row and a header
      // figure on one screen.
      aria-label={title}
      style={{
        width: width ?? "auto",
        height,
        flex: width ? "none" : 1,
        minWidth: 0,
        border: "1px solid var(--rule2)",
        borderRadius: 4,
        background: "var(--surface)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: 24,
          flex: "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          padding: "0 9px",
          borderBottom: "1px solid var(--rule2)",
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 600 }}>{title}</span>
        {typeof note === "string" ? (
          <span className="mono" style={{ fontSize: 9.5, color: "var(--ink3)" }}>
            {note}
          </span>
        ) : (
          note
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        {children}
      </div>
    </section>
  );
}
