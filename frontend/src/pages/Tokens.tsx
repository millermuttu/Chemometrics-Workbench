import { useEffect, useState } from "react";

import { useProjects } from "@/api/queries";
import { TOKEN_GROUPS, type Theme } from "@/styles/tokens";

/** Reads what the browser resolved, not what was written down.
 *
 * A swatch built from a hard-coded value would be a picture of the palette
 * rather than the palette, and would keep looking right after the tokens
 * stopped being applied.
 */
function resolve(theme: Theme, name: string): string {
  const probe = document.createElement("div");
  probe.className = theme;
  document.body.append(probe);
  const value = getComputedStyle(probe).getPropertyValue(`--${name}`).trim();
  probe.remove();
  return value;
}

function Swatch({ theme, name }: { theme: Theme; name: string }) {
  const value = resolve(theme, name);
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-7 w-7 flex-none rounded-[3px]"
        style={{ background: value, border: "1px solid var(--rule)" }}
      />
      <div className="min-w-0">
        <div className="truncate text-[11.5px]">--{name}</div>
        <div className="mono text-[10.5px]" style={{ color: "var(--ink3)" }}>
          {value.toUpperCase()}
        </div>
      </div>
    </div>
  );
}

function Palette({ theme }: { theme: Theme }) {
  return (
    <section
      className={`${theme} flex-1 rounded-[3px] p-4`}
      style={{ background: "var(--panel)", border: "1px solid var(--rule)", color: "var(--ink)" }}
    >
      <h2 className="mono mb-3 text-[10px] uppercase tracking-[.11em]" style={{ color: "var(--ink3)" }}>
        {theme === "t-light" ? "Light" : "Dark"}
      </h2>
      {TOKEN_GROUPS.map((group) => (
        <div key={group.label} className="mb-3">
          <div className="mono mb-1.5 text-[10px] uppercase tracking-[.11em]" style={{ color: "var(--ink3)" }}>
            {group.label}
          </div>
          <div className="grid grid-cols-3 gap-2">
            {group.names.map((name) => (
              <Swatch key={name} theme={theme} name={name} />
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}

/** What the server says, proving the client speaks HTTP from its first commit.
 *
 * No fixture file is imported anywhere in this application. This is the whole
 * point of the stub server: 1.2 swaps its handlers and nothing here changes.
 */
function ServerCheck() {
  const { data, error, isPending } = useProjects();
  const message = isPending
    ? "GET /api/projects …"
    : error
      ? `GET /api/projects failed: ${error.message}`
      : `GET /api/projects → ${data?.[0]?.name ?? "no projects"}`;
  return (
    <p className="mono text-[11px]" style={{ color: error ? "var(--fail)" : "var(--ink3)" }}>
      {message}
    </p>
  );
}

export function Tokens() {
  const [theme, setTheme] = useState<Theme>("t-light");

  // The page chrome follows the toggle; the two palette panels always show
  // both, because the point of this page is comparing them.
  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  return (
    <main className="p-5" style={{ background: "var(--ground)", minHeight: "100%" }}>
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-[13px] font-semibold tracking-[-0.01em]">Design tokens</h1>
          <p className="text-[11px]" style={{ color: "var(--ink3)" }}>
            Ported from design/canvas/_base.css. IBM Plex Sans and Mono, bundled locally.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ServerCheck />
          <button
            className="h-[26px] rounded-[3px] px-2.5 text-[12px] font-medium"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--rule)",
              color: "var(--ink)",
            }}
            onClick={() => setTheme(theme === "t-light" ? "t-dark" : "t-light")}
          >
            {theme === "t-light" ? "Dark chrome" : "Light chrome"}
          </button>
        </div>
      </header>
      <div className="flex gap-4">
        <Palette theme="t-light" />
        <Palette theme="t-dark" />
      </div>
    </main>
  );
}
