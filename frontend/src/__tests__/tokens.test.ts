/** The ported palettes are the artboards' palettes, and stay that way.
 *
 * design/canvas/_base.css is the token source. Porting it is a copy, so the
 * failure mode is drift: someone tunes a colour in the application and the
 * artboards quietly stop describing the product. This parses both files and
 * compares them, which is the only check that catches that.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { TOKEN_NAMES } from "@/styles/tokens";

const ROOT = path.resolve(import.meta.dirname, "../..");

function palettes(file: string): Record<string, Record<string, string>> {
  const css = readFileSync(file, "utf8");
  const out: Record<string, Record<string, string>> = {};
  for (const theme of ["t-light", "t-dark"]) {
    const block = new RegExp(`\\.${theme}\\s*\\{([^}]*)\\}`).exec(css);
    if (!block) throw new Error(`${file} has no .${theme} block`);
    out[theme] = Object.fromEntries(
      block[1]
        .split(";")
        .map((line) => line.trim())
        .filter((line) => line.startsWith("--"))
        .map((line) => {
          const [name, value] = line.split(":");
          return [name.slice(2).trim(), value.trim().toUpperCase()];
        }),
    );
  }
  return out;
}

const source = palettes(path.join(ROOT, "..", "design", "canvas", "_base.css"));
const ported = palettes(path.join(ROOT, "src", "styles", "tokens.css"));

describe.each(["t-light", "t-dark"])("%s", (theme) => {
  it("has every value the artboards have, unchanged", () => {
    expect(ported[theme]).toEqual(source[theme]);
  });

  it("covers exactly the names the interface asks for", () => {
    expect(Object.keys(ported[theme]).sort()).toEqual([...TOKEN_NAMES].sort());
  });
});

it("keeps the accent off the data palette", () => {
  // A failing node must never be readable as a red spectrum, which is why
  // --fail and --stale are semantic and --d1 to --d6 are Okabe-Ito.
  for (const theme of ["t-light", "t-dark"]) {
    const data = ["d1", "d2", "d3", "d4", "d5", "d6"].map((n) => ported[theme][n]);
    for (const semantic of ["accent", "fail", "stale"]) {
      expect(data).not.toContain(ported[theme][semantic]);
    }
  }
});
