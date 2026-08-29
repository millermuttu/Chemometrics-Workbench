/** The screen that says a dataset is past PROPOSAL.md section 13's envelope.
 *
 * `envelope.test.ts` proves the arithmetic; this proves the notice a user
 * actually reads, and that the screen guarding the plot reaches it.
 *
 * It is a component test rather than an end-to-end one, and that is a
 * decision with a reason: `docs/decisions/0005-overloaded-state-coverage.md`.
 * The state is entered by importing a dataset of about a gigabyte, which does
 * not belong in CI, and the envelope is reported rather than enforced - so
 * there is no smaller honest input that produces it. What is claimed here is
 * the component and the guard; what is not claimed is that a real oversize
 * import reaches them.
 *
 * Rendered through `react-dom/server` rather than a DOM: this state draws no
 * plot and runs no effects - that is its whole point - so static markup is
 * the entire output, and jsdom plus a testing library would be two
 * dependencies for a string this test already has.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { Overloaded } from "@/states/Overloaded";

// Plotly is a browser bundle and reads `self` when it loads. SpectraView
// imports it at module scope, and past the envelope it is never called - the
// guard returns before the plot exists. Stubbing the import is what lets that
// be asserted outside a browser.
vi.mock("plotly.js-gl2d-dist-min", () => ({ default: { react: vi.fn(), purge: vi.fn() } }));

describe("the beyond-the-envelope notice", () => {
  const html = renderToStaticMarkup(
    <Overloaded samples={42_000} variables={6_200} what="The spectra plot" />,
  );

  it("names every bound that was crossed, not just 'too big'", () => {
    expect(html).toContain("BEYOND THE V1 ENVELOPE · spectra · variables · memory");
  });

  it("says what is not drawn, what this dataset costs and what the limit is", () => {
    expect(html).toContain("The spectra plot is not drawn for this dataset.");
    expect(html).toContain("42,000 × 6,200 is about 1,042 MB as float32");
    expect(html).toContain("20,000 × 4,000, about 320 MB");
  });

  it("tells the user the two ways back inside the envelope", () => {
    expect(html).toContain("Select a range of variables or a subset of samples");
  });

  it("is announced rather than drawn silently", () => {
    expect(html).toContain('role="alert"');
  });

  it("keeps what is cheap available beside the notice", () => {
    const withMetadata = renderToStaticMarkup(
      <Overloaded samples={42_000} variables={6_200} what="The spectra plot">
        <p>240 samples · Tecator</p>
      </Overloaded>,
    );
    expect(withMetadata).toContain("240 samples · Tecator");
  });
});

describe("the screen that guards the plot", () => {
  it("renders the notice instead of mounting the plot, past the envelope", async () => {
    const { SpectraView } = await import("@/screens/SpectraView");
    const html = renderToStaticMarkup(
      <SpectraView nodeId="source" title="Raw spectra" samples={42_000} variables={6_200} />,
    );
    expect(html).toContain('data-testid="overloaded"');
    // The plot view was never entered: it would have asked for spectra and
    // rendered its loading pane. Past the envelope there is nothing to
    // prepare, which is the freeze this state exists to avoid.
    expect(html).not.toContain("Loading spectra");
  });
});
