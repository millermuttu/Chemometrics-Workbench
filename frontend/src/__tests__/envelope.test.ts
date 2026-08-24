/** The envelope from PROPOSAL.md section 13, and the sentence it produces.
 *
 * The point of this state is honesty: a dataset past the limit gets told so,
 * with numbers, rather than a plot that freezes the tab.
 */
import { describe, expect, it } from "vitest";

import { checkEnvelope, envelopeSentence, ENVELOPE } from "@/states/envelope";

describe("the envelope", () => {
  it("admits the target dataset the proposal names", () => {
    expect(checkEnvelope(ENVELOPE.spectra, ENVELOPE.variables).within).toBe(true);
    expect(checkEnvelope(240, 100).within).toBe(true);
  });

  it("names which bound was crossed rather than saying 'too big'", () => {
    expect(checkEnvelope(42_000, 100).exceeded).toEqual(["spectra"]);
    expect(checkEnvelope(240, 6_200).exceeded).toEqual(["variables"]);
    expect(checkEnvelope(42_000, 6_200).exceeded).toEqual(["spectra", "variables", "memory"]);
  });

  it("reports float32 megabytes, which is the number that matters", () => {
    // 20,000 x 4,000 x 4 bytes is the proposal's ~320 MB.
    expect(checkEnvelope(ENVELOPE.spectra, ENVELOPE.variables).megabytes).toBe(320);
  });

  it("states the limit in a sentence a user can act on", () => {
    const sentence = envelopeSentence(42_000, 6_200);
    expect(sentence).toContain("42,000 × 6,200");
    expect(sentence).toContain("about 320 MB");
    expect(sentence).toMatch(/1,041 MB|1,042 MB/);
  });
});
