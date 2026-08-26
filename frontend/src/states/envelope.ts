/** The performance envelope, from PROPOSAL.md section 13.
 *
 * "up to ~20,000 spectra × ~4,000 variables, held in memory as float32
 * (≈ 320 MB). Beyond this, v1 documents the limit rather than pretending
 * otherwise."
 *
 * Documenting it is this module's whole job: a dataset past the envelope gets
 * a screen that says so, with numbers, instead of a plot that hangs the tab.
 */

export const ENVELOPE = { spectra: 20_000, variables: 4_000, bytesPerValue: 4 };

export interface Envelope {
  within: boolean;
  cells: number;
  megabytes: number;
  /** Which bound is exceeded, for a message that names the actual problem. */
  exceeded: ("spectra" | "variables" | "memory")[];
}

export function checkEnvelope(samples: number, variables: number): Envelope {
  const cells = samples * variables;
  const megabytes = (cells * ENVELOPE.bytesPerValue) / 1_000_000;
  const limitMegabytes =
    (ENVELOPE.spectra * ENVELOPE.variables * ENVELOPE.bytesPerValue) / 1_000_000;

  const exceeded: Envelope["exceeded"] = [];
  if (samples > ENVELOPE.spectra) exceeded.push("spectra");
  if (variables > ENVELOPE.variables) exceeded.push("variables");
  if (megabytes > limitMegabytes) exceeded.push("memory");

  return { within: exceeded.length === 0, cells, megabytes, exceeded };
}

export function envelopeSentence(samples: number, variables: number): string {
  const { megabytes } = checkEnvelope(samples, variables);
  return (
    `${samples.toLocaleString()} × ${variables.toLocaleString()} is about ` +
    `${Math.round(megabytes).toLocaleString()} MB as float32. Version 1 is built and tested to ` +
    `${ENVELOPE.spectra.toLocaleString()} × ${ENVELOPE.variables.toLocaleString()}, about 320 MB.`
  );
}
