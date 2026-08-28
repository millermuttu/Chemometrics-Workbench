/** A CSV the reader really reads, built in the test rather than committed.
 *
 * Phase 1.1's import tests pressed "Use the example file" and the stub replied
 * with a fixture; the file the user picked was discarded (#99). The file is now
 * the input, so a test that imports has to carry one. Generating it beats
 * committing one: the shape is stated where it is asserted, and nothing has to
 * be regenerated when a reader convention changes.
 *
 * The header row is the wavelength axis, so the axis is *read* rather than
 * reconstructed - "an axis is read or numbered, never invented" (#78-#80).
 */
export function spectraCsv(samples: number, channels: number): string {
  const axis = Array.from({ length: channels }, (_, index) => 1000 + index * 100);
  const rows = [["sample_id", ...axis.map((nm) => nm.toFixed(1)), "moisture"].join(",")];

  for (let sample = 0; sample < samples; sample += 1) {
    // Deterministic, and shaped like a spectrum: a broad band, a narrow one,
    // and a per-sample multiplicative offset - the scatter SNV exists for.
    const scale = 0.9 + (sample % 7) * 0.03;
    const values = axis.map((nm) => {
      const broad = 0.5 * Math.exp(-((nm - 1400) ** 2) / 240_000);
      const narrow = 0.2 * Math.exp(-((nm - 1800) ** 2) / 20_000);
      return (0.3 + broad + narrow * (1 + (sample % 5) * 0.1)) * scale;
    });
    const moisture = 60 + (sample % 11) * 0.4;
    rows.push(
      [
        `A${String(sample + 1).padStart(3, "0")}`,
        ...values.map((value) => value.toFixed(6)),
        moisture.toFixed(4),
      ].join(","),
    );
  }
  return rows.join("\n");
}
