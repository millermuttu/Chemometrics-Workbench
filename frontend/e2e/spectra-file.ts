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
 *
 * ## It has to have as many components as anything asks of it
 *
 * The first version of this varied two things across samples - an overall
 * scale and one band's height - so the matrix had **two** real components. The
 * walkthrough then asked PCA for five and got them, because the rank tolerance
 * was quoted in float64 terms and admitted three singular values of numerical
 * noise as though they were structure. #101 fixed the tolerance and this file
 * immediately failed with "5 components were asked of a matrix of rank 3",
 * which was the honest answer to a question that should never have been asked.
 *
 * So each band's height now varies **independently** per sample. Eight bands
 * give a matrix of rank nine after SNV, a first derivative and centring - the
 * singular values run 3.55, 2.98, 1.62, 1.28, 0.55, 0.36, 0.26, so a PCA of
 * five components is fitting structure that is really there. Deterministic,
 * and not random: the same call gives the same file every run.
 */
export function spectraCsv(samples: number, channels: number): string {
  const axis = Array.from({ length: channels }, (_, index) => 1000 + index * 100);
  const rows = [["sample_id", ...axis.map((nm) => nm.toFixed(1)), "moisture"].join(",")];
  const bands = 8;
  const span = channels * 100 - 100;

  for (let sample = 0; sample < samples; sample += 1) {
    const values = axis.map((nm) => {
      // A broad baseline everything sits on, then the bands.
      let y = 0.3 + 0.1 * Math.exp(-((nm - 1500) ** 2) / 400_000);
      for (let band = 0; band < bands; band += 1) {
        const centre = 1050 + (band * span) / bands;
        const width = 60 + 18 * band;
        // The `sample * band` term is what makes these independent rather than
        // eight copies of one pattern: without it every sample moves the same
        // way and the matrix collapses back to a couple of components.
        const height =
          0.05 + 0.3 * (0.5 + 0.5 * Math.sin(2.399 * sample + 1.7 * band + 0.31 * sample * band));
        y += height * Math.exp(-((nm - centre) ** 2) / (2 * width ** 2));
      }
      // A per-sample multiplicative offset - the scatter SNV exists for.
      return y * (0.92 + 0.16 * (0.5 + 0.5 * Math.sin(0.77 * sample)));
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
