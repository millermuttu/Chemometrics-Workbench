import { checkEnvelope, envelopeSentence } from "@/states/envelope";

/** Past the envelope, the application says so.
 *
 * PROPOSAL.md section 13: "Beyond this, v1 documents the limit rather than
 * pretending otherwise." Pretending, here, would be starting a plot of eighty
 * million points and letting the tab freeze. What is cheap stays available -
 * the metadata, the sample table - and only the plot is withheld.
 */
export function Overloaded({
  samples,
  variables,
  what,
  children,
}: {
  samples: number;
  variables: number;
  what: string;
  children?: React.ReactNode;
}) {
  const { exceeded } = checkEnvelope(samples, variables);
  return (
    <div
      role="alert"
      data-testid="overloaded"
      style={{
        margin: 12,
        padding: "10px 12px",
        borderRadius: 3,
        border: "1px solid var(--stale)",
        background: "var(--staleSoft)",
      }}
    >
      <div className="mono" style={{ fontSize: 10, color: "var(--stale)" }}>
        BEYOND THE V1 ENVELOPE · {exceeded.join(" · ")}
      </div>
      <p style={{ margin: "4px 0 0", color: "var(--ink)" }}>
        {what} is not drawn for this dataset. {envelopeSentence(samples, variables)}
      </p>
      <p style={{ margin: "4px 0 0", fontSize: 11, color: "var(--ink3)" }}>
        Select a range of variables or a subset of samples to bring it inside the envelope.
        Everything that does not need the whole array is still available.
      </p>
      {children}
    </div>
  );
}
