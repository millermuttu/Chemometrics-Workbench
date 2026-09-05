/** The artboard's icon set, at the artboard's 13px and 1.8 stroke.
 *
 * Traced from design/canvas/Main.dc.html rather than pulled from a library:
 * six paths weigh less than a dependency, and these are the exact shapes the
 * design uses.
 */
import type { ReactNode } from "react";

function Icon({ children, size = 13 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      aria-hidden="true"
      style={{ flex: "none" }}
    >
      {children}
    </svg>
  );
}

export const FlaskIcon = () => (
  <Icon>
    <path d="M9 3v6L4 19a1.6 1.6 0 0 0 1.4 2h13.2A1.6 1.6 0 0 0 20 19l-5-10V3" />
    <path d="M8 3h8" />
    <path d="M7 14h10" />
  </Icon>
);

export const DatasetIcon = () => (
  <Icon>
    <ellipse cx="12" cy="6" rx="8" ry="3" />
    <path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
    <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
  </Icon>
);

export const NodeIcon = () => (
  <Icon>
    <rect x="3" y="9" width="7" height="6" rx="1" />
    <rect x="14" y="9" width="7" height="6" rx="1" />
    <path d="M10 12h4" />
  </Icon>
);

export const PlotIcon = () => (
  <Icon>
    <path d="M3 20h18" />
    <path d="M4 16l5-6 4 3 6-8" />
  </Icon>
);

export const ModelIcon = () => (
  <Icon>
    <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
    <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
  </Icon>
);

export const LockIcon = () => (
  <Icon size={12}>
    <rect x="4" y="10" width="16" height="10" rx="2" />
    <path d="M8 10V7a4 4 0 0 1 8 0v3" />
  </Icon>
);

export const StopIcon = () => (
  <Icon size={11}>
    <rect x="6" y="6" width="12" height="12" rx="1.5" />
  </Icon>
);

export const SplitIcon = () => (
  <Icon size={12}>
    <rect x="3" y="5" width="18" height="14" rx="1.5" />
    <path d="M12 5v14" />
  </Icon>
);

export const ImportIcon = () => (
  <Icon>
    <path d="M12 3v12" />
    <path d="M7 10l5 5 5-5" />
    <path d="M4 20h16" />
  </Icon>
);

/** Two paths side by side: a comparison is two models on one screen. */
export const CompareIcon = () => (
  <Icon>
    <path d="M4 20V8" />
    <path d="M10 20V4" />
    <path d="M16 20v-9" />
    <path d="M22 20V6" />
  </Icon>
);

export const KIND_ICONS = {
  dataset: DatasetIcon,
  import: ImportIcon,
  pipeline: NodeIcon,
  spectra: PlotIcon,
  results: PlotIcon,
  compare: CompareIcon,
  experiment: FlaskIcon,
  model: ModelIcon,
} as const;
