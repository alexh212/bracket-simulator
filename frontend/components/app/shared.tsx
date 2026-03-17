"use client";

import { useState, type ReactNode } from "react";

export const GREEN = "#22c55e";
export const ACCENT = "#000";
export const ACCENT_SOFT = "#fafafa";
export const BORDER_OUTER = "#ebeff5";
export const BORDER_INNER = "#f1f5f9";
export const BORDER_SUBTLE = "#e8e8e8";
export const BG_HEADER = "#fafafa";
export const BG_ALT = "#f8f8f8";
export const PAD_HEADER = "8px 10px";
export const PAD_BODY = "14px 16px";
export const ROUNDS = ["R32", "S16", "E8", "F4", "Final", "Champ"] as const;
export const REGIONS = ["East", "West", "Midwest", "South"] as const;
export const R64_ORDER: [number, number][] = [[1, 16], [8, 9], [5, 12], [4, 13], [6, 11], [3, 14], [7, 10], [2, 15]];

export const pickKey = (region: string, round: number, gi: number) => `${region}:${round}:${gi}`;

export const top = (o: Record<string, number>, n = 10) =>
  Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, n);

export const pct = (n: number, d = 1) => `${n.toFixed(d)}%`;

export const clampProb = (p: number) => Math.min(0.999, Math.max(0.001, p));

export const toML = (p: number) => {
  const q = clampProb(p);
  return q >= 0.5 ? `-${Math.round((q / (1 - q)) * 100)}` : `+${Math.round(((1 - q) / q) * 100)}`;
};

export const sumPct = (o: Record<string, number>) =>
  Object.values(o).reduce((acc, v) => acc + v, 0);

export function SectionGap() {
  return <div style={{ marginTop: 14 }} />;
}

export function SH({ label }: { label: string }) {
  return (
    <div style={{ background: "#000", padding: "7px 14px" }}>
      <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", color: "#fff" }}>{label}</span>
    </div>
  );
}

export function Seed({ n }: { n: number }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 17,
        height: 17,
        fontSize: 9,
        fontWeight: 600,
        flexShrink: 0,
        border: "1px solid #d1d5db",
        background: n <= 4 ? "#111" : "transparent",
        color: n <= 4 ? "#fff" : "#111",
      }}
    >
      {n}
    </span>
  );
}

export function GroupTitle({ label, hint }: { label: string; hint?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "2px 2px 0" }}>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "#6b7280" }}>{label}</div>
      {hint && <div style={{ fontSize: 10, color: "#9ca3af" }}>{hint}</div>}
    </div>
  );
}

export function Collapse({
  label,
  children,
  defaultOpen = false,
}: {
  label: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, marginTop: 0, background: "#fff", borderRadius: 14, overflow: "hidden" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: "100%",
          padding: "10px 12px",
          background: open ? BG_HEADER : "#fff",
          border: "none",
          borderBottom: open ? `1px solid ${BORDER_INNER}` : "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: 11,
          fontWeight: 600,
          color: "#111",
          textAlign: "left",
        }}
      >
        <span>{label}</span>
        <span style={{ color: "#999", fontSize: 10 }}>{open ? "HIDE" : "SHOW"}</span>
      </button>
      {open && children}
    </div>
  );
}
