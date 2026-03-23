"use client";

import { useMemo } from "react";
import type { RealGame, RealResults } from "@/lib/api";
import { RESULTS_POLL_MS } from "@/lib/api";
import { BORDER_OUTER, TEXT, TEXT_MUTED, TEXT_SUBTLE } from "@/components/app/shared";

interface TickerProps {
  results: RealResults | null;
}

function GameChip({ game }: { game: RealGame }) {
  const isLive = game.status === "live";
  const aWonFinal = game.status === "final" && game.winner === game.team_a;
  const bWonFinal = game.status === "final" && game.winner === game.team_b;
  const aLeadsLive = isLive && game.score_a > game.score_b;
  const bLeadsLive = isLive && game.score_b > game.score_a;
  const aStrong = aWonFinal || aLeadsLive;
  const bStrong = bWonFinal || bLeadsLive;

  const isUpset = game.status === "final" && (
    (aWonFinal && game.seed_a > game.seed_b) ||
    (bWonFinal && game.seed_b > game.seed_a)
  );

  const row = (strong: boolean) => {
    if (isUpset) return strong ? "var(--chip-upset-strong)" : "var(--chip-upset-weak)";
    if (isLive) return strong ? "var(--chip-live-strong)" : "var(--chip-live-weak)";
    return strong ? TEXT : TEXT_SUBTLE;
  };
  const seedParen = (strong: boolean) => {
    if (isUpset) return strong ? "var(--chip-upset-weak)" : "var(--chip-upset-dash)";
    if (isLive) return strong ? "var(--chip-live-weak)" : "var(--chip-live-dash)";
    return strong ? TEXT_MUTED : TEXT_SUBTLE;
  };

  const shell = isUpset
    ? { background: "var(--chip-upset-bg)", border: "var(--chip-upset-border)" }
    : isLive
      ? { background: "var(--chip-live-bg)", border: "var(--chip-live-border)" }
      : { background: "var(--chip-bg)", border: "var(--chip-border)" };

  return (
    <div
      title={isLive && game.status_detail ? game.status_detail : undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0,
        padding: "4px 10px",
        background: shell.background,
        borderRadius: 8,
        border: `1px solid ${shell.border}`,
        fontSize: 11,
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {game.status === "final" && (
        <span style={{
          fontSize: 8,
          fontWeight: 700,
          color: "#fff",
          background: isUpset ? "#d97706" : "#6b7280",
          borderRadius: 3,
          padding: "1px 4px",
          marginRight: 8,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
        }}>
          {isUpset ? "UPSET" : "FINAL"}
        </span>
      )}
      {isLive && (
        <span style={{
          fontSize: 8,
          fontWeight: 700,
          color: "#fff",
          background: "#16a34a",
          borderRadius: 3,
          padding: "1px 4px",
          marginRight: 8,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          animation: "pulse-live 2s ease-in-out infinite",
        }}>
          LIVE
        </span>
      )}
      {game.game_date && (
        <span style={{
          fontSize: 8,
          color: TEXT_SUBTLE,
          marginRight: 6,
          opacity: 0.7,
        }}>
          {game.game_date}
        </span>
      )}
      <span style={{
        fontWeight: aStrong ? 700 : 400,
        color: row(aStrong),
      }}>
        <span style={{ fontSize: 9, color: seedParen(aStrong) }}>({game.seed_a})</span>{" "}
        {game.team_a}{" "}
        <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{game.score_a}</span>
      </span>
      <span style={{ margin: "0 6px", color: isUpset ? "var(--chip-upset-dash)" : isLive ? "var(--chip-live-dash)" : BORDER_OUTER, fontSize: 10 }}>—</span>
      <span style={{
        fontWeight: bStrong ? 700 : 400,
        color: row(bStrong),
      }}>
        <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{game.score_b}</span>{" "}
        {game.team_b}{" "}
        <span style={{ fontSize: 9, color: seedParen(bStrong) }}>({game.seed_b})</span>
      </span>
    </div>
  );
}

function formatLastUpdated(iso: string | null): string | null {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHrs = Math.floor(diffMin / 60);
    if (diffHrs < 24) return `${diffHrs}h ago`;
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return null;
  }
}

export default function Ticker({ results }: TickerProps) {
  const scrollGames = useMemo(
    () => (results?.games || []).filter((g) => g.status === "final" || g.status === "live"),
    [results],
  );

  const finalOnly = useMemo(
    () => scrollGames.filter((g) => g.status === "final"),
    [scrollGames],
  );

  const liveCount = scrollGames.length - finalOnly.length;

  if (!results || scrollGames.length === 0) return null;

  const upsetCount = finalOnly.filter((g) => {
    const aWon = g.winner === g.team_a;
    return (aWon && g.seed_a > g.seed_b) || (!aWon && g.seed_b > g.seed_a);
  }).length;

  const tickerDuration = Math.max(20, scrollGames.length * 3);
  const updatedLabel = formatLastUpdated(results.last_updated);

  return (
    <div style={{
      borderRadius: 10,
      border: `1px solid ${BORDER_OUTER}`,
      overflow: "hidden",
      marginBottom: 12,
      background: "var(--ticker-shell-bg)",
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "5px 12px",
        borderBottom: `1px solid ${BORDER_OUTER}`,
        background: "#111",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {liveCount > 0 && (
            <span style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              fontSize: 9,
              fontWeight: 700,
              color: "#22c55e",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "#22c55e",
                animation: "pulse-live 2s ease-in-out infinite",
                flexShrink: 0,
              }} />
              LIVE
            </span>
          )}
          {results.tournament_status && (
            <span style={{
              fontSize: 10,
              fontWeight: 700,
              color: liveCount > 0 ? "#a3e635" : "#fff",
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}>
              {results.tournament_status}
            </span>
          )}
          <span style={{ fontSize: 9, color: TEXT_SUBTLE }}>
            {finalOnly.length} final{liveCount > 0 ? ` · ${liveCount} live` : ""}
            {upsetCount > 0 ? ` · ${upsetCount} upset${upsetCount !== 1 ? "s" : ""}` : ""}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {updatedLabel && (
            <span style={{ fontSize: 8, color: TEXT_SUBTLE, opacity: 0.7 }}>
              {updatedLabel}
            </span>
          )}
          {results.live_scores_enabled && (
            <span
              style={{ fontSize: 8, color: TEXT_SUBTLE, opacity: 0.6 }}
              title="Scores refresh automatically while the page is open"
            >
              ~{Math.round(RESULTS_POLL_MS / 1000)}s refresh
            </span>
          )}
        </div>
      </div>

      <div style={{ overflow: "hidden", padding: "6px 0" }}>
        <div
          className="ticker-track"
          style={{ "--ticker-duration": `${tickerDuration}s` } as React.CSSProperties}
        >
          {[...scrollGames, ...scrollGames].map((g, i) => (
            <div key={`${g.region}-${g.round}-${g.game_index}-${i}`} style={{ paddingLeft: i === 0 ? 12 : 8, paddingRight: 8 }}>
              <GameChip game={g} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
