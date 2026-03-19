"use client";

import { useEffect, useState, useCallback } from "react";
import type { RealGame, RealResults } from "@/lib/api";
import { getResults } from "@/lib/api";

interface TickerProps {
  onImportResults?: (picks: Record<string, string>, count: number) => void;
}

function GameChip({ game }: { game: RealGame }) {
  const aWon = game.winner === game.team_a;
  const bWon = game.winner === game.team_b;
  const isUpset = game.status === "final" && (
    (aWon && game.seed_a > game.seed_b) ||
    (bWon && game.seed_b > game.seed_a)
  );

  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 0,
      padding: "4px 10px",
      background: isUpset ? "#fef3c7" : "#fff",
      borderRadius: 8,
      border: `1px solid ${isUpset ? "#fbbf24" : "#e5e7eb"}`,
      fontSize: 11,
      whiteSpace: "nowrap",
      flexShrink: 0,
    }}>
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
      <span style={{
        fontWeight: aWon ? 700 : 400,
        color: aWon ? "#111" : "#9ca3af",
      }}>
        <span style={{ fontSize: 9, color: aWon ? "#6b7280" : "#d1d5db" }}>({game.seed_a})</span>{" "}
        {game.team_a}{" "}
        <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{game.score_a}</span>
      </span>
      <span style={{ margin: "0 6px", color: "#d1d5db", fontSize: 10 }}>—</span>
      <span style={{
        fontWeight: bWon ? 700 : 400,
        color: bWon ? "#111" : "#9ca3af",
      }}>
        <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>{game.score_b}</span>{" "}
        {game.team_b}{" "}
        <span style={{ fontSize: 9, color: bWon ? "#6b7280" : "#d1d5db" }}>({game.seed_b})</span>
      </span>
    </div>
  );
}

export default function Ticker({ onImportResults }: TickerProps) {
  const [data, setData] = useState<RealResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [imported, setImported] = useState(false);

  useEffect(() => {
    getResults()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  const handleImport = useCallback(() => {
    if (!data?.games.length || !onImportResults) return;
    const picks: Record<string, string> = {};
    for (const g of data.games) {
      if (g.status === "final" && g.winner) {
        picks[`${g.region}:${g.round}:${g.game_index}`] = g.winner;
      }
    }
    onImportResults(picks, Object.keys(picks).length);
    setImported(true);
  }, [data, onImportResults]);

  if (loading) return null;

  const finalGames = data?.games.filter(g => g.status === "final") || [];
  if (finalGames.length === 0) return null;

  const upsetCount = finalGames.filter(g => {
    const aWon = g.winner === g.team_a;
    return (aWon && g.seed_a > g.seed_b) || (!aWon && g.seed_b > g.seed_a);
  }).length;

  const tickerDuration = Math.max(20, finalGames.length * 3);

  return (
    <div style={{
      borderRadius: 10,
      border: "1px solid #e5e7eb",
      overflow: "hidden",
      marginBottom: 12,
      background: "#fafafa",
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "5px 12px",
        borderBottom: "1px solid #e5e7eb",
        background: "#111",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 9,
            fontWeight: 700,
            color: "#fff",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
          }}>
            Live Results
          </span>
          <span style={{ fontSize: 9, color: "#9ca3af" }}>
            {finalGames.length} games · {upsetCount} upset{upsetCount !== 1 ? "s" : ""}
          </span>
          {data?.tournament_status && (
            <span style={{ fontSize: 9, color: "#6b7280" }}>
              · {data.tournament_status}
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {onImportResults && (
            <button
              onClick={handleImport}
              disabled={imported}
              title="Lock all completed game winners as forced picks — your next simulation will use these real results and only simulate remaining games"
              style={{
                fontSize: 9,
                fontWeight: 600,
                padding: "3px 10px",
                borderRadius: 6,
                border: "none",
                cursor: imported ? "default" : "pointer",
                background: imported ? "#22c55e" : "#fff",
                color: imported ? "#fff" : "#111",
                transition: "all 0.2s",
              }}
            >
              {imported ? `✓ ${finalGames.length} results locked` : "Lock results as picks →"}
            </button>
          )}
        </div>
      </div>

      <div style={{ overflow: "hidden", padding: "6px 0" }}>
        <div
          className="ticker-track"
          style={{ "--ticker-duration": `${tickerDuration}s` } as React.CSSProperties}
        >
          {[...finalGames, ...finalGames].map((g, i) => (
            <div key={i} style={{ paddingLeft: i === 0 ? 12 : 8, paddingRight: 8 }}>
              <GameChip game={g} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
