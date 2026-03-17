"use client";

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { useSimulation } from "@/components/app/useSimulation";

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function HookHarness() {
  const { sim, running, teamsCatalog, startSim } = useSimulation();
  return (
    <div>
      <button
        onClick={() =>
          startSim({ n_sims: 100, latent_sigma: 0.06, team_overrides: {}, forced_picks: {} })
        }
      >
        Run
      </button>
      <div data-testid="running">{String(running)}</div>
      <div data-testid="done">{sim.done}</div>
      <div data-testid="leader">{Object.keys(sim.champion_pct)[0] || ""}</div>
      <div data-testid="catalog-count">{Object.keys(teamsCatalog).length}</div>
    </div>
  );
}

describe("useSimulation", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string) => {
        if (input.includes("/teams")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ Duke: { seed: 1 }, Auburn: { seed: 2 } }),
          });
        }

        if (input.includes("/model-info")) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ inference_data: { dataset: "TEAMS_2026" } }),
          });
        }

        return Promise.reject(new Error(`Unexpected fetch ${input}`));
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads bootstrap data and processes SSE progress/completion", async () => {
    render(<HookHarness />);

    await waitFor(() => {
      expect(screen.getByTestId("catalog-count")).toHaveTextContent("2");
    });

    fireEvent.click(screen.getByText("Run"));

    expect(MockEventSource.instances).toHaveLength(1);
    const stream = MockEventSource.instances[0];

    await act(async () => {
      stream.emit({ type: "start", n_sims: 100, assumptions_applied: 0, forced_picks_applied: 0 });
      stream.emit({
        type: "progress",
        done: 40,
        total: 100,
        champion_pct: { Duke: 55.2 },
        final_four_pct: { Duke: 78.1 },
      });
      stream.emit({
        type: "complete",
        champion_pct: { Duke: 61.5 },
        title_game_pct: { Duke: 72.1 },
        final_four_pct: { Duke: 83.3 },
        elite_eight_pct: { Duke: 88.4 },
        sweet_sixteen_pct: { Duke: 92.0 },
        round_of_32_pct: { Duke: 97.0 },
        first_four_pct: {},
        predicted_bracket: {},
        upset_watch: [],
        matchup_probs: {},
        n_sims: 100,
        model_used: "calibrated_ml_stack",
        elapsed_sec: 1.2,
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId("running")).toHaveTextContent("false");
    });

    expect(screen.getByTestId("done")).toHaveTextContent("100");
    expect(screen.getByTestId("leader")).toHaveTextContent("Duke");
    expect(stream.close).toHaveBeenCalled();
  });
});
