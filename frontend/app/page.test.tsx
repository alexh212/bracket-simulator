import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

vi.mock("@/components/App", () => ({
  default: function MockApp() {
    return <div>Bracket Simulator Shell</div>;
  },
}));

import Home from "./page";

describe("Home page", () => {
  it("renders the main app shell", () => {
    render(<Home />);
    expect(screen.getByText("Bracket Simulator Shell")).toBeInTheDocument();
  });
});
