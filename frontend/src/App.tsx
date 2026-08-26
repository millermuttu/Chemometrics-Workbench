import { Tokens } from "./pages/Tokens";
import { Shell } from "./shell/Shell";

/** The screens are #44 onward; the shell is what they open inside.
 *
 * The token page that stood here through #42 moves to /tokens and stays
 * reachable: it is what proves the palettes and the bundled type, and the
 * end-to-end test still checks it. One comparison is cheaper than a router,
 * and a router arrives when a screen needs a URL of its own.
 */
export function App() {
  return window.location.pathname === "/tokens" ? <Tokens /> : <Shell />;
}
