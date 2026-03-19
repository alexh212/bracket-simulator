/** Keep in sync with `STORAGE_KEY` in `theme.ts` — inlined in layout script */
export const THEME_STORAGE_KEY = "bracket-simulator-theme";

/** Runs in <head> before first paint to prevent theme flash */
export const THEME_BOOTSTRAP_SCRIPT = `
(function(){
  try {
    var k=${JSON.stringify(THEME_STORAGE_KEY)};
    var t=localStorage.getItem(k);
    if(t!=="dark"&&t!=="light")
      t=window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";
    document.documentElement.setAttribute("data-theme",t);
  } catch(e) {
    document.documentElement.setAttribute("data-theme","light");
  }
})();`;
