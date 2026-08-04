import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';

import App from './App';
import './styles.css';

/// Reload once when the browser is running a page older than what is deployed.
///
/// GitHub Pages serves index.html with `max-age=600`, and a tab left open
/// never re-fetches it at all — so a deploy can leave someone on a stale
/// bundle indefinitely, which looks exactly like the new behaviour being
/// broken rather than simply not loaded. (That is how the browser-aware demo
/// tab appeared not to work: right code, cached page.) Asset filenames are
/// content-hashed, so comparing the one this page is running against the one
/// the server currently references is an exact staleness test — no version
/// number anybody has to remember to bump.
///
/// Guarded by sessionStorage so it can reload at most once per tab: if the
/// comparison ever went wrong the cost is one wasted reload, not a loop.
async function reloadIfStale() {
  const running = document.querySelector<HTMLScriptElement>('script[type=module][src]')?.src;
  if (!running) return;
  try {
    if (sessionStorage.getItem('zealotStaleReload')) return;
    const res = await fetch(location.pathname, {cache: 'reload'});
    if (!res.ok) return;
    const served = (await res.text()).match(/src="([^"]*index-[^"]*\.js)"/)?.[1];
    const file = (u: string) => u.split('/').pop();
    if (served && file(served) !== file(running)) {
      sessionStorage.setItem('zealotStaleReload', '1');
      location.reload();
    }
  } catch {
    // Offline, or storage blocked in a private window: nothing worth doing.
  }
}

void reloadIfStale();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
