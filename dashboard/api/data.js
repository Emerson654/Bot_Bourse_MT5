// Données du dashboard, lues dans le dépôt GitHub du bot (public) et dans l'API GitHub Actions.
// Aucun secret n'est renvoyé au navigateur. Mise en cache 60 s chez Vercel.
const OWNER = "Emerson654";
const REPO = "Bot_Bourse_MT5";
const BRANCH = process.env.GH_BRANCH || "claude/charming-bardeen-c5d0o9";
const CAPITAL = Number(process.env.START_CAPITAL || 100000); // capital de départ de la démo (indice × capital)

const raw = (path, ref = BRANCH) => `https://raw.githubusercontent.com/${OWNER}/${REPO}/refs/heads/${ref}/${path}?t=${Date.now()}`;
const gh = (path) => {
  const headers = { Accept: "application/vnd.github+json", "User-Agent": "dashboard-bot" };
  if (process.env.GH_TOKEN) headers.Authorization = `Bearer ${process.env.GH_TOKEN}`;
  return fetch(`https://api.github.com/repos/${OWNER}/${REPO}${path}`, { headers });
};

async function json(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

async function csv(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return [];
    const [head, ...lines] = (await r.text()).trim().split("\n");
    const keys = head.split(",");
    return lines.filter(Boolean).map((l) => Object.fromEntries(l.split(",").map((v, i) => [keys[i], isNaN(+v) || v === "" ? v : +v])));
  } catch {
    return [];
  }
}

async function runs() {
  try {
    const r = await gh(`/actions/runs?per_page=40&branch=${encodeURIComponent(BRANCH)}`);
    if (!r.ok) return {};
    const { workflow_runs = [] } = await r.json();
    const pick = (file) => {
      const w = workflow_runs.find((x) => x.path === `.github/workflows/${file}`);
      return w ? { status: w.status, conclusion: w.conclusion, debut: w.run_started_at || w.created_at, maj: w.updated_at, lien: w.html_url } : null;
    };
    return { session: pick("live.yml"), backtest: pick("weekly.yml"), tournoi: pick("tournament8.yml") || pick("tournament7.yml") };
  } catch {
    return {};
  }
}

export default async function handler(req, res) {
  const [status, live, weekly, trades, equity, wf] = await Promise.all([
    json(raw("status.json", "status")),
    json(raw("results/live/backtest.json")),
    json(raw("results/weekly/latest.json")),
    csv(raw("journal/trades.csv")),
    csv(raw("journal/equity.csv")),
    runs(),
  ]);
  res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=300");
  res.status(200).json({
    capital: CAPITAL,
    statut: status,
    validation: live && { valide: live.validated, criteres: live.criteria, hors_echantillon: live.out_of_sample && {
      pf: live.out_of_sample.pf, max_dd: live.out_of_sample.max_dd, cagr: live.out_of_sample.cagr, trades: live.out_of_sample.trades,
      par_annee: live.out_of_sample.par_annee } , actifs: live.production_assets },
    backtest: weekly,
    trades,
    equity,
    workflows: wf,
    genere: new Date().toISOString(),
  });
}
