// Lance un backtest (workflow « Backtest hebdomadaire ») depuis le dashboard, protégé par mot de passe.
// Variables Vercel : GH_TOKEN (jeton GitHub, droit « Actions : écriture » sur le dépôt) et DASHBOARD_PASSWORD.
const OWNER = "Emerson654";
const REPO = "Bot_Bourse_MT5";
const BRANCH = process.env.GH_BRANCH || "claude/charming-bardeen-c5d0o9";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, erreur: "POST uniquement" });
  const { password } = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  if (!process.env.DASHBOARD_PASSWORD || password !== process.env.DASHBOARD_PASSWORD) {
    return res.status(401).json({ ok: false, erreur: "Mot de passe incorrect" });
  }
  if (!process.env.GH_TOKEN) return res.status(500).json({ ok: false, erreur: "GH_TOKEN absent dans Vercel" });
  const r = await fetch(`https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/weekly.yml/dispatches`, {
    method: "POST",
    headers: { Accept: "application/vnd.github+json", Authorization: `Bearer ${process.env.GH_TOKEN}`, "User-Agent": "dashboard-bot" },
    body: JSON.stringify({ ref: BRANCH }),
  });
  if (r.status === 204) return res.status(200).json({ ok: true });
  return res.status(502).json({ ok: false, erreur: `GitHub a répondu ${r.status}` });
}
