# ==================================================================
# TIPPING FLOOR - full stack food-waste system, one Colab cell
#   frontend : HTML/JS dashboard           (static/index.html)
#   backend  : Flask REST API              (app.py)
#   database : SQLite waste log + your entries (kitchen.py, tickets.py)
#   ML       : forecasting + anomaly det.  (pipeline.py)
# Pick a site tab and add your own tickets: the forecast and charts recompute live.
# Run this cell, upload your archive zip when asked, and the app opens below.
# ==================================================================
import os, sys, glob, zipfile, threading, time
IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
ROOT = "/content/tipping-floor" if IN_COLAB else os.path.abspath("tipping-floor-colab")
os.makedirs(ROOT + "/static", exist_ok=True); os.makedirs(ROOT + "/data", exist_ok=True)

FILES = {}

FILES['pipeline.py'] = r'''"""Tipping Floor - data cleaning, forecasting, anomaly detection.
Everything here is computed from the weighbridge CSVs in ./data. No fake numbers."""
import glob, os, functools
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"boralasgamuwa": "Boralesgamuwa", "dehiwala": "Dehiwala", "homagama": "Homagama",
         "moratuwa": "Moratuwa", "austin": "Austin, TX", "ballarat": "Ballarat, AU"}
ORGANIC = {"Biodegradable", "Sorted Organic Waste"}
audit = []  # human-readable notes about problems found in the raw data


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "*.csv"))):
        key = os.path.basename(f).replace("open_source_", "").split("_")[0]
        d = pd.read_csv(f)
        d["site"] = key
        if "waste_type" not in d:
            d["waste_type"] = "MSW"  # open datasets only give one total stream
        d["date"] = pd.to_datetime(d["ticket_date"])
        frames.append(d[["site", "date", "waste_type", "net_weight_kg"]])
    df = pd.concat(frames, ignore_index=True)
    df["kg"] = df.pop("net_weight_kg").astype(float)

    # Moratuwa: first rows look like YYYY-DD-MM (01-12 ... 09-12) sitting before 2014-09-16
    m = (df.site == "moratuwa") & (df.date.dt.day == 12) & (df.date < "2014-09-16")
    if m.any():
        df.loc[m, "date"] = pd.to_datetime(df.loc[m, "date"].dt.strftime("%Y-%d-%m"), errors="coerce")
        audit.append(f"Moratuwa: {int(m.sum())} early-2014 tickets looked day/month swapped (2014-01-12 -> 1 Dec). "
                     "Reinterpreted as YYYY-DD-MM. Best guess, not verified.")
    for s, n in df[df.kg == 0].groupby("site").size().items():
        audit.append(f"{NAMES[s]}: {n} tickets with weight = 0 (excluded from forecasts, listed as flagged).")
    if df.waste_type.isin(["Slaghter House Waste", "Indutrial Sludge Waste", "Mesuring", "Wood Trank"]).any():
        audit.append("Some waste-type labels are misspelt ('Slaghter', 'Indutrial'...). Kept as-is.")
    audit.append("Austin and Ballarat report one total per ticket, so no organic analysis there, just forecast + anomalies.")
    # drop a site's sparse lead-in (Austin has 1 ticket in 2003) so the model doesn't learn from gaps
    keep = []
    for s, g in df.groupby("site"):
        dts = g.date.sort_values().values
        cnt = pd.Series(1, index=pd.DatetimeIndex(dts)).rolling("90D").sum()
        first = cnt[cnt >= 40].index[0] - pd.Timedelta(days=89)
        if first > g.date.min() + pd.Timedelta(days=30):
            audit.append(f"{NAMES[s]}: dropped sparse tickets before {first:%Y-%m-%d} ({int((g.date < first).sum())} rows).")
        keep.append(g[g.date >= first])
    df = pd.concat(keep)
    return df[df.kg > 0].reset_index(drop=True), df[df.kg == 0]


def wend(d):
    """Sunday that ends the week containing date d (matches resample('W'))."""
    d = pd.Timestamp(d).normalize()
    return d + pd.Timedelta(days=6 - d.dayofweek)


def weekly(daily, keep=(), overrides=None):
    """Weekly tonnes. `keep` = weeks the user touched (never treated as partial/incomplete);
    `overrides` = {week_end: tonnes} the user set directly."""
    keep, overrides = pd.DatetimeIndex(list(keep)), overrides or {}
    w = daily.resample("W").sum(min_count=1)
    n = daily.resample("W").count().reindex(w.index)
    med = n[n > 0].median()
    w = w.iloc[1:]  # first week is partial
    if len(w) and w.index[-1] not in keep:
        w = w.iloc[:-1]  # last week is partial unless the user filled it in
    n = n.reindex(w.index)
    w[(n < 0.5 * med) & ~w.index.isin(keep)] = np.nan  # weeks with half the usual tickets are incomplete, not "low waste"
    if overrides:
        w = w.reindex(pd.date_range(min(w.index[0], min(overrides)), max(w.index[-1], max(overrides)), freq="W"))
        for k, t in overrides.items():
            w.loc[k] = t * 1000.0
    filled = w.isna()
    return w.interpolate(limit=3) / 1000.0, filled  # tonnes


def feats(s):
    d = pd.DataFrame({"y": np.log1p(s)})
    for l in (1, 2, 3, 4, 8, 52):
        d[f"l{l}"] = d.y.shift(l)
    d["m4"] = d.y.shift(1).rolling(4).mean()
    d["m13"] = d.y.shift(1).rolling(13).mean()
    wk = s.index.isocalendar().week.astype(int).values
    d["ws"], d["wc"] = np.sin(2 * np.pi * wk / 52), np.cos(2 * np.pi * wk / 52)
    return d


NEED = ["l1", "l2", "l3", "l4", "m4"]


def fit(d):
    d = d.dropna(subset=["y"] + NEED)
    X = d.drop(columns="y")
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                         min_samples_leaf=8, random_state=7).fit(X, d.y), list(X.columns)


def forecast_site(s, horizon=8, holdout=26):
    s = s[s > 0].asfreq("W").interpolate(limit=4).dropna()
    F = feats(s)
    # honest backtest: train before the last `holdout` weeks, predict each held-out week one step ahead
    mdl, cols = fit(F.iloc[:-holdout])
    ok = F.iloc[-holdout:].dropna(subset=NEED)
    act, base = np.expm1(ok.y), np.expm1(ok.m4)  # baseline = mean of previous 4 weeks
    gb = mdl.predict(ok[cols])
    pred = np.expm1(0.5 * gb + 0.5 * ok.m4.values)  # blend: GBM + recent average, in log space
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    mape = lambda a, b: float(np.mean(np.abs(a - b) / a) * 100)
    metrics = {"model_mae": mae(act, pred), "base_mae": mae(act, base),
               "model_mape": mape(act, pred), "base_mape": mape(act, base), "weeks": int(len(ok))}
    metrics["beats_baseline"] = bool(metrics["model_mae"] < metrics["base_mae"])
    lo_q, hi_q = np.quantile(np.log1p(act.values) - np.log1p(pred), [0.1, 0.9])
    # final model on all data, recursive forecast (each predicted week feeds the next)
    mdl, cols = fit(F)
    hist, out = s.copy(), []
    for _ in range(horizon):
        nxt = hist.index[-1] + pd.Timedelta(weeks=1)
        probe = pd.concat([hist, pd.Series([hist.iloc[-1]], index=[nxt])])  # placeholder value; features use only the past
        fr = feats(probe).iloc[[-1]]
        v = float(np.expm1(0.5 * mdl.predict(fr[cols])[0] + 0.5 * fr.m4.iloc[0]))
        hist.loc[nxt] = v
        out.append({"week": nxt.strftime("%Y-%m-%d"), "t": v,
                    "lo": v * float(np.exp(lo_q)), "hi": v * float(np.exp(hi_q))})
    bt = [{"week": d.strftime("%Y-%m-%d"), "actual": float(a), "pred": float(p)} for d, a, p in zip(ok.index, act, pred)]
    return out, metrics, bt


def anomalies(daily, zeros_site, user_dates=()):
    """Robust z-score vs the previous 10 same-weekday tickets (median/MAD), so Sundays aren't 'drops'."""
    grp = daily.groupby(daily.index.dayofweek)
    med = grp.transform(lambda x: x.rolling(10, min_periods=5).median().shift(1))
    mad = (daily - med).abs().groupby(daily.index.dayofweek).transform(
        lambda x: x.rolling(10, min_periods=5).median().shift(1)) * 1.4826
    z = ((daily - med) / mad.replace(0, np.nan)).dropna()
    flag = z[z.abs() > 5]
    rows = [{"date": d.strftime("%Y-%m-%d"), "tonnes": float(daily[d] / 1000), "typical": float(med[d] / 1000),
             "z": float(v), "kind": "spike" if v > 0 else "drop",
             "user": d.strftime("%Y-%m-%d") in user_dates} for d, v in flag.items()]
    rows = sorted(rows, key=lambda r: -abs(r["z"]))[:12]
    for d in zeros_site.date.dt.strftime("%Y-%m-%d").unique()[:5]:
        rows.append({"date": d, "tonnes": 0.0, "typical": None, "z": None, "kind": "zero-weight"})
    return rows, int(len(flag))


_BASE = {}


def base():
    """Load + clean the raw CSVs once; reused every time a site is rebuilt with user data."""
    if not _BASE:
        audit.clear()
        df, zeros = load()
        _BASE.update(df=df, zeros=zeros, audit=list(audit))
    return _BASE["df"], _BASE["zeros"]


@functools.lru_cache(maxsize=None)
def bounds(site):
    """What the user is allowed to enter for a site, plus 'typical' values to help them."""
    df, _ = base()
    g = df[df.site == site]
    daily = g.groupby("date").kg.sum()
    wk = daily.resample("W").sum()
    last = g.date.max()
    return {"first": g.date.min().strftime("%Y-%m-%d"), "last": last.strftime("%Y-%m-%d"),
            "day_max": (last + pd.Timedelta(days=35)).strftime("%Y-%m-%d"),
            "week_max": (wend(last) + pd.Timedelta(weeks=4)).strftime("%Y-%m-%d"),
            "max_day_t": float(daily.max() / 1000 * 3), "max_week_t": float(wk.max() / 1000 * 3),
            "typical_day_t": float(daily[daily > 0].tail(60).median() / 1000),
            "typical_week_t": float(wk.iloc[-9:-1].mean() / 1000),
            "types": sorted(g.waste_type.unique())}


def validate(site, kind, day, wtype, tonnes):
    """Return a clean (day, waste_type, tonnes) or raise ValueError with a message for the user."""
    b = bounds(site)
    try:
        d = pd.Timestamp(day).normalize()
    except Exception:
        raise ValueError("Pick a valid date.")
    if pd.isna(d):
        raise ValueError("Pick a valid date.")
    try:
        t = float(tonnes)
    except (TypeError, ValueError):
        raise ValueError("Tonnes must be a number.")
    cap = b["max_week_t"] if kind == "week" else b["max_day_t"]
    if not (0 < t <= cap):
        raise ValueError(f"Tonnes must be above 0 and at most {cap:,.0f} (3x the largest {kind} seen at this site).")
    if d < pd.Timestamp(b["first"]):
        raise ValueError(f"This site's data starts on {b['first']}.")
    limit = b["week_max"] if kind == "week" else b["day_max"]
    if d > pd.Timestamp(limit):
        raise ValueError(f"Too far ahead. The latest date you can add here is {limit} (the forecast needs a continuous history).")
    if kind == "week":
        return wend(d).strftime("%Y-%m-%d"), "ALL", t
    if wtype not in b["types"]:
        raise ValueError("Unknown waste type for this site.")
    return d.strftime("%Y-%m-%d"), wtype, t


def build_site(site, days=(), weeks=None):
    """days = [(Timestamp, waste_type, kg)] extra tickets; weeks = {week_end: tonnes} weekly totals set by the user."""
    df, zeros = base()
    g, z, weeks = df[df.site == site], zeros[zeros.site == site], weeks or {}
    if len(days):
        extra = pd.DataFrame(list(days), columns=["date", "waste_type", "kg"])
        extra["site"] = site
        g = pd.concat([g, extra], ignore_index=True)
    daily = g.groupby("date").kg.sum()
    user_dates = {pd.Timestamp(d[0]).strftime("%Y-%m-%d") for d in days}
    touched = {wend(d[0]) for d in days} | set(weeks)
    wk, filled = weekly(daily, touched, weeks)
    fc, metrics, bt = forecast_site(wk)
    metrics["includes_user"] = bool(touched)
    an, n_an = anomalies(daily, z, user_dates)
    dow = (daily.groupby(daily.index.dayofweek).mean() / 1000).round(2).reindex(range(7)).fillna(0)
    gy = g.assign(y=g.date.dt.year, org=g.waste_type.isin(ORGANIC))
    yearly = [{"year": int(y), "total": float(x.kg.sum() / 1000), "organic": float(x[x.org].kg.sum() / 1000), "n": int(len(x))}
              for y, x in gy.groupby("y")]
    med_n = float(np.median([r["n"] for r in yearly]))
    for r in yearly:  # 'sparse' = nearly empty year (e.g. one user ticket); 'full' = a complete year of tickets
        r["sparse"], r["full"] = bool(r["n"] < 0.25 * med_n), bool(r["n"] >= 0.8 * med_n)
    hist = [{"week": d.strftime("%Y-%m-%d"), "t": float(v), "u": d in touched, "i": bool(filled.get(d, False))}
            for d, v in wk.dropna().items()]
    return {"id": site, "name": NAMES[site], "history": hist,
            "forecast": fc, "metrics": metrics, "backtest": bt, "anomalies": an, "anomaly_count": n_an,
            "weekday": dow.tolist(), "yearly": yearly, "has_types": bool(g.waste_type.nunique() > 1),
            "tickets": int(len(g)), "span": [g.date.min().strftime("%Y-%m-%d"), g.date.max().strftime("%Y-%m-%d")]}


def build():
    df, _ = base()
    res = {s: build_site(s) for s in sorted(df.site.unique())}
    return res, list(_BASE["audit"])


if __name__ == "__main__":
    r, a = build()
    for k, v in r.items():
        m = v["metrics"]
        print(f"{k:14s} MAE model {m['model_mae']:.1f}t vs base {m['base_mae']:.1f}t | MAPE {m['model_mape']:.1f}% vs {m['base_mape']:.1f}% | anomalies {v['anomaly_count']} | next wk {v['forecast'][0]['t']:.0f}t")
    print(*a, sep="\n")
'''

FILES['kitchen.py'] = r'''"""Database + analysis for the 'My Kitchen' waste log (SQLite)."""
import sqlite3, os, random, datetime as dt
import numpy as np, pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kitchen.db")
CATS = ["Vegetables", "Fruit", "Cooked meals", "Bread & bakery", "Dairy", "Meat & fish", "Other"]
REASONS = ["Spoiled before use", "Cooked too much", "Plate leftovers", "Expired date", "Bad storage", "Changed plans"]
PRICE = {"Vegetables": 2.0, "Fruit": 3.0, "Cooked meals": 6.0, "Bread & bakery": 3.5, "Dairy": 4.0, "Meat & fish": 9.0, "Other": 3.0}  # $/kg, rough


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init(seed=True):
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS waste_log(id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL,
                     item TEXT NOT NULL, category TEXT NOT NULL, kg REAL NOT NULL, reason TEXT NOT NULL, cost REAL NOT NULL, demo INTEGER DEFAULT 0)""")
        if seed and c.execute("SELECT COUNT(*) FROM waste_log").fetchone()[0] == 0:
            random.seed(3)
            items = {"Vegetables": ["Spinach", "Tomatoes", "Onions"], "Fruit": ["Bananas", "Apples"], "Cooked meals": ["Rice", "Dal", "Pasta"],
                     "Bread & bakery": ["Sliced bread"], "Dairy": ["Milk", "Yoghurt"], "Meat & fish": ["Chicken"], "Other": ["Leftover sauce"]}
            w = [3, 2, 4, 2, 1.5, 1, 1]
            today = dt.date.today()
            for d in range(90, 0, -1):
                day = today - dt.timedelta(days=d)
                n = random.choice([0, 1, 1, 2]) + (1 if day.weekday() in (5, 6) else 0)  # weekends waste more
                for _ in range(n):
                    cat = random.choices(CATS, w)[0]
                    kg = round(max(0.05, random.gauss(0.35, 0.2)) * (1 - d / 400), 2)  # slowly improving
                    if random.random() < 0.03: kg = round(kg * 6, 2)  # the odd disaster
                    reason = random.choices(REASONS, [3, 3, 2, 2, 2, 1])[0]
                    c.execute("INSERT INTO waste_log(day,item,category,kg,reason,cost,demo) VALUES(?,?,?,?,?,?,1)",
                              (day.isoformat(), random.choice(items[cat]), cat, kg, reason, round(kg * PRICE[cat], 2)))


def add(day, item, category, kg, reason):
    if category not in CATS or reason not in REASONS: raise ValueError("bad category/reason")
    kg = float(kg)
    if not 0 < kg < 100: raise ValueError("kg out of range")
    with conn() as c:
        c.execute("INSERT INTO waste_log(day,item,category,kg,reason,cost) VALUES(?,?,?,?,?,?)",
                  (day, item.strip()[:60] or "Unnamed", category, kg, reason, round(kg * PRICE[category], 2)))


def delete(i):
    with conn() as c: c.execute("DELETE FROM waste_log WHERE id=?", (i,))


def entries(limit=40):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM waste_log ORDER BY day DESC, id DESC LIMIT ?", (limit,))]


def summary():
    with conn() as c:
        df = pd.read_sql("SELECT * FROM waste_log", c, parse_dates=["day"])
    if df.empty: return {"empty": True}
    # detection: entry far above its own category's typical size (robust z on log kg)
    lk = np.log(df.kg)
    med, mad = lk.groupby(df.category).transform("median"), (lk - lk.groupby(df.category).transform("median")).abs().groupby(df.category).transform("median") * 1.4826
    z = (lk - med) / mad.replace(0, np.nan)
    big = df[z > 3].sort_values("kg", ascending=False).head(5)
    wk_all = df.set_index("day").kg.resample("W").sum()
    partial = len(wk_all) > 3 and wk_all.index[-1].date() >= dt.date.today()  # week still in progress
    wk = wk_all.iloc[:-1] if partial else wk_all   # trend and forecast use finished weeks only
    current = {"week": wk_all.index[-1].strftime("%Y-%m-%d"), "kg": float(wk_all.iloc[-1])} if partial else None
    x = np.arange(len(wk)); slope, icpt = np.polyfit(x, wk.values, 1) if len(wk) > 2 else (0, wk.mean())
    fc = [max(0.0, float(slope * (len(wk) + i) + icpt)) for i in range(4)]
    last4, prev4 = wk.tail(4).sum(), wk.iloc[-8:-4].sum() if len(wk) >= 8 else None
    by_reason = df.groupby("reason").kg.sum().sort_values(ascending=False)
    by_cat = df.groupby("category").agg(kg=("kg", "sum"), cost=("cost", "sum")).sort_values("kg", ascending=False)
    wd = df.groupby(df.day.dt.dayofweek).kg.sum().reindex(range(7)).fillna(0)
    tips = {"Spoiled before use": "Shop smaller and more often; move older produce to the front of the fridge.",
            "Cooked too much": "Cook by the cup, not by the pot. Freeze half of any big batch on day one.",
            "Plate leftovers": "Serve smaller first portions; leftovers become tomorrow's lunch.",
            "Expired date": "Check dates when you unpack groceries; 'best before' is not 'unsafe after'.",
            "Bad storage": "Store greens with paper towel, keep bread out of the fridge, bananas away from other fruit.",
            "Changed plans": "Plan three dinners, not seven. Keep one flexible freezer meal."}[by_reason.index[0]]
    return {"empty": False, "total_kg": float(df.kg.sum()), "total_cost": float(df.cost.sum()), "entries": int(len(df)),
            "weekly": [{"week": d.strftime("%Y-%m-%d"), "kg": float(v)} for d, v in wk.items()], "current": current,
            "forecast": fc, "trend_kg_per_week": float(slope),
            "change_vs_prev4": None if not prev4 else float((last4 - prev4) / prev4 * 100),
            "by_reason": {k: float(v) for k, v in by_reason.items()},
            "by_category": [{"category": k, "kg": float(r.kg), "cost": float(r.cost)} for k, r in by_cat.iterrows()],
            "weekday": wd.tolist(), "oversized": [{"day": r.day.strftime("%Y-%m-%d"), "item": r["item"], "kg": float(r.kg)} for _, r in big.iterrows()],
            "tip": tips, "top_reason": by_reason.index[0], "has_demo": bool(df.demo.any())}
'''

FILES['tickets.py'] = r'''"""Database for user-entered weighbridge data (SQLite). Original CSVs are never modified."""
import sqlite3, os

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.db")
MAX_PER_SITE = 200


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS user_tickets(id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT NOT NULL,
                     kind TEXT NOT NULL, day TEXT NOT NULL, waste_type TEXT NOT NULL, tonnes REAL NOT NULL)""")


def add(site, kind, day, waste_type, tonnes):
    with conn() as c:
        if c.execute("SELECT COUNT(*) FROM user_tickets WHERE site=?", (site,)).fetchone()[0] >= MAX_PER_SITE:
            raise ValueError(f"Limit of {MAX_PER_SITE} entries per site reached. Delete some or reset the site.")
        if kind == "week":  # one total per week: a new one replaces the old one
            c.execute("DELETE FROM user_tickets WHERE site=? AND kind='week' AND day=?", (site, day))
        c.execute("INSERT INTO user_tickets(site,kind,day,waste_type,tonnes) VALUES(?,?,?,?,?)",
                  (site, kind, day, waste_type, tonnes))


def delete(site, i):
    with conn() as c:
        c.execute("DELETE FROM user_tickets WHERE site=? AND id=?", (site, i))


def clear(site):
    with conn() as c:
        c.execute("DELETE FROM user_tickets WHERE site=?", (site,))


def rows(site):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM user_tickets WHERE site=? ORDER BY day, id", (site,))]


def counts():
    with conn() as c:
        return {r["site"]: r["n"] for r in c.execute("SELECT site, COUNT(*) n FROM user_tickets GROUP BY site")}
'''

FILES['app.py'] = r'''"""Tipping Floor API. Run: python app.py  ->  http://localhost:5000"""
import threading
import pandas as pd
from flask import Flask, jsonify, send_from_directory, abort, request
import pipeline, kitchen, tickets

app = Flask(__name__, static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def no_cache(r):  # always serve the newest page and data, never a stale copy
    r.headers["Cache-Control"] = "no-store"
    return r

LOCK = threading.Lock()  # one retrain at a time
print("training models on the weighbridge tickets...")
ORIGINAL, AUDIT = pipeline.build()   # untouched results, kept so 'reset' and 'before vs after' are instant
DATA = dict(ORIGINAL)                # what the dashboard shows (original + user entries)
kitchen.init()
tickets.init()


def recompute(site):
    rows = tickets.rows(site)
    days = [(pd.Timestamp(r["day"]), r["waste_type"], r["tonnes"] * 1000.0) for r in rows if r["kind"] == "day"]
    weeks = {pipeline.wend(r["day"]): r["tonnes"] for r in rows if r["kind"] == "week"}
    DATA[site] = pipeline.build_site(site, days, weeks) if rows else ORIGINAL[site]


for _s in tickets.counts():          # entries saved in an earlier session
    recompute(_s)
print("ready.")


def view(sid):
    d = dict(DATA[sid])
    d["user"] = tickets.rows(sid)
    d["bounds"] = pipeline.bounds(sid)
    d["baseline"] = {"forecast": ORIGINAL[sid]["forecast"], "metrics": ORIGINAL[sid]["metrics"]}
    return d


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/sites")
def sites():
    n = tickets.counts()
    return jsonify([{"id": k, "name": v["name"], "tickets": v["tickets"], "span": v["span"],
                     "has_types": v["has_types"], "user_entries": n.get(k, 0)} for k, v in DATA.items()])


@app.get("/api/site/<sid>")
def site(sid):
    if sid not in DATA:
        abort(404)
    return jsonify(view(sid))


@app.post("/api/site/<sid>/ticket")
def add_ticket(sid):
    if sid not in DATA:
        abort(404)
    j = request.get_json(force=True, silent=True) or {}
    kind = j.get("kind", "day")
    try:
        if kind not in ("day", "week"):
            raise ValueError("Unknown entry type.")
        day, wtype, t = pipeline.validate(sid, kind, j.get("day"), j.get("waste_type"), j.get("tonnes"))
        with LOCK:
            tickets.add(sid, kind, day, wtype, t)
            recompute(sid)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(view(sid))


@app.delete("/api/site/<sid>/ticket/<int:i>")
def del_ticket(sid, i):
    if sid not in DATA:
        abort(404)
    with LOCK:
        tickets.delete(sid, i)
        recompute(sid)
    return jsonify(view(sid))


@app.delete("/api/site/<sid>/tickets")
def reset_site(sid):
    if sid not in DATA:
        abort(404)
    with LOCK:
        tickets.clear(sid)
        recompute(sid)
    return jsonify(view(sid))


@app.get("/api/audit")
def audit():
    return jsonify(AUDIT)


@app.get("/api/kitchen")
def k_summary():
    return jsonify({"summary": kitchen.summary(), "entries": kitchen.entries(), "cats": kitchen.CATS, "reasons": kitchen.REASONS})


@app.post("/api/kitchen")
def k_add():
    j = request.get_json(force=True)
    try:
        kitchen.add(j["day"], j["item"], j["category"], j["kg"], j["reason"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return k_summary()


@app.delete("/api/kitchen/<int:i>")
def k_del(i):
    kitchen.delete(i)
    return k_summary()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
'''

FILES['static/index.html'] = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tipping Floor</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--paper:#ece4d0;--slip:#f7f1e1;--ink:#1e1c17;--mute:#7a7260;--stamp:#b3321f;--leaf:#2f5d3a;--rule:#cfc4a8}
@media(prefers-color-scheme:dark){:root{--paper:#1b1913;--slip:#24211a;--ink:#ece4d0;--mute:#a69d86;--stamp:#e06a52;--leaf:#7fb58a;--rule:#3d382b}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 'DM Mono',ui-monospace,monospace;padding:28px 18px 80px}
.wrap{max-width:980px;margin:0 auto}
h1{font:800 clamp(40px,8vw,84px)/.9 Fraunces,Georgia,serif;margin:0;letter-spacing:-.03em}
h1 em{color:var(--stamp);font-style:normal}
h2{font:500 22px Fraunces,Georgia,serif;margin:0 0 4px}
.sub{color:var(--mute);max-width:56ch;margin:10px 0 26px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
.tabs button{font:inherit;background:none;border:1.5px dashed var(--mute);color:var(--ink);padding:5px 12px;cursor:pointer;transform:rotate(var(--r,0deg))}
.tabs button.on{background:var(--ink);color:var(--paper);border-style:solid}
.slip{background:var(--slip);padding:20px 22px;margin-bottom:18px;position:relative;border-bottom:3px double var(--rule);box-shadow:0 1px 0 var(--rule)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.k{color:var(--mute);font-size:11px;letter-spacing:.12em;text-transform:uppercase}
.big{font:800 40px/1 Fraunces,serif}
svg{width:100%;height:auto;display:block;overflow:visible}
.stamp{display:inline-block;border:2px solid var(--stamp);color:var(--stamp);padding:1px 8px;font-weight:500;text-transform:uppercase;letter-spacing:.1em;font-size:12px;transform:rotate(-3deg)}
.stamp.ok{border-color:var(--leaf);color:var(--leaf)}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:4px 0;border-bottom:1px dotted var(--rule)}td:last-child{text-align:right}
label{display:block;margin:10px 0 2px}
input[type=range]{width:100%;accent-color:var(--stamp)}
.note{color:var(--mute);font-size:12px;margin-top:8px}
ul{padding-left:18px;margin:6px 0}li{margin:4px 0}
.err{color:var(--stamp);font-size:12px}
input,select{font:inherit;background:var(--paper);color:var(--ink);border:1.5px solid var(--rule);padding:4px 6px}
.chip{font:inherit;font-size:12px;background:none;border:1.5px dashed var(--mute);color:var(--ink);padding:3px 9px;margin:3px 6px 3px 0;cursor:pointer}
.chip:hover{border-style:solid;border-color:var(--stamp);color:var(--stamp)}
.hl{background:linear-gradient(transparent 55%,color-mix(in srgb,var(--stamp) 30%,transparent) 0)}
</style></head><body><div class="wrap">
<div class="k" id="tix"></div>
<h1>Tipping<br><em>Floor</em></h1>
<p class="sub">Every truck that crosses a council weighbridge leaves a ticket. This reads ~25,000 of them, forecasts next week's tonnage, flags tickets that look wrong, and lets you test how much a food-waste cut would save. Pick a site and add your own tickets: everything recomputes.</p>
<div class="tabs" id="tabs"></div>
<div id="main"></div>
<div class="slip"><h2>Things wrong with the data</h2><div class="note">Found while cleaning. Not hidden.</div><ul id="audit"></ul></div>
</div>
<script>
const $=s=>document.querySelector(s),fmt=n=>Math.round(n).toLocaleString();
const MON=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
let sites=[],cur=null;
document.getElementById('tix').textContent='TICKET #'+String(Math.floor(Math.random()*90000+10000))+' / WEIGHBRIDGE COPY · v2 INTERACTIVE';
async function init(){
  sites=await (await fetch('/api/sites')).json();
  drawTabs();
  $('#tabs').onclick=e=>e.target.dataset.id&&load(e.target.dataset.id);
  $('#audit').innerHTML=(await (await fetch('/api/audit')).json()).map(a=>`<li>${a}</li>`).join('');
  load('kitchen');
}
let zoomW=104,active='kitchen',form={kind:'day',day:'',tonnes:'',wt:''},fErr='';
function drawTabs(){
  $('#tabs').innerHTML='<button data-id="kitchen" style="--r:-1deg">★ My Kitchen</button>'+sites.map((s,i)=>`<button data-id="${s.id}" style="--r:${(i%3-1)*.8}deg">${s.name}${s.user_entries?' ●':''}</button>`).join('');
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.id==active));
}
async function load(id){
  active=id;drawTabs();
  if(id=='kitchen'){return loadKitchen();}
  cur=await (await fetch('/api/site/'+id)).json();
  form={kind:'day',day:'',tonnes:'',wt:''};fErr='';zoomW=cur.user.length?26:104;render();
}
function setZoom(n){zoomW=n;render();}
// ---- user data entry on a site tab: send it, the server retrains that site, redraw everything ----
function setKind(k){readForm();form.kind=k;form.day='';form.tonnes='';render();}
function readForm(){const g=i=>document.getElementById(i);if(g('e-day')){form.day=g('e-day').value;form.tonnes=g('e-t').value;if(g('e-type'))form.wt=g('e-type').value;}}
function prefill(kind,mult){const b=cur.bounds;form.kind=kind;fErr='';
  if(kind=='day'){const d=new Date(b.last);d.setDate(d.getDate()+1);form.day=d.toISOString().slice(0,10);form.tonnes=(b.typical_day_t*mult).toFixed(1);}
  else{const d=new Date(b.last);d.setDate(d.getDate()+7);form.day=d.toISOString().slice(0,10);form.tonnes=(b.typical_week_t*mult).toFixed(0);}
  render();}
async function addTicket(){
  readForm();fErr='';
  const r=await fetch('/api/site/'+cur.id+'/ticket',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:form.kind,day:form.day,tonnes:form.tonnes,waste_type:form.wt||cur.bounds.types[0]})});
  const j=await r.json();
  if(j.error){fErr=j.error;render();return;}
  cur=j;form.tonnes='';zoomW=26;syncTab();render();
}
async function delTicket(i){cur=await (await fetch('/api/site/'+cur.id+'/ticket/'+i,{method:'DELETE'})).json();syncTab();render();}
async function resetSite(){cur=await (await fetch('/api/site/'+cur.id+'/tickets',{method:'DELETE'})).json();syncTab();render();}
function syncTab(){const s=sites.find(x=>x.id==cur.id);if(s){s.user_entries=cur.user.length;drawTabs();}}
function entryPanel(d){
  const b=d.bounds,w=form.kind=='week',day=form.day||(()=>{const x=new Date(b.last);x.setDate(x.getDate()+(w?7:1));return x.toISOString().slice(0,10)})();
  const cap=w?b.max_week_t:b.max_day_t,ty=w?b.typical_week_t:b.typical_day_t;
  const types=d.has_types?`<label>Waste type<select id="e-type" style="width:100%">${b.types.map(t=>`<option ${t==form.wt?'selected':''}>${t}</option>`).join('')}</select></label>`:'';
  const list=d.user.length?`<h2 style="margin-top:16px">Your entries (${d.user.length})</h2><table>${d.user.map(e=>`<tr><td>${e.day}</td><td>${e.kind=='week'?'week total':e.waste_type}</td><td>${e.tonnes} t</td><td><a href="#" onclick="delTicket(${e.id});return false" style="color:var(--stamp)">x</a></td></tr>`).join('')}</table><a href="#" onclick="resetSite();return false" style="color:var(--stamp)">reset ${d.name} to the original data</a>`:'';
  return `<div class="slip"><h2>Add your own data</h2><div class="note">Stamp a ticket and this site's forecast, anomaly flags and charts are recomputed on the spot. Saved in SQLite here; the original CSVs are never touched.</div>
  <div class="row" style="margin-top:10px"><div>
    <label><input type="radio" name="kind" ${!w?'checked':''} onchange="setKind('day')"> One weighbridge ticket <span class="note">(adds to that day)</span></label>
    <label><input type="radio" name="kind" ${w?'checked':''} onchange="setKind('week')"> A whole week's total <span class="note">(sets that week)</span></label>
    <label>${w?'Any day in the week':'Date'}<input id="e-day" type="date" min="${b.first}" max="${w?b.week_max:b.day_max}" value="${day}" style="width:100%"></label>
    <label>Tonnes <span class="note">typical ${w?'week':'day'}: ${fmt(ty)} t · max ${fmt(cap)}</span><input id="e-t" type="number" step="0.1" min="0.1" value="${form.tonnes}" placeholder="e.g. ${fmt(ty)}" style="width:100%"></label>
    ${w?'':types}
    <p><button class="stamp" style="background:none;cursor:pointer;font:inherit" onclick="addTicket()">Stamp it</button></p>
    <div class="err">${fErr}</div>
    <div class="note">Dates run from ${b.first} to ${w?b.week_max:b.day_max}. Weeks you touch count as complete, so a lone ticket in a new week makes that week look small.</div></div>
  <div><div class="k">Try a what-if</div><p class="note">Fills the form. Press Stamp it to send.</p>
    <button class="chip" onclick="prefill('day',4)">spike day: 4× normal</button><button class="chip" onclick="prefill('day',.1)">near-empty day</button>
    <button class="chip" onclick="prefill('week',1.4)">heavy week: +40%</button><button class="chip" onclick="prefill('week',.6)">quiet week: −40%</button>
    ${list}</div></div></div>`;
}
let K=null;
async function loadKitchen(r){K=r||await (await fetch('/api/kitchen')).json();renderKitchen();}
async function addEntry(){
  const b={day:$('#f-day').value,item:$('#f-item').value,category:$('#f-cat').value,kg:$('#f-kg').value,reason:$('#f-reason').value};
  const r=await fetch('/api/kitchen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json();if(j.error){alert(j.error);return;}loadKitchen(j);
}
async function delEntry(i){loadKitchen(await (await fetch('/api/kitchen/'+i,{method:'DELETE'})).json());}
function renderKitchen(){
  const s=K.summary,opt=a=>a.map(x=>`<option>${x}</option>`).join(''),today=new Date(Date.now()-new Date().getTimezoneOffset()*60000).toISOString().slice(0,10);
  let h=`<div class="slip"><h2>Log what you threw out</h2><div class="note">Weigh it if you can, guess if you can't. Entries go into a SQLite database on this machine.</div>
  <div class="row" style="margin-top:10px"><div><label>Day<input id="f-day" type="date" value="${today}" style="width:100%"></label><label>What<input id="f-item" placeholder="e.g. rice" style="width:100%"></label><label>Kilograms<input id="f-kg" type="number" step="0.05" min="0.05" value="0.3" style="width:100%"></label></div>
  <div><label>Category<select id="f-cat" style="width:100%">${opt(K.cats)}</select></label><label>Why<select id="f-reason" style="width:100%">${opt(K.reasons)}</select></label><br><button class="stamp" style="background:none;cursor:pointer;font:inherit" onclick="addEntry()">Stamp it</button></div></div></div>`;
  if(!s.empty){
    const ch=s.change_vs_prev4,trend=s.trend_kg_per_week;
    h+=`<div class="slip"><div class="row"><div><div class="k">${s.entries} entries${s.has_demo?' · includes demo data':''}</div><div class="big">${s.total_kg.toFixed(1)} kg</div><div class="k">≈ $${s.total_cost.toFixed(0)} thrown away</div></div>
    <div><div class="k">last 4 weeks vs the 4 before</div><div class="big" style="color:${ch>0?'var(--stamp)':'var(--leaf)'}">${ch==null?'–':(ch>0?'+':'')+ch.toFixed(0)+'%'}</div><div class="k">trend ${trend>0?'up':'down'} ${Math.abs(trend).toFixed(2)} kg per week</div></div></div>
    <p class="note">Next 4 weeks if nothing changes: ${s.forecast.map(v=>v.toFixed(1)+' kg').join(' · ')}</p>${wkchart(s)}</div>`;
    const mr=Math.max(...Object.values(s.by_reason));
    h+=`<div class="row"><div class="slip"><h2>Why it happens</h2>${Object.entries(s.by_reason).map(([k,v])=>`<div style="margin:6px 0">${k} <span class="note">${v.toFixed(1)} kg</span><div style="height:8px;background:${k==s.top_reason?'var(--stamp)':'var(--ink)'};width:${v/mr*100}%"></div></div>`).join('')}<p class="hl">${s.tip}</p></div>
    <div class="slip"><h2>What it costs</h2><table>${s.by_category.map(c=>`<tr><td>${c.category}</td><td>${c.kg.toFixed(1)} kg</td><td>$${c.cost.toFixed(0)}</td></tr>`).join('')}</table>
    <div class="note" style="margin-top:10px">Prices are rough $/kg guesses.</div>
    ${s.oversized.length?`<h2 style="margin-top:14px">Unusually big dumps</h2><table>${s.oversized.map(o=>`<tr><td>${o.day}</td><td>${o.item}</td><td>${o.kg} kg</td></tr>`).join('')}</table>`:''}</div></div>`;
    h+=`<div class="slip"><h2>Your log</h2><table>${K.entries.slice(0,15).map(e=>`<tr><td>${e.day}</td><td>${e.item}</td><td>${e.kg} kg · ${e.reason}</td><td><a href="#" onclick="delEntry(${e.id});return false" style="color:var(--stamp)">x</a></td></tr>`).join('')}</table></div>`;
  }
  $('#main').innerHTML=h;
}
function wkchart(s){
  const a=s.weekly.slice(-20),cur=s.current,f=s.forecast;
  const it=[...a.map(w=>({l:w.week.slice(5),kg:w.kg,t:0})),...(cur?[{l:cur.week.slice(5),kg:cur.kg,t:1}]:[]),...f.map((k,i)=>({l:'+'+(i+1)+'wk',kg:k,t:2}))];
  const W=900,H=250,p=36,top=34,bw=(W-p-10)/it.length,mx=Math.max(...it.map(b=>b.kg))*1.18||1,y=v=>H-30-v/mx*(H-30-top);
  const col=['var(--ink)','var(--leaf)','var(--stamp)'];
  const grid=[0,.5,1].map(t=>`<line x1="${p}" x2="${W}" y1="${y(mx/1.18*t)}" y2="${y(mx/1.18*t)}" stroke="var(--rule)" stroke-dasharray="2 4"/><text x="0" y="${y(mx/1.18*t)+4}" font-size="10" fill="var(--mute)">${(mx/1.18*t).toFixed(1)}</text>`).join('');
  const bars=it.map((b,i)=>{const x=p+i*bw+bw*.12,w=bw*.76;return `<rect x="${x}" y="${y(b.kg)}" width="${w}" height="${H-30-y(b.kg)}" fill="${col[b.t]}" opacity="${b.t==2?.55:1}"/><text x="${x+w/2}" y="${y(b.kg)-4}" font-size="9" text-anchor="middle" fill="var(--ink)">${b.kg.toFixed(1)}</text>${i%2==0||b.t?`<text x="${x+w/2}" y="${H-14}" font-size="9" text-anchor="middle" fill="var(--mute)">${b.l}</text>`:''}`}).join('');
  return `<svg viewBox="0 0 ${W} ${H}">${grid}${bars}<text x="${p}" y="14" font-size="11" fill="var(--mute)">kg thrown out per week (label = week ending)</text><text x="${p}" y="27" font-size="11" fill="var(--mute)">black = logged · <tspan fill="var(--leaf)">green = this week so far</tspan> · <tspan fill="var(--stamp)">red = forecast</tspan></text></svg>`;}
function chart(d){
  const H=d.history.slice(-zoomW),F=d.forecast,all=H.concat(F),W=900,Ht=300,p=34;
  const B=d.user.length?d.baseline.forecast:[],pos={};all.forEach((a,i)=>pos[a.week]=i);
  const mx=Math.max(...all.map(a=>a.hi||a.t),...B.map(a=>a.t))*1.05,n=all.length;
  const x=i=>p+i*(W-p*2)/(n-1),y=v=>Ht-p-v/mx*(Ht-p*2);
  const line=H.map((a,i)=>(i?'L':'M')+x(i)+','+y(a.t)).join('');
  const f0=H.length-1,fl=F.map((a,i)=>'L'+x(f0+1+i)+','+y(a.t)).join('');
  const band=F.map((a,i)=>(i?'L':'M')+x(f0+1+i)+','+y(a.hi)).join('')+F.slice().reverse().map((a,i)=>'L'+x(n-1-i)+','+y(a.lo)).join('')+'Z';
  const grid=[0,.5,1].map(t=>`<line x1="${p}" x2="${W-p}" y1="${y(mx*t/1.05)}" y2="${y(mx*t/1.05)}" stroke="var(--rule)" stroke-dasharray="2 4"/><text x="0" y="${y(mx*t/1.05)+4}" font-size="11" fill="var(--mute)">${fmt(mx*t/1.05)}t</text>`).join('');
  const bp=B.filter(a=>a.week in pos),old=bp.length?`<path d="${bp.map((a,i)=>(i?'L':'M')+x(pos[a.week])+','+y(a.t)).join('')}" fill="none" stroke="var(--mute)" stroke-width="1.6" stroke-dasharray="2 4"/><text x="${p}" y="${p-12}" font-size="11" fill="var(--mute)">dotted grey = forecast before your entries</text>`:'';
  const dots=H.map((a,i)=>a.u?`<circle cx="${x(i)}" cy="${y(a.t)}" r="7" fill="var(--stamp)" stroke="var(--ink)" stroke-width="2"/>`:a.i?`<circle cx="${x(i)}" cy="${y(a.t)}" r="3" fill="var(--slip)" stroke="var(--mute)"/>`:'').join('');
  return `<svg viewBox="0 0 ${W} ${Ht}">${grid}<path d="${band}" fill="var(--stamp)" opacity=".18"/><path d="${line}" fill="none" stroke="var(--ink)" stroke-width="1.6"/><path d="M${x(f0)},${y(H[f0].t)}${fl}" fill="none" stroke="var(--stamp)" stroke-width="2.4" stroke-dasharray="6 4"/>${old}${dots}<text x="${x(f0+1)}" y="${p-12}" font-size="12" fill="var(--stamp)">next 8 weeks →</text><text x="${p}" y="${Ht-6}" font-size="11" fill="var(--mute)">${H[0].week}</text><text x="${W-p}" y="${Ht-6}" font-size="11" text-anchor="end" fill="var(--mute)">${F[n-H.length-1].week}</text></svg>`;
}
function fullYear(d){return d.yearly.filter(y=>y.full).slice(-1)[0]||d.yearly.slice(-1)[0];}
function impact(d,nx){
  if(!d.user.length)return '';
  const b=d.baseline.forecast[0].t,ch=(nx.t-b)/b*100;
  return `<p class="hl" style="margin:8px 0 0">Your ${d.user.length} entr${d.user.length>1?'ies':'y'} moved next week's forecast from <b>${fmt(b)} t</b> to <b>${fmt(nx.t)} t</b> (${ch>0?'+':''}${ch.toFixed(1)}%).</p>`;
}
function bars(vals,labels,color){const m=Math.max(...vals)||1;return `<svg viewBox="0 0 ${vals.length*60} 130">`+vals.map((v,i)=>`<rect x="${i*60+8}" y="${100-v/m*90}" width="40" height="${v/m*90}" fill="${color(i,v,m)}"/><text x="${i*60+28}" y="118" font-size="11" text-anchor="middle" fill="var(--mute)">${labels[i]}</text><text x="${i*60+28}" y="${94-v/m*90}" font-size="10" text-anchor="middle" fill="var(--ink)">${v>=100?Math.round(v):v.toFixed(1)}</text>`).join('')+'</svg>'}
function render(){
  const d=cur,m=d.metrics,nx=d.forecast[0];
  const full=fullYear(d),peak=d.weekday.indexOf(Math.max(...d.weekday));
  const gain=((1-m.model_mae/m.base_mae)*100);
  let h=`<div class="slip"><div class="row"><div><div class="k">${d.name} · ${d.span[0]} → ${d.span[1]} · ${fmt(d.tickets)} tickets</div><div class="big">${fmt(nx.t)} t</div><div class="k">expected next week (${fmt(nx.lo)}–${fmt(nx.hi)}, 80% band)</div>${impact(d,nx)}</div>
  <div><span class="stamp ${m.beats_baseline?'ok':''}">${m.beats_baseline?'beats naive':'naive wins'}</span><p class="note">Tested on the last ${m.weeks} weeks it never saw. Model error ${m.model_mape.toFixed(1)}% vs ${m.base_mape.toFixed(1)}% for "same as the last 4 weeks"${m.beats_baseline?`, about ${gain.toFixed(0)}% less absolute error`:''}. ${m.beats_baseline?'':'Use it with care here.'}${m.includes_user?' <b>Includes your entries.</b>':''}</p></div></div>${chart(d)}<div class="note">zoom: ${[[26,'6 months'],[104,'2 years'],[9999,'all']].map(([n,l])=>`<a href="#" onclick="setZoom(${n});return false" style="color:${zoomW==n?'var(--stamp)':'var(--ink)'}">${l}</a>`).join(' · ')}</div>${d.user.length?'<div class="note">● red dot = week you changed</div>':''}</div>${entryPanel(d)}`;
  h+=`<div class="row"><div class="slip"><h2>Which day is the heavy one?</h2><div class="note">Mean tonnes per ticket-day. Peak: <span class="hl">${MON[peak]}</span> — plan crews around it.</div>${bars(d.weekday,MON,(i)=>i==peak?'var(--stamp)':'var(--ink)')}</div>
  <div class="slip"><h2>Tickets that look off</h2><div class="note">${d.anomaly_count} flagged vs the previous 10 same-weekday tickets (robust z &gt; 5). Worst:</div><table>${d.anomalies.slice(0,7).map(a=>`<tr><td>${a.date}</td><td>${a.kind}${a.z?' '+Math.abs(a.z).toFixed(0)+'σ':''}${a.user?' <span class="stamp" style="font-size:10px">yours</span>':''}</td><td>${a.tonnes.toFixed(1)}t${a.typical?` <span class="note">vs ${a.typical.toFixed(0)}</span>`:''}</td></tr>`).join('')}</table></div></div>`;
  if(d.has_types){const ys=d.yearly.filter(y=>y.total>0&&!y.sparse);
    h+=`<div class="slip"><h2>How much of it is organic?</h2><div class="note">Share of weighed tonnes labelled Biodegradable / Sorted Organic. Rising share = separation at source is working.</div>${bars(ys.map(y=>y.organic/y.total*100),ys.map(y=>y.year),()=>'var(--leaf)')}<div class="note">% by year</div></div>
    <div class="slip"><h2>What if we cut food waste?</h2><div class="note">Baseline: ${full.year} (last full year). Sliders are <b>your assumptions</b>, not measured facts.</div><div class="row"><div>
    <label>Food share of the total stream: <b id="v1"></b></label><input id="s1" type="range" min="10" max="70" value="40">
    <label>Prevented at source (never wasted): <b id="v2"></b></label><input id="s2" type="range" min="0" max="50" value="15">
    <label>Diverted to compost / biogas: <b id="v3"></b></label><input id="s3" type="range" min="0" max="80" value="30">
    <label>Landfill cost per tonne: <b id="v4"></b></label><input id="s4" type="range" min="5" max="120" value="30"></div><div id="sim"></div></div></div>`;
  }else h+=`<div class="slip"><h2>No waste-type breakdown here</h2><div class="note">This city's open data has one total per ticket, so food-waste analysis isn't possible. Pick a Sri Lankan council to try the reduction simulator.</div></div>`;
  $('#main').innerHTML=h;
  if(d.has_types){['s1','s2','s3','s4'].forEach(i=>$('#'+i).oninput=sim);sim();}
}
function sim(){
  const d=cur,full=fullYear(d);
  const [a,b,c,e]=['s1','s2','s3','s4'].map(i=>+$('#'+i).value);
  $('#v1').textContent=a+'%';$('#v2').textContent=b+'%';$('#v3').textContent=c+'%';$('#v4').textContent='$'+e;
  const food=full.total*a/100,prev=food*b/100,div=(food-prev)*c/100,out=prev+div;
  const co2=out*0.7; // ~0.7 tCO2e per tonne food waste kept out of landfill (rough literature-range assumption)
  $('#sim').innerHTML=`<div class="k">food waste in the stream</div><div class="big">${fmt(food)} t</div><br><div class="k">kept out of landfill</div><div class="big" style="color:var(--leaf)">${fmt(out)} t</div><p>≈ <b>${fmt(co2)}</b> t CO₂e avoided · <b>$${fmt(out*e)}</b> saved on tipping fees per year</p><div class="note">0.7 tCO₂e/t is a rough assumption; real value depends on landfill gas capture. Treatment costs are not subtracted.</div>`;
}
init();
</script></body></html>
'''

for name, text in FILES.items():          # write the project to disk (open these to read / edit the code)
    open(os.path.join(ROOT, name), "w", encoding="utf-8").write(text)

# ---- data upload ----
if not glob.glob(ROOT + "/data/*.csv"):
    if IN_COLAB:
        from google.colab import files
        print("Upload archive__1_.zip (or the 6 CSV files):")
        for n in files.upload():
            if n.endswith(".zip"): zipfile.ZipFile(n).extractall(ROOT + "/data")
            elif n.endswith(".csv"): os.replace(n, ROOT + "/data/" + n)
    else:  # running on your own computer: look for the zip / CSVs next to where you ran the script
        for z in glob.glob("archive*.zip"): zipfile.ZipFile(z).extractall(ROOT + "/data")
        for c in glob.glob("*.csv"): os.replace(c, ROOT + "/data/" + c)
# zip may contain a sub-folder: flatten
for f in glob.glob(ROOT + "/data/**/*.csv", recursive=True):
    if os.path.dirname(f) != ROOT + "/data": os.replace(f, ROOT + "/data/" + os.path.basename(f))
assert glob.glob(ROOT + "/data/*.csv"), "No CSV files found - put archive__1_.zip next to this script (or upload it in Colab) and run again."

# ---- start the backend ----
sys.path.insert(0, ROOT); os.chdir(ROOT)
for _m in ("app", "pipeline", "kitchen", "tickets"):   # re-running the cell must use the NEW code, not a cached copy
    sys.modules.pop(_m, None)
import app as tf_app

if IN_COLAB:
    from werkzeug.serving import make_server
    _old = globals().get("_TF_SERVER")                 # stop the server from the previous run of this cell
    if _old is not None:
        try: _old.shutdown(); _old.server_close()
        except Exception: pass
    import socket
    def _free(port):                                   # werkzeug exits the program on a busy port, so check first
        s = socket.socket()
        try: s.bind(("0.0.0.0", port)); return True
        except OSError: return False
        finally: s.close()
    PORT = next((q for q in range(5000, 5020) if _free(q)), None)   # an OLD copy of the app may still hold 5000
    if PORT: _TF_SERVER = make_server("0.0.0.0", PORT, tf_app.app, threaded=True)
    assert PORT, "No free port. Use Runtime > Restart session, then run this cell again."
    threading.Thread(target=_TF_SERVER.serve_forever, daemon=True).start()
    time.sleep(1)
    from google.colab import output
    output.serve_kernel_port_as_iframe(PORT, height=900)   # shown inside the notebook
    print(f"Running on port {PORT}. You should see 'v2 INTERACTIVE' at the top of the page.")
    print("If the box above is blank, open the app in its own tab with the link below:")
    output.serve_kernel_port_as_window(PORT)

    # ---- public link: a temporary https address you can open in ANY browser or phone (needs internet in Colab) ----
    import subprocess, re, urllib.request
    def public_link(port, wait=45):
        exe = os.path.join(ROOT, "cloudflared")
        old = globals().get("_TF_TUNNEL")
        if old is not None:
            try: old.kill()
            except Exception: pass
        if not os.path.exists(exe):
            urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
            os.chmod(exe, 0o755)
        proc = subprocess.Popen([exe, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        found = []
        def drain():                                   # keep reading so the pipe never fills up
            for line in proc.stdout:
                m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
                if m and not found: found.append(m.group(0))
        threading.Thread(target=drain, daemon=True).start()
        t0 = time.time()
        while not found and time.time() - t0 < wait and proc.poll() is None: time.sleep(0.5)
        globals()["_TF_TUNNEL"] = proc
        return found[0] if found else None
    try:
        link = public_link(PORT)
        print("\nPUBLIC LINK (open in any browser):", link if link else "could not be created, use the link above instead")
        if link: print("It works only while this Colab session is running, and anyone with the link can use the app and add data.")
    except Exception as e:
        print("Could not create a public link:", e, "\nUse the link above instead.")
else:
    print("\nOpen  http://127.0.0.1:5000  in your browser. Press Ctrl+C here to stop.")
    tf_app.app.run(host="127.0.0.1", port=5000, use_reloader=False)   # keeps running until you stop it
