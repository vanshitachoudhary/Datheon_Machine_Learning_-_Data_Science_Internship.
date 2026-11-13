import streamlit as st, joblib, os
import numpy as np, pandas as pd

AUTHOR = ""   # optional: put your name here to show "Built by ..." in the footer

st.set_page_config(page_title="California Home Value Estimator", page_icon="🏡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&display=swap');
:root { --green:#24473f; --brass:#a98b4e; --line:#e2d8bd; --ink:#1d2a27; --muted:#5d6b66; --panel:#fffdf8; }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.topbar { display:flex; justify-content:space-between; align-items:center; padding-bottom:.9rem; border-bottom:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:.6rem; font-family:'Cormorant Garamond',Georgia,serif; font-size:1.5rem; font-weight:600; color:var(--green); }
.mark { display:inline-flex; width:1.9rem; height:1.9rem; border:1px solid var(--brass); border-radius:50%; align-items:center; justify-content:center; font-size:1.05rem; color:var(--brass); }
.where { color:var(--muted); font-size:.9rem; font-style:italic; }
.hero { padding:2.4rem 0 1.6rem; max-width:720px; }
.hero-title { font-family:'Cormorant Garamond',Georgia,serif; font-size:3.2rem; font-weight:600; line-height:1.08; color:var(--green); }
.hero-sub { color:var(--muted); font-size:1.08rem; line-height:1.65; margin-top:.8rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--panel); border:1px solid var(--line) !important; border-radius:4px; }
.card-title { font-family:'Cormorant Garamond',Georgia,serif; font-size:1.45rem; font-weight:600; color:var(--green); margin:.1rem 0 .2rem; }
[data-testid="stWidgetLabel"] p { font-size:.92rem; color:#3b4a45; }
.estimate { background:var(--panel); border:1px solid var(--brass); outline:1px solid var(--line); outline-offset:5px; padding:1.7rem 1.5rem 1.4rem; text-align:center; margin:.4rem 0 1.6rem; }
.estimate .label { font-family:'Cormorant Garamond',Georgia,serif; font-style:italic; font-size:1.2rem; color:#7a6a43; }
.estimate .figure { font-family:'Cormorant Garamond',Georgia,serif; font-size:4rem; font-weight:700; color:var(--green); line-height:1.1; font-variant-numeric:lining-nums; }
.estimate .range { color:var(--ink); font-size:1.02rem; margin-top:.2rem; }
.estimate .note { color:var(--muted); font-size:.85rem; margin-top:.3rem; }
.pbar { position:relative; height:6px; background:#ece4cf; border-radius:3px; margin:1.3rem auto .5rem; max-width:420px; }
.pbar i { position:absolute; top:-5px; width:3px; height:16px; background:var(--green); border-radius:2px; }
.ptext { color:var(--muted); font-size:.9rem; }
.sect { font-family:'Cormorant Garamond',Georgia,serif; font-size:1.45rem; font-weight:600; color:var(--green); margin:1.4rem 0 .1rem; }
.sect-sub { color:var(--muted); font-size:.9rem; margin-bottom:.6rem; }
.frow { display:grid; grid-template-columns:120px 1fr 96px; align-items:center; gap:.7rem; padding:.42rem 0; border-bottom:1px solid #efe8d5; font-size:.95rem; }
.ftrack { position:relative; height:10px; }
.ftrack:before { content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:#cfc4a4; }
.fbar { position:absolute; top:1px; height:8px; border-radius:2px; }
.fpos { background:#2f6b4f; left:50%; } .fneg { background:#8a3b3b; right:50%; }
.fval { text-align:right; font-variant-numeric:lining-nums; }
.disc { color:var(--muted); font-size:.82rem; line-height:1.6; margin-top:2rem; padding-top:1rem; border-top:1px solid var(--line); }
</style>
<div class="topbar"><div class="brand"><span class="mark">&#8962;</span>Home Value Estimator</div>
<div class="where">California census districts</div></div>
<div class="hero"><div class="hero-title">What is a California home worth?</div>
<div class="hero-sub">Describe the area and the homes around it. The estimate updates as you change each detail.</div></div>
""", unsafe_allow_html=True)

if not os.path.exists("models/housing_model.joblib"):
    st.warning("The model file is missing. Run the notebook first to create models/housing_model.joblib.")
    st.stop()
h = joblib.load("models/housing_model.joblib")
rng, cats = h["ranges"], h["cat_options"]

def engineer(d):
    d = dict(d)
    d["rooms_per_household"] = d["total_rooms"] / max(d["households"], 1)
    d["bedrooms_ratio"] = d["total_bedrooms"] / max(d["total_rooms"], 1)
    d["population_per_household"] = d["population"] / max(d["households"], 1)
    d["income_x_rooms"] = d["median_income"] * d["rooms_per_household"]
    d["income_x_age"] = d["median_income"] * d["housing_median_age"]
    return d

def group_of(n):
    if n in ("longitude", "latitude") or n.startswith("ocean_proximity"): return "Location"
    if "income" in n: return "Income"
    if n in ("total_rooms", "total_bedrooms", "rooms_per_household", "bedrooms_ratio"): return "Size of homes"
    return "Neighbourhood"

def contributions(model, X):
    try:
        pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        pre, reg = pipe.named_steps["pre"], pipe.named_steps["m"]
        x = pre.transform(X); x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        means = h.get("feature_means")
        centred = means is not None and len(means) == len(names)
        x0 = np.asarray(means) if centred else np.zeros(len(names))
        base = (float(reg.intercept_) + float(np.dot(reg.coef_, x0))) * 100000
        groups = {}
        for n, v in zip(names, (x[0] - x0) * reg.coef_):
            groups[group_of(n)] = groups.get(group_of(n), 0.0) + float(v) * 100000
        return base, groups, centred
    except Exception:
        return None

def money(v): return f"${v:,.0f}"
def signed(v): return ("+" if v >= 0 else "\u2212") + f"${abs(v):,.0f}"

PRESETS = {"Choose an area": None, "San Francisco": (37.77, -122.42, "NEAR BAY"), "Los Angeles": (34.05, -118.24, "<1H OCEAN"),
           "San Diego": (32.72, -117.16, "NEAR OCEAN"), "Sacramento": (38.58, -121.49, "INLAND"), "Fresno": (36.74, -119.79, "INLAND")}
for c in h["num_base"]:
    st.session_state.setdefault(c, float(rng[c][2]))
for c in h["cat_base"]:
    st.session_state.setdefault(c, cats[c][0])

def clamp(v, col): return float(min(max(v, rng[col][0]), rng[col][1]))
def apply_preset():
    p = PRESETS.get(st.session_state.get("preset"))
    if p:
        st.session_state["latitude"] = clamp(p[0], "latitude"); st.session_state["longitude"] = clamp(p[1], "longitude")
        if "ocean_proximity" in cats and p[2] in cats["ocean_proximity"]:
            st.session_state["ocean_proximity"] = p[2]

step = {"longitude": 0.01, "latitude": 0.01, "median_income": 0.1}
fmt = {"longitude": "%.2f", "latitude": "%.2f", "median_income": "%.1f"}
help_ = {"median_income": "Typical household income in the district, in tens of thousands of dollars (3.5 means about $35,000).",
         "housing_median_age": "Median age of the homes in the district.",
         "total_rooms": "All rooms in every home of the district.", "total_bedrooms": "All bedrooms in every home of the district.",
         "population": "Number of residents in the district.", "households": "Number of households in the district."}
label = {"longitude": "Longitude", "latitude": "Latitude", "median_income": "Median household income (x $10,000)",
         "housing_median_age": "Median age of homes (years)", "total_rooms": "Total rooms", "total_bedrooms": "Total bedrooms",
         "population": "Residents", "households": "Households"}

def slide(col, box):
    box.slider(label[col], float(rng[col][0]), float(rng[col][1]), step=step.get(col, 1.0), format=fmt.get(col, "%.0f"),
               key=col, help=help_.get(col))

left, right = st.columns([5, 6], gap="large")
with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Location</div>', unsafe_allow_html=True)
        st.selectbox("Start from an area", list(PRESETS), key="preset", on_change=apply_preset)
        c1, c2 = st.columns(2)
        slide("longitude", c1); slide("latitude", c2)
        for c in h["cat_base"]:
            st.selectbox("Distance to the ocean" if c == "ocean_proximity" else c, cats[c], key=c)
    with st.container(border=True):
        st.markdown('<div class="card-title">Neighbourhood</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        slide("median_income", c1); slide("housing_median_age", c2)
        slide("population", c1); slide("households", c2)
    with st.container(border=True):
        st.markdown('<div class="card-title">Homes</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        slide("total_rooms", c1); slide("total_bedrooms", c2)

d = {c: st.session_state[c] for c in h["num_base"] + h["cat_base"]}
X = pd.DataFrame([engineer(d)])
est = max(float(h["model"].predict(X)[0]) * 100000, 0.0)
err = float(h["rmse_usd"])
lo, hi = max(est - err, 0.0), est + err

with right:
    pct_html = ""
    if h.get("price_percentiles"):
        pct = int(min(max(np.searchsorted(h["price_percentiles"], est / 100000), 1), 99))
        pct_html = (f'<div class="pbar"><i style="left:{pct}%"></i></div>'
                    f'<div class="ptext">Higher than about {pct}% of California districts</div>')
    st.markdown(f'<div class="estimate"><div class="label">Estimated median home value</div>'
                f'<div class="figure">{money(est)}</div>'
                f'<div class="range">Likely range {money(lo)} &ndash; {money(hi)}</div>{pct_html}'
                f'<div class="note">Range reflects the model&rsquo;s typical error of about {money(err)}.</div></div>',
                unsafe_allow_html=True)
    try:
        st.map(pd.DataFrame({"lat": [d["latitude"]], "lon": [d["longitude"]]}), size=2500, color="#24473f", zoom=7)
    except Exception:
        pass
    res = contributions(h["model"], X)
    if res and abs(res[0] + sum(res[1].values()) - est) < 1.0:
        base, groups, centred = res
        mx = max(abs(v) for v in groups.values()) or 1.0
        rows = ""
        for name, v in sorted(groups.items(), key=lambda kv: -abs(kv[1])):
            w = abs(v) / mx * 50
            rows += (f'<div class="frow"><div>{name}</div><div class="ftrack"><div class="fbar {"fpos" if v >= 0 else "fneg"}" '
                     f'style="width:{w:.1f}%"></div></div><div class="fval">{signed(v)}</div></div>')
        st.markdown(f'<div class="sect">What shapes this estimate</div>'
                    f'<div class="sect-sub">Starting from {"the average California district" if centred else "the model baseline"} at about {money(base)}.</div>{rows}',
                    unsafe_allow_html=True)

by = f" Built by {AUTHOR}." if AUTHOR else ""
st.markdown(f'<div class="disc"><b>How this works.</b> The estimate comes from a regularised linear model ({h["name"]}) trained on '
            f'California census-district data, with an R&sup2; of {h["r2"]:.2f} on held-out districts. It describes a whole district, '
            f'not a single property, and is a statistical estimate rather than an appraisal.{by}</div>', unsafe_allow_html=True)