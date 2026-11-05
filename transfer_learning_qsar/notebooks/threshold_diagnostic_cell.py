# =====================================================================================
# Part A.2b — Label-distribution diagnostic: where should the activity-class cuts go?
#
# Drop this in as a new cell immediately AFTER cell 9 (the BindingDB pull), which is
# where `raw_bdb_df` and `train_df` come into existence. It is read-only: it adds no
# variables the rest of the pipeline depends on and changes nothing.
#
# It answers four questions:
#   1. What shape is the label distribution really? (percentiles, skew, bimodality)
#   2. How many censored (">" / "<") rows were dropped, and what would they add back
#      as confident inactives / actives?
#   3. How many compounds land in each class under candidate threshold schemes?
#   4. How noisy is the labelling at the boundaries, measured by pIC50-vs-pKi
#      disagreement on compounds that have both?
# =====================================================================================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.width", 160)

EPS = list(CONFIG.endpoint_names)          # ['pIC50', 'pKi']
_COLS = dict(zip(CONFIG.endpoint_names, CONFIG.affinity_columns))

print("=" * 86)
print("1. SHAPE OF THE LABEL DISTRIBUTION")
print("=" * 86)

_shape_rows = []
for ep in EPS:
    v = train_df[ep].dropna().values
    if len(v) < 10:
        continue
    q = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    _shape_rows.append({
        "endpoint": ep, "n": len(v),
        "mean": v.mean(), "sd": v.std(ddof=0),
        "p1": q[0], "p5": q[1], "p10": q[2], "p25": q[3], "p50": q[4],
        "p75": q[5], "p90": q[6], "p95": q[7], "p99": q[8],
        "skew": stats.skew(v), "excess_kurt": stats.kurtosis(v),
        "min": v.min(), "max": v.max(),
    })
shape_df = pd.DataFrame(_shape_rows).set_index("endpoint").round(3)
display(shape_df)

# --- is it one mode or several? -------------------------------------------------------
# A kernel-density scan: local minima of the density between the 5th and 95th percentile
# are the natural "valleys", i.e. the least-populated places to put a cut.
def density_valleys(v, n_grid=512, lo_pct=5, hi_pct=95):
    kde = stats.gaussian_kde(v)
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    grid = np.linspace(lo, hi, n_grid)
    dens = kde(grid)
    # interior local minima
    is_min = (dens[1:-1] < dens[:-2]) & (dens[1:-1] < dens[2:])
    valleys = grid[1:-1][is_min]
    depths = dens[1:-1][is_min]
    order = np.argsort(depths)              # shallowest density = emptiest = best cut
    return grid, dens, valleys[order], depths[order]

print("\nBimodality check and natural density valleys")
print("-" * 86)
valley_store = {}
for ep in EPS:
    v = train_df[ep].dropna().values
    if len(v) < 30:
        print(f"  {ep}: too few labelled rows for a density scan.")
        continue
    grid, dens, valleys, depths = density_valleys(v)
    valley_store[ep] = (grid, dens, valleys)
    # Crude unimodality signal: compare the best single Gaussian to a 2-component mix.
    try:
        from sklearn.mixture import GaussianMixture
        X = v.reshape(-1, 1)
        bic1 = GaussianMixture(1, random_state=0).fit(X).bic(X)
        bic2 = GaussianMixture(2, random_state=0).fit(X).bic(X)
        bic3 = GaussianMixture(3, random_state=0).fit(X).bic(X)
        best = {1: bic1, 2: bic2, 3: bic3}
        n_modes = min(best, key=best.get)
        verdict = (f"{n_modes}-component mixture fits best "
                   f"(BIC 1/2/3 = {bic1:.0f} / {bic2:.0f} / {bic3:.0f})")
    except Exception as e:
        verdict = f"mixture fit unavailable ({e})"
    vstr = ", ".join(f"{x:.2f}" for x in valleys[:4]) if len(valleys) else "none found"
    print(f"  {ep}: {verdict}")
    print(f"         emptiest cut points (shallowest density first): {vstr}")
    print(f"         -> a threshold placed in a valley splits fewer near-identical "
          f"compounds across the boundary")

print("\n" + "=" * 86)
print("2. CENSORED-ROW AUDIT  (the data classification would recover)")
print("=" * 86)

_cens_rows = []
_recovered = {}          # endpoint -> DataFrame of censored measurements
for ep in EPS:
    col = _COLS[ep]
    sub = raw_bdb_df[raw_bdb_df["endpoint_name"] == ep].copy()
    raw = sub["affinity_raw"].astype(str)
    sub["qualifier"] = raw.str.extract(r"^\s*([<>]=?)")[0]
    sub["value_nm"] = pd.to_numeric(raw.str.replace(r"^\s*[<>]=?\s*", "", regex=True),
                                    errors="coerce")
    cens = sub[sub["qualifier"].notna() & sub["value_nm"].notna()].copy()
    cens["pAffinity_bound"] = 9 - np.log10(cens["value_nm"])
    _recovered[ep] = cens

    greater = cens[cens["qualifier"].str.startswith(">")]
    lesser = cens[cens["qualifier"].str.startswith("<")]
    _cens_rows.append({
        "endpoint": ep,
        "raw_rows": len(sub),
        "censored_rows": len(cens),
        "'>' rows (-> INACTIVE)": len(greater),
        "'<' rows (-> ACTIVE)": len(lesser),
        "distinct cpds from '>'": greater["smiles"].nunique(),
        "distinct cpds from '<'": lesser["smiles"].nunique(),
        "numeric cpds now used": int(train_df[ep].notna().sum()),
    })

cens_df = pd.DataFrame(_cens_rows).set_index("endpoint")
display(cens_df)

print("Where the '>' bounds sit (these become inactives; a '>1 uM' is a much weaker")
print("statement than a '>100 uM', so they are not interchangeable):")
for ep in EPS:
    g = _recovered[ep]
    g = g[g["qualifier"].str.startswith(">")]
    if len(g) == 0:
        print(f"  {ep}: no '>' rows.")
        continue
    vc = g["value_nm"].value_counts().sort_index()
    print(f"  {ep}:")
    for val, n in vc.items():
        print(f"      > {val:>12,.1f} nM  (pAffinity < {9 - np.log10(val):.2f})   n = {n}")

# How many of the '>' compounds are genuinely NEW, vs already present with a number?
print("\nOverlap with the compounds already in train_df:")
for ep in EPS:
    g = _recovered[ep]
    g = g[g["qualifier"].str.startswith(">")]
    if len(g) == 0:
        continue
    new = set(g["smiles"]) - set(train_df["smiles"])
    print(f"  {ep}: {g['smiles'].nunique()} distinct '>' compounds, "
          f"of which {len(new)} are not in the current training set at all")
print("  (Note: matching is on raw SMILES strings, as the pipeline does, so this is an")
print("   under-count of true overlap - standardisation would merge more. Dedup on")
print("   InChIKey after standardisation before trusting these numbers.)")

print("\n" + "=" * 86)
print("3. CLASS COUNTS UNDER CANDIDATE THRESHOLD SCHEMES")
print("=" * 86)

# (name, active_cut, inactive_cut) on the pAffinity scale; class is
#   ACTIVE    if  p >= active_cut
#   INACTIVE  if  p <  inactive_cut
#   MODERATE  otherwise
_p = lambda nm: 9 - np.log10(nm)

SCHEMES = {
    "field-standard (100 nM / 10 uM)": (_p(100), _p(10_000)),
    "PSMA-tightened (10 nM / 1 uM)":   (_p(10),  _p(1_000)),
    "PSMA-strict (1 nM / 1 uM)":       (_p(1),   _p(1_000)),
}
# add tertiles of the observed numeric distribution, per endpoint, for comparison only
for ep in EPS:
    v = train_df[ep].dropna().values
    if len(v) > 30:
        lo, hi = np.percentile(v, [33.3, 66.7])
        SCHEMES[f"tertiles of observed {ep} ({hi:.2f} / {lo:.2f})"] = (hi, lo)
# add the emptiest density valleys, if two were found
for ep in EPS:
    if ep in valley_store and len(valley_store[ep][2]) >= 2:
        two = sorted(valley_store[ep][2][:2])
        SCHEMES[f"density valleys of {ep} ({two[1]:.2f} / {two[0]:.2f})"] = (two[1], two[0])


def classify(p, active_cut, inactive_cut):
    if not np.isfinite(p):
        return np.nan
    if p >= active_cut:
        return "active"
    if p < inactive_cut:
        return "inactive"
    return "moderate"


rows = []
for name, (a_cut, i_cut) in SCHEMES.items():
    for ep in EPS:
        v = train_df[ep].dropna()
        cls = v.map(lambda p: classify(p, a_cut, i_cut))
        counts = cls.value_counts()
        # censored '>' rows become inactives, '<' rows become actives
        g = _recovered[ep]
        add_inact = g[g["qualifier"].str.startswith(">")]
        # only count a '>' row as inactive if its bound actually implies inactivity
        add_inact = add_inact[add_inact["pAffinity_bound"] <= i_cut]
        add_act = g[g["qualifier"].str.startswith("<")]
        add_act = add_act[add_act["pAffinity_bound"] >= a_cut]
        rows.append({
            "scheme": name, "endpoint": ep,
            "active": int(counts.get("active", 0)),
            "moderate": int(counts.get("moderate", 0)),
            "inactive": int(counts.get("inactive", 0)),
            "+inactive from '>'": add_inact["smiles"].nunique(),
            "+active from '<'": add_act["smiles"].nunique(),
            "inactive TOTAL": int(counts.get("inactive", 0)) + add_inact["smiles"].nunique(),
        })

class_counts = pd.DataFrame(rows).set_index(["scheme", "endpoint"])
display(class_counts)

print("Reading this table:")
print("  * 'inactive TOTAL' is the figure that decides 3-class vs binary. Under ~80")
print("    compounds and three classes will not train reliably - go binary")
print("    (active vs not-active) instead.")
print("  * A scheme that puts >70% of compounds in one class will need class weights")
print("    or stratified sampling whichever way you go.")
print("  * Prefer a pharmacologically meaningful cut that happens to sit near a density")
print("    valley. Do NOT pick tertiles because they balance the classes - that makes")
print("    'active' mean 'top third of this snapshot', which moves when data is added.")

print("\n" + "=" * 86)
print("4. BOUNDARY NOISE: do pIC50 and pKi agree on the class?")
print("=" * 86)
print("Compounds measured both ways should land in the same class. Where they do not,")
print("the disagreement is assay noise, and it bounds how good any classifier can be.\n")

both = train_df[train_df[EPS[0]].notna() & train_df[EPS[1]].notna()]
print(f"Compounds with both {EPS[0]} and {EPS[1]}: {len(both)}")
if len(both) >= 10:
    d = (both[EPS[0]] - both[EPS[1]]).abs()
    print(f"  |{EPS[0]} - {EPS[1]}|:  median {d.median():.2f}, mean {d.mean():.2f}, "
          f"90th pct {d.quantile(0.9):.2f}, max {d.max():.2f} log units")
    print(f"  fraction differing by >0.5 log: {100 * (d > 0.5).mean():.1f}%")
    print(f"  fraction differing by >1.0 log: {100 * (d > 1.0).mean():.1f}%")
    print(f"  Pearson r between the two endpoints: {both[EPS[0]].corr(both[EPS[1]]):.3f}\n")
    for name, (a_cut, i_cut) in SCHEMES.items():
        c0 = both[EPS[0]].map(lambda p: classify(p, a_cut, i_cut))
        c1 = both[EPS[1]].map(lambda p: classify(p, a_cut, i_cut))
        agree = (c0 == c1).mean()
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(c0, c1)
            kstr = f", kappa = {kappa:.3f}"
        except Exception:
            kstr = ""
        print(f"  {name:<46s} agreement = {100 * agree:5.1f}%{kstr}")
    print("\n  Agreement here is a CEILING on classifier accuracy: the same molecule,")
    print("  measured twice, disagrees this often. If a scheme falls below ~80%, its")
    print("  boundaries are cutting through the noise rather than through the signal.")
else:
    print("  Too few dual-measured compounds to assess boundary noise.")

print("\n" + "=" * 86)
print("5. PLOTS")
print("=" * 86)

fig, axes = plt.subplots(1, len(EPS), figsize=(7.5 * len(EPS), 4.8))
axes = np.atleast_1d(axes)
_cut_colors = plt.get_cmap("tab10")
for ax, ep in zip(axes, EPS):
    v = train_df[ep].dropna().values
    ax.hist(v, bins=40, color="#9aa0a6", edgecolor="white", linewidth=0.5,
            alpha=0.85, label=f"{ep} (n={len(v)})")
    if ep in valley_store:
        grid, dens, valleys = valley_store[ep]
        ax2 = ax.twinx()
        ax2.plot(grid, dens, color="#1a1a1a", linewidth=1.6, label="density")
        ax2.set_yticks([])
        for x in valleys[:3]:
            ax2.axvline(x, color="#1a1a1a", linestyle=":", linewidth=1.2, alpha=0.7)
    for ci, (name, (a_cut, i_cut)) in enumerate(list(SCHEMES.items())[:3]):
        col = _cut_colors(ci)
        ax.axvline(a_cut, color=col, linestyle="--", linewidth=1.8,
                   label=f"{name.split(' (')[0]}: active")
        ax.axvline(i_cut, color=col, linestyle="-.", linewidth=1.4, alpha=0.8)
    g = _recovered[ep]
    g = g[g["qualifier"].str.startswith(">")]
    if len(g):
        ax.scatter(g["pAffinity_bound"], np.full(len(g), ax.get_ylim()[1] * 0.04),
                   marker="|", s=180, color="#d1495b", alpha=0.6,
                   label=f"censored '>' bounds (n={len(g)})")
    ax.set_xlabel(ep)
    ax.set_ylabel("compounds")
    ax.set_title(f"{ep}: distribution, density valleys (dotted) and candidate cuts")
    ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.show()

print("Dotted vertical lines are density valleys - the emptiest places to cut.")
print("Red ticks are the censored '>' bounds that classification would recover as")
print("inactives; where they pile up tells you where the real inactive mass sits.")
