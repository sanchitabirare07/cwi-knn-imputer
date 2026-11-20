"""
cwi_knn.py  --  CWI-kNN v2: Correlation/Importance-Weighted Iterative k-NN imputation
Works on ANY tabular dataset: numeric columns are detected automatically.
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------
# 1. AUTOMATIC NUMERIC-COLUMN DETECTION
# ----------------------------------------------------------------------
def _try_parse_numeric(series, min_parse_ratio=0.9):
    """Text column like '$1,200' or '45%' -> numbers, if (almost) all values parse."""
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return None
    cleaned = series.astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})
    parsed = pd.to_numeric(cleaned, errors="coerce")
    non_null = series.notna() & ~series.astype(str).str.strip().isin(["", "NA", "N/A", "nan", "None"])
    if non_null.sum() == 0:
        return None
    if parsed[non_null].notna().mean() >= min_parse_ratio:
        return parsed
    return None


def detect_numeric_columns(df, max_missing_ratio=0.6, min_unique=3, verbose=True):
    """
    Decide automatically which columns can be imputed by the numeric method.
    Returns (clean_df, numeric_cols, report) where report explains every skipped column.
    """
    out = df.copy()
    numeric_cols, report = [], {}
    n = len(df)

    for col in df.columns:
        s = out[col]

        # text that is secretly numeric ("1,200", "$45", "12%")
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            parsed = _try_parse_numeric(s)
            if parsed is not None:
                out[col] = parsed
                s = out[col]

        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            report[col] = "skipped: not numeric (text / date / category)"
            continue

        non_null = s.dropna()
        if len(non_null) == 0 or s.isna().mean() > max_missing_ratio:
            report[col] = f"skipped: more than {int(max_missing_ratio*100)}% missing, too little to learn from"
            continue
        if non_null.nunique() <= 1:
            report[col] = "skipped: constant column (same value everywhere)"
            continue
        if non_null.nunique() < min_unique:
            report[col] = "skipped: only 2 distinct values, looks like a yes/no flag (use mode imputation)"
            continue
        looks_int = np.allclose(non_null, non_null.round())
        if looks_int and non_null.nunique() == len(non_null) and len(non_null) > 20:
            diffs = np.diff(non_null.values)
            if (diffs > 0).all() or re.search(r"(^|[^a-z])(id|index|no|number)($|[^a-z])", col.lower()):
                report[col] = "skipped: looks like an ID / row counter"
                continue
        numeric_cols.append(col)
        report[col] = "USED"

    if verbose:
        print(f"Detected {len(numeric_cols)} numeric columns to impute: {numeric_cols}")
        for c, why in report.items():
            if why != "USED":
                print(f"  - {c}: {why}")
    return out, numeric_cols, report


# ----------------------------------------------------------------------
# 2. THE IMPUTER
# ----------------------------------------------------------------------
def _column_weights(X, y, method, random_state):
    """How useful is each feature for predicting the target? (higher = more useful)"""
    if method == "importance":
        et = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=3, max_features=1.0,
                                 n_jobs=-1, random_state=random_state)
        et.fit(X, y)
        return et.feature_importances_
    # fallback: plain absolute correlation
    return np.nan_to_num(np.abs(pd.DataFrame(X).corrwith(pd.Series(y)).values))


def cwi_knn_impute(data_missing, columns=None, k=5, top_m=3, n_iterations=2,
                   weighting="importance", preserve_distribution=False,
                   random_state=0):
    """
    data_missing          : DataFrame containing NaNs
    columns               : numeric columns to impute (None -> detected automatically)
    k                     : neighbours averaged per missing cell
    top_m                 : how many helper columns define 'similar rows'
    n_iterations          : refinement passes
    weighting             : 'importance' (tree-based, catches interactions like price x qty)
                            or 'correlation' (the original v1 idea)
    preserve_distribution : True -> copy a real neighbour value instead of averaging, so the
                            imputed values keep the natural spread of the column
    """
    rng = np.random.default_rng(random_state)
    if columns is None:
        data_missing, columns, _ = detect_numeric_columns(data_missing)
    work = data_missing[columns].copy()
    miss = work.isna()
    work = work.fillna(work.median())          # rough start (median is safer than mean)
    order = miss.sum().sort_values().index.tolist()   # fewest missing first

    for it in range(n_iterations):
        last_pass = it == n_iterations - 1
        for target in order:
            to_fill = np.where(miss[target].values)[0]
            if len(to_fill) == 0:
                continue
            donors = np.where(~miss[target].values)[0]
            others = [c for c in columns if c != target]

            # a) which helper columns matter, and how much?
            w_all = _column_weights(work.iloc[donors][others].values,
                                    work.iloc[donors][target].values, weighting, random_state)
            top_idx = np.argsort(w_all)[::-1][:min(top_m, len(others))]
            top_cols = [others[i] for i in top_idx]
            w = w_all[top_idx]
            w = w / w.sum() if w.sum() > 0 else np.ones(len(top_idx)) / len(top_idx)

            # b) weighted feature space
            F = work[top_cols].values
            sd = F.std(axis=0); sd[sd == 0] = 1
            F = (F - F.mean(axis=0)) / sd * np.sqrt(w)      # sqrt: weights act on squared distance

            # c) neighbours among donors
            kk = min(k, len(donors))
            nn = NearestNeighbors(n_neighbors=kk).fit(F[donors])
            dist, nb = nn.kneighbors(F[to_fill])
            nb_rows = donors[nb]
            vals = work[target].values[nb_rows]
            inv = 1.0 / (dist + 1e-6)

            if preserve_distribution and last_pass:
                p = inv / inv.sum(axis=1, keepdims=True)
                pick = np.array([rng.choice(kk, p=p[i]) for i in range(len(to_fill))])
                new = vals[np.arange(len(to_fill)), pick]
            else:
                new = (vals * inv).sum(axis=1) / inv.sum(axis=1)

            work.iloc[to_fill, work.columns.get_loc(target)] = new

    result = data_missing.copy()
    result[columns] = work
    return result


# ----------------------------------------------------------------------
# 3. FAIR EVALUATION (per-column, so big-number columns don't dominate)
# ----------------------------------------------------------------------
def normalized_rmse(true_df, imputed_df, mask, columns):
    """RMSE of each column divided by that column's std  ->  1.0 = as bad as guessing the mean."""
    out = {}
    for c in columns:
        m = mask[c].values
        if m.sum() == 0:
            continue
        err = np.sqrt(np.mean((true_df.loc[m, c].values - imputed_df.loc[m, c].values) ** 2))
        out[c] = err / (true_df[c].std() + 1e-12)
    return pd.Series(out)


def distribution_distance(true_df, imputed_df, mask, columns):
    """Kolmogorov-Smirnov distance between hidden true values and predictions (0 = identical shape)."""
    from scipy.stats import ks_2samp
    return pd.Series({c: ks_2samp(true_df.loc[mask[c].values, c], imputed_df.loc[mask[c].values, c]).statistic
                      for c in columns if mask[c].sum() > 0})


# ----------------------------------------------------------------------
# 4. AUTO-TUNING (so the app picks k / top_m itself for any dataset)
# ----------------------------------------------------------------------
def auto_tune(data_missing, columns, k_grid=(3, 5, 8), top_m_grid=(2, 3, 4),
              probe_rate=0.10, random_state=0, **fixed):
    """Hide an extra 10% of the KNOWN values, try each setting, keep the best."""
    rng = np.random.default_rng(random_state)
    known = data_missing[columns].notna()
    probe = pd.DataFrame(False, index=data_missing.index, columns=columns)
    probed = data_missing.copy()
    for c in columns:
        idx = np.where(known[c].values)[0]
        pick = rng.choice(idx, size=max(1, int(len(idx) * probe_rate)), replace=False)
        probe.iloc[pick, probe.columns.get_loc(c)] = True
        probed.iloc[pick, probed.columns.get_loc(c)] = np.nan
    best, best_score = None, np.inf
    for k in k_grid:
        for m in top_m_grid:
            imp = cwi_knn_impute(probed, columns, k=k, top_m=min(m, len(columns) - 1), **fixed)
            score = normalized_rmse(data_missing, imp, probe, columns).mean()
            if score < best_score:
                best, best_score = {"k": k, "top_m": min(m, len(columns) - 1)}, score
    return best, best_score


# ----------------------------------------------------------------------
# 5. ADAPTIVE CWI  --  the "works on any dataset" version
# ----------------------------------------------------------------------
def _run_candidate(name, dm, columns, random_state=0):
    """Run one candidate imputer; every candidate returns a DataFrame with `columns` filled."""
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler
    X = dm[columns]
    if name == "mean":
        return X.fillna(X.mean())
    if name == "median":
        return X.fillna(X.median())
    if name == "knn":
        sc = StandardScaler(); Z = sc.fit_transform(X)
        out = KNNImputer(n_neighbors=5).fit_transform(Z)
        return pd.DataFrame(sc.inverse_transform(out), columns=columns, index=X.index)
    if name == "regression":      # good when relationships are roughly linear
        out = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=random_state).fit_transform(X)
        return pd.DataFrame(out, columns=columns, index=X.index)
    if name == "cwi_corr":
        return cwi_knn_impute(dm, columns, weighting="correlation", random_state=random_state)[columns]
    if name == "cwi_importance":
        return cwi_knn_impute(dm, columns, weighting="importance", random_state=random_state)[columns]
    raise ValueError(name)


CANDIDATES = ["mean", "median", "knn", "regression", "cwi_corr", "cwi_importance"]


def adaptive_impute(data_missing, columns=None, candidates=CANDIDATES, mode="accurate",
                    n_probes=2, probe_rate=0.10, random_state=0, verbose=True):
    """
    Pick the best imputation method FOR EACH COLUMN of THIS dataset, then use it.

    How: hide a further 10% of the cells we KNOW (a 'probe'), let every candidate guess them,
    and score each guess against the real values. The winner per column is used on the real gaps.

    mode = "accurate"  -> smallest possible error (predictions sit near the conditional average,
                          so their histogram is narrower than the real one -- unavoidable)
    mode = "realistic" -> same prediction + a realistic random error taken from the probe, so the
                          imputed values have the SAME SPREAD/SHAPE as the real column
                          (clipped to the observed range, rounded for whole-number columns).
    Returns (imputed_df, report_df).
    """
    rng = np.random.default_rng(random_state)
    if columns is None:
        data_missing, columns, _ = detect_numeric_columns(data_missing, verbose=verbose)
    base = data_missing[columns]
    scores = {c: {m: [] for m in candidates} for c in columns}
    resid = {c: {m: [] for m in candidates} for c in columns}

    for p in range(n_probes):
        probed, probe = base.copy(), pd.DataFrame(False, index=base.index, columns=columns)
        for c in columns:
            known = np.where(base[c].notna().values)[0]
            pick = rng.choice(known, size=max(1, int(len(known) * probe_rate)), replace=False)
            probe.iloc[pick, probe.columns.get_loc(c)] = True
            probed.iloc[pick, probed.columns.get_loc(c)] = np.nan
        for m in candidates:
            imp = _run_candidate(m, probed, columns, random_state + p)
            nr = normalized_rmse(base, imp, probe, columns)
            for c in columns:
                scores[c][m].append(nr[c])
                pm = probe[c].values
                resid[c][m].extend((base.loc[pm, c].values - imp.loc[pm, c].values).tolist())

    mean_scores = pd.DataFrame({c: {m: np.mean(v) for m, v in scores[c].items()} for c in columns}).T
    best = mean_scores.idxmin(axis=1)

    fills = {m: _run_candidate(m, base, columns, random_state) for m in set(best.values)}
    result = data_missing.copy()
    for c in columns:
        gaps = base[c].isna().values
        if gaps.sum() == 0:
            continue
        pred = fills[best[c]].loc[gaps, c].values.astype(float)
        if mode == "realistic":
            pred = pred + rng.choice(np.array(resid[c][best[c]]), size=gaps.sum(), replace=True)
            obs = base[c].dropna()
            pred = np.clip(pred, obs.min(), obs.max())
            if np.allclose(obs, obs.round()):
                pred = np.round(pred)
        result.loc[gaps, c] = pred

    report = mean_scores.round(3)
    report["chosen"] = best
    if verbose:
        print(f"Mode: {mode}. Method chosen per column (probe error, lower = better):")
        print(report)
    return result, report


def plot_distribution_check(true_df, imputed_df, mask, columns, ncols=3):
    """Hidden REAL values vs the values our method predicted for them, one histogram per column."""
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(4.5 * ncols, 3.2 * rows))
    for ax, c in zip(np.atleast_1d(axes).ravel(), columns):
        m = mask[c].values
        ax.hist(true_df.loc[m, c], bins=20, alpha=0.55, density=True, label="real (hidden)")
        ax.hist(imputed_df.loc[m, c], bins=20, alpha=0.55, density=True, label="imputed")
        ax.set_title(c); ax.legend(fontsize=7)
    for ax in np.atleast_1d(axes).ravel()[len(columns):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()


def hide_values(df, columns, rate=0.15, seed=0):
    """For testing on a COMPLETE dataset: hide `rate` of the values, return (df_with_gaps, mask)."""
    r = np.random.default_rng(seed)
    dm, mask = df.copy(), pd.DataFrame(False, index=df.index, columns=columns)
    for c in columns:
        idx = r.choice(len(df), int(len(df) * rate), replace=False)
        mask.iloc[idx, mask.columns.get_loc(c)] = True
        dm.iloc[idx, dm.columns.get_loc(c)] = np.nan
    return dm, mask


# ----------------------------------------------------------------------
# 6. CWI-kNN ONLY  --  our own method is the single imputation engine
# ----------------------------------------------------------------------
def _hide_known(base, columns, rng, probe_rate):
    """Hide a share of the KNOWN values (the 'probe') so we have real answers to check against."""
    probed, probe = base.copy(), pd.DataFrame(False, index=base.index, columns=columns)
    for c in columns:
        known = np.where(base[c].notna().values)[0]
        pick = rng.choice(known, size=max(1, int(len(known) * probe_rate)), replace=False)
        probe.iloc[pick, probe.columns.get_loc(c)] = True
        probed.iloc[pick, probed.columns.get_loc(c)] = np.nan
    return probed, probe


def cwi_knn_auto(data_missing, columns=None, mode="accurate", k_grid=(3, 5, 8),
                 top_m_grid=(2, 3, 4), probe_rate=0.10, random_state=0):
    """
    Fill missing values with CWI-kNN v2 ONLY.
    1) auto_tune picks the best k and top_m for CWI-kNN on this dataset.
    2) CWI-kNN fills the real gaps with those settings.
    3) mode='realistic' adds natural random error (taken from a hidden-value test) so the filled
       values keep the spread of the real column; mode='accurate' keeps the lowest-error guess.
    Returns (filled_df, info) where info = {'params': {...}, 'est_error': Series per column}.
    """
    rng = np.random.default_rng(random_state)
    if columns is None:
        data_missing, columns, _ = detect_numeric_columns(data_missing, verbose=False)
    params, _ = auto_tune(data_missing, columns, k_grid, top_m_grid, probe_rate, random_state)
    base = data_missing[columns]

    probed, probe = _hide_known(base, columns, rng, probe_rate)
    guess = cwi_knn_impute(probed, columns, **params, random_state=random_state)[columns]
    est = normalized_rmse(base, guess, probe, columns)

    result = cwi_knn_impute(data_missing, columns, **params, random_state=random_state)
    if mode == "realistic":
        for c in columns:
            gaps = base[c].isna().values
            if gaps.sum() == 0:
                continue
            pm = probe[c].values
            resid = base.loc[pm, c].values - guess.loc[pm, c].values
            vals = result.loc[gaps, c].values + rng.choice(resid, size=gaps.sum(), replace=True)
            obs = base[c].dropna()
            vals = np.clip(vals, obs.min(), obs.max())
            if np.allclose(obs, obs.round()):
                vals = np.round(vals)
            result.loc[gaps, c] = vals
    return result, {"params": params, "est_error": est}


def compare_with_baselines(data_missing, columns, params, n_probes=2, probe_rate=0.10, random_state=0):
    """
    Benchmark only: hide known values, let mean / median / KNN / CWI-kNN guess them, and score each.
    Returns (error_table, distribution_gap_table); lower is better in both.
    """
    rng = np.random.default_rng(random_state)
    base = data_missing[columns]
    errs, kss = [], []
    for _ in range(n_probes):
        probed, probe = _hide_known(base, columns, rng, probe_rate)
        guesses = {"Mean": _run_candidate("mean", probed, columns),
                   "Median": _run_candidate("median", probed, columns),
                   "KNN": _run_candidate("knn", probed, columns),
                   "CWI-kNN (ours)": cwi_knn_impute(probed, columns, **params,
                                                   random_state=random_state)[columns]}
        errs.append(pd.DataFrame({m: normalized_rmse(base, g, probe, columns) for m, g in guesses.items()}))
        kss.append(pd.DataFrame({m: distribution_distance(base, g, probe, columns) for m, g in guesses.items()}))
    err, ks = sum(errs) / n_probes, sum(kss) / n_probes
    err.loc["AVERAGE"], ks.loc["AVERAGE"] = err.mean(), ks.mean()
    return err, ks


def benchmark_guesses(data_missing, columns, params, probe_rate=0.10, random_state=0):
    """
    One hidden-value test for plotting. Returns {column: DataFrame} where each DataFrame has the
    real hidden values and the guesses of Mean, Median, KNN and CWI-kNN for those same cells.
    (Uses the same hidden cells as the first test inside compare_with_baselines.)
    """
    rng = np.random.default_rng(random_state)
    base = data_missing[columns]
    probed, probe = _hide_known(base, columns, rng, probe_rate)
    guesses = {"Mean": _run_candidate("mean", probed, columns),
               "Median": _run_candidate("median", probed, columns),
               "KNN": _run_candidate("knn", probed, columns),
               "CWI-kNN (ours)": cwi_knn_impute(probed, columns, **params,
                                               random_state=random_state)[columns]}
    out = {}
    for c in columns:
        m = probe[c].values
        d = {"Real (hidden)": base.loc[m, c].values}
        for name, g in guesses.items():
            d[name] = g.loc[m, c].values
        out[c] = pd.DataFrame(d)
    return out
