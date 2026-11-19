"""CWI Imputer -- Streamlit web app.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from imputer.cwi_knn import detect_numeric_columns, cwi_knn_auto, compare_with_baselines

st.set_page_config(page_title="Adaptive CWI Imputer", page_icon="🧩", layout="wide")
MODES = {"Most accurate (lowest error)": "accurate",
         "Realistic (keeps the natural spread of the data)": "realistic"}


def sample_data(n=600, gap=0.12, seed=1):
    r = np.random.default_rng(seed)
    price = r.uniform(10, 100, n).round(2)
    qty = r.integers(1, 11, n)
    cogs = price * qty
    d = pd.DataFrame({"Branch": r.choice(list("ABC"), n), "Unit price": price, "Quantity": qty,
                      "Tax 5%": 0.05 * cogs, "Sales": 1.05 * cogs, "cogs": cogs,
                      "Rating": r.uniform(4, 10, n).round(1)})
    for c in d.columns[1:]:
        d.loc[r.random(n) < gap, c] = np.nan
    return d


def read_file(f):
    return pd.read_excel(f) if f.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(f)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Imputation mode", list(MODES), help=(
        "Accurate: smallest error, but filled values are a bit 'average-looking'. "
        "Realistic: adds natural variation so the filled values look like real data."))
    mode = MODES[mode_label]
    st.caption("Numeric columns are detected automatically. Text columns are left unchanged.")

# ---------------- 1. load data ----------------
st.title("🧩 Adaptive CWI Imputer")
st.write("Upload a CSV or Excel file. We find the numeric columns, fill the gaps with our CWI-kNN method, "
         "tuned automatically on *your* data.")

c1, c2 = st.columns([3, 1])
src = c1.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
if c2.button("Try sample data", use_container_width=True):
    st.session_state.update(df=sample_data(), key=("sample",), result=None)
if src is not None and st.session_state.get("key") != (src.name, src.size):
    try:
        st.session_state.update(df=read_file(src), key=(src.name, src.size), result=None)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        st.stop()

df = st.session_state.get("df")
if df is None:
    st.info("Upload a file or click **Try sample data** to begin.")
    st.stop()

st.subheader("1. Your data")
st.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(10), use_container_width=True)

# ---------------- 2. detect columns ----------------
clean, cols, report = detect_numeric_columns(df, verbose=False)
st.subheader("2. Numeric columns detected automatically")
if len(cols) < 2:
    st.error("Need at least 2 usable numeric columns to learn relationships between columns.")
    st.stop()
miss = clean[cols].isna().sum()
table = pd.DataFrame({"Missing": miss, "Missing %": (miss / len(clean) * 100).round(1)})
st.dataframe(table, use_container_width=True)
skipped = {c: why for c, why in report.items() if why != "USED"}
if skipped:
    with st.expander(f"{len(skipped)} column(s) skipped"):
        for c, why in skipped.items():
            st.write(f"**{c}** — {why}")
if miss.sum() == 0:
    st.success("No missing values in the numeric columns. Nothing to fill.")
    st.stop()
if len(clean) > 20000:
    st.warning("Large file: the method tries several imputers, so this may take a few minutes.")

# ---------------- 3. impute ----------------
st.subheader("3. Fill the missing values with CWI-kNN")
if st.button("Fill missing values", type="primary"):
    with st.spinner("Tuning CWI-kNN on your data and filling the gaps…"):
        res, info = cwi_knn_auto(clean, cols, mode=mode)
        err_tbl, ks_tbl = compare_with_baselines(clean, cols, info["params"])
    st.session_state["result"] = (res, info, mode, err_tbl, ks_tbl)

out = st.session_state.get("result")
if out:
    res, info, used_mode, err_tbl, ks_tbl = out
    gap_cols = [c for c in cols if miss[c] > 0]
    st.success(f"Filled {int(miss.sum()):,} missing values with CWI-kNN ({used_mode} mode). "
               f"Settings chosen automatically: k = {info['params']['k']}, "
               f"helper columns = {info['params']['top_m']}.")

    st.markdown("**Estimated error per column (CWI-kNN)**")
    st.dataframe(info["est_error"].loc[gap_cols].round(3).to_frame("Estimated error"), width="stretch")
    st.caption("From hiding some known values and re-guessing them: 0 = perfect, "
               "1 = no better than guessing the column average. An estimate, not a guarantee.")

    st.markdown("**How does CWI-kNN compare with standard methods?** (lower = better)")
    t1, t2 = st.columns(2)
    t1.caption("Error per column")
    t1.dataframe(err_tbl.round(3), width="stretch")
    t2.caption("Distribution gap (0 = same shape as real values)")
    t2.dataframe(ks_tbl.round(3), width="stretch")
    st.caption("Mean, median and KNN are shown only as benchmarks. The filled values come from CWI-kNN.")

    st.markdown("**Filled-in values vs. observed values**")
    grid = st.columns(3)
    for i, c in enumerate(gap_cols):
        fig, ax = plt.subplots(figsize=(4, 2.8))
        gaps = clean[c].isna()
        ax.hist(clean.loc[~gaps, c], bins=20, alpha=0.55, density=True, label="observed")
        ax.hist(res.loc[gaps, c], bins=20, alpha=0.55, density=True, label="filled-in")
        ax.set_title(c, fontsize=10)
        ax.legend(fontsize=7)
        grid[i % 3].pyplot(fig)
        plt.close(fig)

    st.markdown("**Result**")
    st.dataframe(res.head(20), width="stretch")
    st.download_button("Download cleaned CSV", res.to_csv(index=False).encode("utf-8"),
                       file_name="imputed_data.csv", mime="text/csv")
