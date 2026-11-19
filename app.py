
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CWI-kNN | Missing Value Imputation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

[data-testid="stSidebar"] hr {
    border-color: #374151;
}

.hero {
    background: linear-gradient(120deg, #172554, #2563eb);
    padding: 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
}

.section-heading {
    color: #172554;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748b;
}

[data-testid="stMetricValue"] {
    color: #172554;
}

.dataset-status {
    background-color: #c4b5fd;
    color: #312e81 !important;
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 30px;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. SESSION STATE INITIALIZATION
# =========================================================

defaults = {
    "dataset": None,
    "filename": None,
    "file_identity": None,
    "imputed_dataset": None,
    "imputation_details": None,
    "evaluation_results": None,
    "experiment_history": [],
    "saved_config": {
        "k": 5,
        "top_n": 3,
        "iterations": 2
    },
    "config_k": 5,
    "config_top_n": 3,
    "config_iterations": 2
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_eligible_columns(df):
    eligible = []

    for col in get_numeric_columns(df):

        if df[col].notna().sum() == 0:
            continue

        if df[col].nunique(dropna=True) <= 1:
            continue

        eligible.append(col)

    return eligible


def get_excluded_columns(df):
    excluded = []

    for col in get_numeric_columns(df):

        if df[col].notna().sum() == 0:
            excluded.append((col, "All values are missing"))

        elif df[col].nunique(dropna=True) <= 1:
            excluded.append((col, "Constant column"))

    return excluded


def get_missing_summary(df):
    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    return summary.sort_values(
        by="Missing Values",
        ascending=False
    )


def get_correlation_data(df, columns):
    if len(columns) < 2:
        return None

    # Replace infinite values before correlation calculation.
    numeric_data = df[columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return numeric_data.corr()


def reset_results():
    st.session_state.imputed_dataset = None
    st.session_state.imputation_details = None
    st.session_state.evaluation_results = None
    st.session_state.experiment_history = []


def download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def show_heading(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        st.caption(description)


def get_evaluation_table(results):
    """
    Convert supported backend evaluation results
    into a DataFrame for display.
    """

    if isinstance(results, pd.DataFrame):
        return results.copy()

    if isinstance(results, list):
        return pd.DataFrame(results)

    if isinstance(results, dict):

        # A dictionary containing a results list/table.
        if "results" in results:
            return get_evaluation_table(results["results"])

        # A dictionary containing metric values.
        return pd.DataFrame(
            list(results.items()),
            columns=["Metric", "Value"]
        )

    return pd.DataFrame()


def get_metric_value(results, metric_name):
    """
    Read an RMSE or MAE value from common backend
    result formats.
    """

    if isinstance(results, dict):

        for key, value in results.items():

            if str(key).strip().lower() == metric_name.lower():

                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None

        if "metrics" in results and isinstance(results["metrics"], dict):
            return get_metric_value(
                results["metrics"],
                metric_name
            )

    if isinstance(results, pd.DataFrame):

        if results.empty:
            return None

        # Format: columns named RMSE and MAE.
        for col in results.columns:

            if str(col).strip().lower() == metric_name.lower():

                values = pd.to_numeric(
                    results[col],
                    errors="coerce"
                ).dropna()

                if not values.empty:
                    return float(values.iloc[0])

        # Format: Metric | Value.
        if "Metric" in results.columns and "Value" in results.columns:

            matches = results[
                results["Metric"].astype(str).str.lower()
                == metric_name.lower()
            ]

            if not matches.empty:

                try:
                    return float(matches.iloc[0]["Value"])
                except (ValueError, TypeError):
                    return None

    return None


# =========================================================
# 5. CORRELATION ANALYSIS
# =========================================================

def show_correlation_analysis(df, columns):

    if len(columns) < 2:

        st.info(
            "At least two eligible numerical columns "
            "are required for correlation analysis."
        )

        return

    corr = get_correlation_data(df, columns)

    if corr is None or corr.empty:

        st.info("Correlation values could not be calculated.")

        return

    # Make an independent copy.
    # Undefined correlations remain NaN and are masked
    # in the heatmap instead of being shown as zero.

    display_corr = corr.copy()

    mask = display_corr.isna()

    fig, ax = plt.subplots(
        figsize=(
            max(9, len(columns) * 1.15),
            max(6, len(columns) * 0.85)
        )
    )

    sns.heatmap(
        display_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.caption(
        "Blank cells indicate correlations that could not be calculated."
    )

    st.markdown("### How to Read This Heatmap")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("🔵 **Positive Correlation**")
        st.caption(
            "Values closer to +1 indicate a positive relationship."
        )

    with col2:
        st.markdown("🔴 **Negative Correlation**")
        st.caption(
            "Values closer to -1 indicate an inverse relationship."
        )

    with col3:
        st.markdown("⚪ **Weak Correlation**")
        st.caption(
            "Values closer to 0 indicate a weak linear relationship."
        )

    st.markdown("### Strongest Feature Relationships")

    pairs = []

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = corr.iloc[i, j]

            if pd.notna(value):

                pairs.append({
                    "Feature 1": columns[i],
                    "Feature 2": columns[j],
                    "Correlation": round(float(value), 3),
                    "Absolute Correlation": abs(float(value))
                })

    pairs.sort(
        key=lambda x: x["Absolute Correlation"],
        reverse=True
    )

    if pairs:

        pairs_df = pd.DataFrame(pairs)

        st.dataframe(
            pairs_df[
                ["Feature 1", "Feature 2", "Correlation"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No valid feature relationships were found.")

    st.markdown("### How CWI-kNN Uses Correlation")

    st.write("""
    CWI-kNN examines the relationship between numerical features
    to identify the most relevant columns for estimating missing
    values.

    It selects strongly correlated features and gives them
    importance during the nearest-neighbour search.
    """)


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# 📊 CWI-kNN")

    st.caption("Missing Value Imputation System")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Upload Dataset",
            "CWI-kNN Imputation",
            "Imputation Results",
            "Benchmark & Evaluation",
            "Experiment History"
        ]
    )

    st.markdown("---")

    st.markdown("### Project Information")

    st.caption("Algorithm")
    st.write("Correlation-Weighted Iterative KNN")

    st.caption("Technology")
    st.write("Python · Streamlit · Pandas")

    st.caption("Supported Data")
    st.write("Numerical tabular data")

    st.markdown("---")

    if st.session_state.dataset is not None:

        st.markdown(
            '<div class="dataset-status">✓ Dataset Loaded</div>',
            unsafe_allow_html=True
        )

        st.caption(st.session_state.filename)

    else:

        st.caption("No dataset uploaded yet.")


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to CWI-kNN</h1>
        <p>
        An Adaptive Framework for Missing-Value Imputation
        Using Correlation-Weighted Iterative K-Nearest Neighbors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    st.write("""
    CWI-kNN is a missing-value imputation framework designed
    to estimate missing numerical values by considering
    relationships between features in a dataset.

    It combines correlation-based feature selection,
    weighted similarity and iterative refinement to improve
    the imputation process.
    """)

    st.markdown("## How It Works")

    steps = [
        ("01", "Upload Dataset", "Import your CSV dataset."),
        ("02", "Analyze Data", "Identify missing values and correlations."),
        ("03", "Select Features", "Find relevant correlated features."),
        ("04", "Apply CWI-kNN", "Use weighted K-nearest neighbors."),
        ("05", "Refine Results", "Iteratively improve estimated values."),
        ("06", "Evaluate", "Compare results using RMSE and MAE.")
    ]

    for number, title, description in steps:

        with st.container(border=True):

            st.markdown(f"### {number} | {title}")

            st.write(description)

    st.markdown("---")

    st.markdown("## Current Dataset Status")

    if st.session_state.dataset is not None:

        df = st.session_state.dataset

        numeric_cols = get_numeric_columns(df)

        total_missing = int(df.isnull().sum().sum())

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", f"{df.shape[0]:,}")
        c2.metric("Total Columns", f"{df.shape[1]:,}")
        c3.metric("Numerical Columns", len(numeric_cols))
        c4.metric("Missing Values", f"{total_missing:,}")

        st.success(
            f"Currently loaded: {st.session_state.filename}"
        )

    else:

        st.info(
            "Upload a dataset to view its statistics "
            "and begin analysis."
        )

        st.markdown("""
        **Getting Started**

        Go to the Upload Dataset section from the sidebar
        and upload a CSV file to begin.
        """)


# =========================================================
# 8. UPLOAD DATASET
# =========================================================

elif page == "Upload Dataset":

    st.title("📁 Upload Dataset")

    st.write(
        "Upload a CSV file to analyze its structure, "
        "missing values and numerical feature relationships."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a dataset in CSV format."
    )

    if uploaded_file is not None:

        try:

            file_bytes = uploaded_file.getvalue()

            identity = (
                uploaded_file.name,
                len(file_bytes),
                hash(file_bytes)
            )

            if identity != st.session_state.file_identity:

                df = pd.read_csv(uploaded_file)

                if df.empty or len(df.columns) == 0:
                    st.error("The uploaded CSV file is empty.")

                else:

                    st.session_state.dataset = df
                    st.session_state.filename = uploaded_file.name
                    st.session_state.file_identity = identity

                    reset_results()

                    st.success("Dataset uploaded successfully!")

        except pd.errors.EmptyDataError:

            st.error("The uploaded CSV file contains no readable data.")

        except Exception as e:

            st.error(f"Unable to read this CSV file: {e}")

    if st.session_state.dataset is not None:

        df = st.session_state.dataset

        numeric_cols = get_numeric_columns(df)

        eligible_cols = get_eligible_columns(df)

        excluded_cols = get_excluded_columns(df)

        st.markdown("---")

        st.markdown("## Dataset Overview")

        st.write(f"**File Name:** {st.session_state.filename}")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Numerical Columns", len(numeric_cols))
        c4.metric("Total Missing", f"{df.isnull().sum().sum():,}")

        st.markdown("---")

        show_heading(
            "Dataset Preview",
            "A preview of the uploaded dataset."
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        show_heading("Column Information")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique(dropna=True).values
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        show_heading("Missing Value Analysis")

        missing_summary = get_missing_summary(df)

        st.dataframe(
            missing_summary,
            use_container_width=True,
            hide_index=True
        )

        missing_columns = missing_summary[
            missing_summary["Missing Values"] > 0
        ]

        if not missing_columns.empty:

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.bar(
                missing_columns["Column"].astype(str),
                missing_columns["Missing Values"]
            )

            ax.set_title(
                "Missing Values by Column",
                fontsize=14,
                fontweight="bold"
            )

            ax.set_ylabel("Number of Missing Values")
            ax.set_xlabel("Columns")

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

        else:

            st.success(
                "No missing values were found in this dataset."
            )

        st.download_button(
            "⬇ Download Missing Value Report",
            data=download_csv(missing_summary),
            file_name="missing_value_report.csv",
            mime="text/csv"
        )

        st.markdown("---")

        show_heading("Numerical and Non-Numerical Columns")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Numerical Columns**")

            st.write(
                numeric_cols if numeric_cols else "None"
            )

        with col2:

            st.markdown("**Non-Numerical Columns**")

            non_numeric = df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            st.write(
                non_numeric if non_numeric else "None"
            )

        if excluded_cols:

            st.warning(
                "Some numerical columns cannot be processed:"
            )

            for column, reason in excluded_cols:

                st.write(f"• **{column}** — {reason}")

        st.caption(
            "The current CWI-kNN implementation is designed "
            "for eligible numerical columns. Non-numerical "
            "columns are retained but are not imputed."
        )

        st.markdown("---")

        show_heading(
            "Correlation Analysis",
            "Explore relationships between eligible numerical features."
        )

        if len(eligible_cols) >= 2:

            show_correlation_analysis(df, eligible_cols)

        else:

            st.info(
                "At least two eligible numerical columns "
                "are needed for correlation analysis."
            )

        st.markdown("---")

        show_heading("Dataset Quality Check")

        duplicate_count = int(df.duplicated().sum())

        missing_count = int(df.isnull().sum().sum())

        q1, q2, q3 = st.columns(3)

        q1.metric("Duplicate Rows", duplicate_count)

        q2.metric("Missing Values", missing_count)

        q3.metric(
            "Eligible Numerical Columns",
            len(eligible_cols)
        )

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count} duplicate rows were detected."
            )

        else:

            st.success("No duplicate rows detected.")

        if len(eligible_cols) == 0:

            st.warning(
                "No eligible numerical columns were found "
                "for CWI-kNN processing."
            )

        st.markdown("---")

        if st.button(
            "🗑 Remove Uploaded Dataset",
            type="secondary"
        ):

            st.session_state.dataset = None
            st.session_state.filename = None
            st.session_state.file_identity = None

            reset_results()

            st.rerun()

    else:

        st.info(
            "Please upload a CSV file to begin dataset analysis."
        )


# =========================================================
# 9. CWI-kNN IMPUTATION
# =========================================================

elif page == "CWI-kNN Imputation":

    st.title("⚙️ CWI-kNN Imputation")

    st.write(
        "Configure the algorithm parameters before "
        "performing missing-value imputation."
    )

    if st.session_state.dataset is None:

        st.warning("Please upload a dataset first.")

        st.info("Go to Upload Dataset from the sidebar.")

    else:

        df = st.session_state.dataset

        eligible_cols = get_eligible_columns(df)

        missing_numeric_cols = [
            col for col in eligible_cols
            if df[col].isnull().any()
        ]

        st.markdown("## Dataset Information")

        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", f"{df.shape[0]:,}")

        c2.metric(
            "Eligible Numerical Columns",
            len(eligible_cols)
        )

        c3.metric(
            "Columns with Missing Values",
            len(missing_numeric_cols)
        )

        st.markdown("---")

        show_heading("Algorithm Configuration")

        st.write(
            "Adjust the following parameters for the CWI-kNN algorithm."
        )

        max_features = max(1, len(eligible_cols) - 1)

        # Keep widget values synchronized with saved configuration.
        st.session_state.config_k = int(
            st.session_state.saved_config["k"]
        )

        st.session_state.config_top_n = min(
            max_features,
            max(1, int(st.session_state.saved_config["top_n"]))
        )

        st.session_state.config_iterations = int(
            st.session_state.saved_config["iterations"]
        )

        with st.form("imputation_config_form"):

            k_value = st.number_input(
                "Number of Nearest Neighbors (K)",
                min_value=1,
                max_value=20,
                step=1,
                key="config_k",
                help="Number of nearest observations used for estimation."
            )

            top_n = st.number_input(
                "Top Correlated Features",
                min_value=1,
                max_value=max_features,
                step=1,
                key="config_top_n",
                help="Number of relevant correlated features selected."
            )

            iterations = st.number_input(
                "Number of Iterations",
                min_value=1,
                max_value=10,
                step=1,
                key="config_iterations",
                help="Number of refinement rounds."
            )

            submitted = st.form_submit_button(
                "Save Configuration",
                use_container_width=True
            )

            if submitted:

                st.session_state.saved_config = {
                    "k": int(k_value),
                    "top_n": int(top_n),
                    "iterations": int(iterations)
                }

                st.success("Algorithm configuration saved!")

        config = st.session_state.saved_config

        st.markdown("---")

        show_heading("Selected Parameters")

        p1, p2, p3 = st.columns(3)

        p1.metric("K Neighbors", config["k"])

        p2.metric("Top Features", config["top_n"])

        p3.metric("Iterations", config["iterations"])

        st.markdown("---")

        show_heading("Columns Selected for Processing")

        if missing_numeric_cols:

            st.write(
                "The following eligible numerical columns "
                "contain missing values:"
            )

            for col in missing_numeric_cols:

                st.write(
                    f"• **{col}** — "
                    f"{df[col].isnull().sum()} missing values"
                )

        else:

            st.success(
                "No missing values were found in eligible "
                "numerical columns."
            )

        st.markdown("---")

        show_heading("CWI-kNN Workflow")

        workflow = [
            "Identify missing values",
            "Perform correlation analysis",
            "Select relevant correlated features",
            "Apply correlation-weighted similarity",
            "Find K nearest neighbors",
            "Estimate missing values",
            "Iteratively refine estimates"
        ]

        for i, step in enumerate(workflow, start=1):

            st.write(f"**Step {i}:** {step}")

        st.markdown("---")

        st.info(
            "The configuration is ready. Actual imputation "
            "will be enabled after the backend is connected."
        )

        st.button(
            "Run CWI-kNN Imputation",
            disabled=True,
            use_container_width=True,
            help="This will be enabled after backend integration."
        )


# =========================================================
# 10. IMPUTATION RESULTS
# =========================================================

elif page == "Imputation Results":

    st.title("📈 Imputation Results")

    st.write(
        "View the original dataset, imputed dataset "
        "and evaluation results."
    )

    if st.session_state.dataset is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.dataset

        eligible_cols = get_eligible_columns(df)

        before_missing = (
            int(df[eligible_cols].isnull().sum().sum())
            if eligible_cols else 0
        )

        st.markdown("## Dataset Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", f"{df.shape[0]:,}")

        c2.metric("Columns", f"{df.shape[1]:,}")

        c3.metric(
            "Eligible Numerical Columns",
            len(eligible_cols)
        )

        st.markdown("---")

        show_heading("Before and After Imputation")

        result1, result2 = st.columns(2)

        with result1:

            st.markdown("### Before Imputation")

            st.metric(
                "Missing Numerical Values",
                before_missing
            )

        with result2:

            st.markdown("### After Imputation")

            if st.session_state.imputed_dataset is not None:

                after_df = st.session_state.imputed_dataset

                if all(col in after_df.columns for col in eligible_cols):

                    after_missing = int(
                        after_df[eligible_cols].isnull().sum().sum()
                    )

                    st.metric(
                        "Missing Numerical Values",
                        after_missing
                    )

                else:

                    st.warning(
                        "The returned dataset does not contain "
                        "all expected numerical columns."
                    )

            else:

                st.metric(
                    "Missing Numerical Values",
                    "Pending"
                )

        st.markdown("---")

        show_heading("Imputation Status")

        if st.session_state.imputed_dataset is not None:

            imputed_df = st.session_state.imputed_dataset

            st.success("Imputed dataset is available.")

            st.dataframe(
                imputed_df.head(10),
                use_container_width=True
            )

            st.download_button(
                "⬇ Download Imputed Dataset",
                data=download_csv(imputed_df),
                file_name="CWI_kNN_imputed_dataset.csv",
                mime="text/csv"
            )

            if st.session_state.imputation_details is not None:

                st.write(
                    st.session_state.imputation_details
                )

        else:

            st.info(
                "Waiting for backend integration. "
                "The actual imputed dataset will appear here "
                "after the algorithm is connected and executed."
            )

            st.write("**Columns to be processed:**")

            if eligible_cols:

                for col in eligible_cols:

                    if df[col].isnull().any():

                        st.write(f"• {col}")

            else:

                st.write("No eligible numerical columns found.")

        st.markdown("---")

        show_heading("Evaluation Metrics")

        evaluation = st.session_state.evaluation_results

        rmse_value = get_metric_value(evaluation, "RMSE")
        mae_value = get_metric_value(evaluation, "MAE")

        m1, m2 = st.columns(2)

        with m1:

            st.markdown("### RMSE")

            st.write("Root Mean Square Error")

            if rmse_value is not None:
                st.metric("RMSE", f"{rmse_value:.4f}")
            else:
                st.metric("RMSE", "Pending")

            st.caption(
                "Measures the square root of the average "
                "squared difference between actual and predicted values."
            )

        with m2:

            st.markdown("### MAE")

            st.write("Mean Absolute Error")

            if mae_value is not None:
                st.metric("MAE", f"{mae_value:.4f}")
            else:
                st.metric("MAE", "Pending")

            st.caption(
                "Measures the average absolute difference "
                "between actual and predicted values."
            )

        st.latex(
            r"RMSE = \sqrt{\frac{1}{n}"
            r"\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
        )

        st.latex(
            r"MAE = \frac{1}{n}"
            r"\sum_{i=1}^{n}|y_i-\hat{y}_i|"
        )

        st.caption(
            "RMSE and MAE require known original values. "
            "For evaluation, observed values must be artificially "
            "hidden and then compared with their original values."
        )


# =========================================================
# 11. BENCHMARK & EVALUATION
# =========================================================

elif page == "Benchmark & Evaluation":

    st.title("📊 Benchmark & Evaluation")

    st.write(
        "Compare CWI-kNN with traditional missing-value "
        "imputation techniques."
    )

    st.markdown("## Benchmark Methods")

    methods = [
        (
            "CWI-kNN",
            "Uses target-specific correlation weighting, "
            "K-nearest neighbors and iterative refinement."
        ),
        (
            "Mean Imputation",
            "Replaces missing numerical values with the "
            "mean of the available values."
        ),
        (
            "Median Imputation",
            "Replaces missing numerical values with the "
            "median of the available values."
        ),
        (
            "Standard KNN",
            "Estimates missing values using the nearest "
            "observations based on distance."
        )
    ]

    for name, description in methods:

        with st.container(border=True):

            st.markdown(f"### {name}")

            st.write(description)

    st.markdown("---")

    show_heading("Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### RMSE")

        st.write("Root Mean Square Error")

        st.latex(
            r"RMSE = \sqrt{\frac{1}{n}"
            r"\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}"
        )

        st.caption(
            "Lower RMSE indicates better prediction accuracy."
        )

    with col2:

        st.markdown("### MAE")

        st.write("Mean Absolute Error")

        st.latex(
            r"MAE = \frac{1}{n}"
            r"\sum_{i=1}^{n}|y_i-\hat{y}_i|"
        )

        st.caption(
            "Lower MAE indicates smaller average prediction errors."
        )

    st.markdown("---")

    show_heading("Missing Data Experiment Settings")

    rates = [5, 10, 20, 30]

    st.write(
        "The planned missing-value experiment uses these rates:"
    )

    rate_columns = st.columns(4)

    for col, rate in zip(rate_columns, rates):

        col.metric("Missing Rate", f"{rate}%")

    st.caption(
        "The experiment artificially hides a percentage of known "
        "values so that predictions can be compared with their "
        "original ground-truth values."
    )

    st.markdown("---")

    show_heading("Benchmark Results")

    if st.session_state.evaluation_results is not None:

        results = get_evaluation_table(
            st.session_state.evaluation_results
        )

        if not results.empty:

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "Lower RMSE and MAE values indicate better performance."
            )

        else:

            st.warning(
                "Evaluation data was received, but it could not "
                "be converted into a displayable table. "
                "Check the backend result format."
            )

    else:

        st.info(
            "Benchmark results will be displayed here after "
            "the backend evaluation module is integrated."
        )

        placeholder = pd.DataFrame({
            "Method": [
                "CWI-kNN",
                "Mean",
                "Median",
                "Standard KNN"
            ],
            "RMSE": [
                "Pending",
                "Pending",
                "Pending",
                "Pending"
            ],
            "MAE": [
                "Pending",
                "Pending",
                "Pending",
                "Pending"
            ]
        })

        st.dataframe(
            placeholder,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    st.markdown("### Interpretation")

    st.write("""
    RMSE and MAE are used to evaluate the accuracy of
    imputed values. A lower score indicates that the
    estimated values are closer to the original values.

    CWI-kNN will be compared against Mean, Median and
    Standard KNN methods to understand its performance.
    """)


# =========================================================
# 12. EXPERIMENT HISTORY
# =========================================================

elif page == "Experiment History":

    st.title("🕒 Experiment History")

    st.write(
        "View previous imputation experiments and their configurations."
    )

    if st.session_state.experiment_history:

        history_df = pd.DataFrame(
            st.session_state.experiment_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Experiment History",
            data=download_csv(history_df),
            file_name="CWI_kNN_experiment_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No completed experiments are available yet. "
            "Experiment records will appear here after "
            "the backend is connected and records are saved."
        )

        st.markdown("### Experiment Details to Be Recorded")

        details = [
            "Dataset name",
            "Date and time",
            "Number of rows and columns",
            "Missing-value percentage",
            "K-nearest neighbors value",
            "Number of selected features",
            "Number of iterations",
            "RMSE and MAE results"
        ]

        for item in details:

            st.write(f"• {item}")


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <b>CWI-kNN | Missing Value Imputation System</b><br>
    Correlation-Weighted Iterative K-Nearest Neighbors<br>
    Developed using Python and Streamlit
</div>
""", unsafe_allow_html=True)
