import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

# --- Configuration ---
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Assume the data file name is 'CC GENERAL.csv' based on typical analysis
DATA_PATH = "CC GENERAL.csv"

# --- 1. Data Loading and Preprocessing (Cached for performance) ---

@st.cache_data
def load_data(path):
    """Loads the dataset and handles file errors."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file '{path}' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_data
def preprocess_and_transform(df_original):
    """Performs cleaning, scaling, and UMAP dimensionality reduction."""
    df = df_original.copy()

    # 1. Handle missing values
    # We attempt imputation for known columns (MIN_PAYMENTS, CREDIT_LIMIT)
    missing_cols = ['MIN_PAYMENTS', 'CREDIT_LIMIT']
    for col in missing_cols:
        if col in df.columns:
            # FIX: Avoid FutureWarning by using direct assignment instead of inplace=True
            df[col] = df[col].fillna(df[col].median())
        else:
            # Log a warning if a common expected column is missing
            st.warning(f"Column '{col}' not found in data. Skipping imputation for this column.")

    # 2. Drop CUST_ID and capture feature names
    if 'CUST_ID' in df.columns:
        df_clustering = df.drop('CUST_ID', axis=1)
    else:
        df_clustering = df.copy()

    # 3. Handle remaining NaNs (if any, in other columns) by dropping rows.
    # This ensures no NaNs reach StandardScaler or UMAP.
    rows_before_drop = len(df_clustering)
    df_clustering.dropna(inplace=True)
    rows_after_drop = len(df_clustering)
    if rows_after_drop < rows_before_drop:
        st.info(f"Dropped {rows_before_drop - rows_after_drop} rows with remaining missing values to ensure model compatibility.")
        
    if df_clustering.empty:
        st.error("After cleaning, the dataset is empty. Check your input data for excessive missing values.")
        return pd.DataFrame(), pd.DataFrame(), []

    # 4. Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustering)
    df_scaled = pd.DataFrame(scaled_data, index=df_clustering.index, columns=df_clustering.columns)

    # 5. UMAP Dimensionality Reduction 
    # FIX: Explicitly set n_jobs=1 to suppress the UserWarning when using random_state
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42, n_jobs=1)
    umap_coords = umap_model.fit_transform(scaled_data)
    df_umap = pd.DataFrame(umap_coords, index=df_clustering.index, columns=['UMAP_1', 'UMAP_2'])

    return df_scaled, df_umap, df_clustering.columns

# --- 2. Clustering Model ---

def perform_clustering(df_scaled, n_clusters, method='KMeans'):
    """Applies K-Means and returns labels and Silhouette Score."""
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    else:
        return None, None

    labels = model.fit_predict(df_scaled)

    # Calculate silhouette score (handle potential single cluster failure)
    try:
        score = silhouette_score(df_scaled, labels)
    except ValueError:
        score = -1

    return labels, score

# --- 3. Visualization ---

def plot_clusters(df_umap, labels, n_clusters):
    """Generates the scatter plot of UMAP coordinates with cluster colors."""
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        df_umap['UMAP_1'],
        df_umap['UMAP_2'],
        c=labels,
        cmap='Spectral',
        s=30,
        alpha=0.7
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Segments", loc="lower left")
    ax.add_artist(legend1)
    ax.set_title(f'Customer Segments in UMAP Space (K={n_clusters})', fontsize=14)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    st.pyplot(fig)


# FIX 1: Add df_umap as a parameter to the function
def plot_feature_means(df_original, labels, features, df_umap):
    """Generates a heatmap of scaled feature means for cluster profiling."""
    
    # The NameError is resolved here
    # Ensure the original DataFrame index aligns with the labels index
    df_with_labels = df_original.loc[df_original.index.isin(df_umap.index)].copy()
    df_with_labels['Cluster'] = labels
    
    # Calculate means only for the defined features
    cluster_means = df_with_labels.groupby('Cluster')[features].mean().T

    # Normalize feature means (Z-score scaling across clusters for comparison)
    scaler_viz = StandardScaler()
    scaled_means = scaler_viz.fit_transform(cluster_means)
    cluster_means_scaled = pd.DataFrame(
        scaled_means,
        index=cluster_means.index,
        columns=[f'Cluster {c}' for c in cluster_means.columns]
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        cluster_means_scaled,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0, # Center the color map at 0 (average)
        linewidths=.5,
        linecolor='black',
        cbar_kws={'label': 'Z-Score (Relative Importance)'}
    )
    ax.set_title('Cluster Profile: Feature Means (Z-Score Scaled)', fontsize=16)
    st.pyplot(fig)

# =========================================================================
# --- MAIN APPLICATION LOGIC ---
# =========================================================================

def main():
    st.title("💳 Credit Card Customer Segmentation Dashboard")
    st.markdown("Use this interactive tool to explore different customer segments found using UMAP for dimensionality reduction and K-Means clustering.")

    # --- Load Data ---
    df_original = load_data(DATA_PATH)

    if df_original is None:
        st.warning("Please upload the 'CC GENERAL.csv' file to the same directory as this app.py file to run the analysis.")
        st.stop()

    # --- Preprocess and Transform Data ---
    with st.spinner('Running the full data pipeline (Cleaning, Scaling, UMAP Reduction)...'):
        # df_umap is defined here
        df_scaled, df_umap, features = preprocess_and_transform(df_original)

    # Stop if the cleaning resulted in an empty DataFrame
    if df_scaled.empty:
        st.stop()
        
    st.sidebar.success("✅ Data Pipeline Complete")
    st.markdown("---")


    # --- Sidebar Configuration ---
    st.sidebar.header("Segmentation Control")

    # K-Means Parameter Slider
    n_clusters = st.sidebar.slider(
        'Select Number of Customer Segments (K)',
        min_value=2,
        max_value=8,
        value=3, # Starting point
        step=1
    )

    # --- Perform Clustering ---
    st.subheader(f"Results for K-Means with K = {n_clusters}")

    # Run K-Means
    labels, score = perform_clustering(df_scaled, n_clusters, method='KMeans')

    # --- Results and Metrics ---
    col_metrics, col_plot = st.columns([1, 2])

    with col_metrics:
        st.markdown("**Clustering Performance**")
        st.metric(
            label="Silhouette Score",
            value=f"{score:.4f}",
            delta="Score near 1 indicates better, denser clusters.",
            delta_color="off"
        )
        st.info("Adjust the 'Segments (K)' in the sidebar to maximize the Silhouette Score while maintaining meaningful segments.")


    # --- Visualization ---

    with col_plot:
        # Plot 1: UMAP Visualization
        st.markdown("**1. Visualizing Segments in Reduced UMAP Space**")
        plot_clusters(df_umap, labels, n_clusters)

    st.markdown("---")

    # Plot 2: Cluster Profiling
    st.markdown("## 2. Segment Profile Analysis")
    st.markdown(
        """
        This heatmap compares the average feature value for each segment against the overall customer average (Z-Score scaled).
        This helps define the *persona* of each segment:
        * **Red (Positive Z-Score):** Features significantly *higher* than average.
        * **Blue (Negative Z-Score):** Features significantly *lower* than average.
        """
    )
    # FIX 2: Pass df_umap to the function call
    plot_feature_means(df_original, labels, df_scaled.columns, df_umap)


if __name__ == "__main__":
    main()
