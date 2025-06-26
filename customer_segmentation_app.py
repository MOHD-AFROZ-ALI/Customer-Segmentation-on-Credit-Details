import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from streamlit_option_menu import option_menu
import io
import base64
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI/UX
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--primary-color);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    .metric-card h3 {
        color: var(--primary-color);
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: 600;
    }

    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dark-bg);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, var(--light-bg) 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid var(--primary-color);
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    .section-header h2 {
        margin: 0;
        color: var(--primary-color);
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(31, 119, 180, 0.4);
        background: linear-gradient(135deg, #1a6ba8 0%, #e6700d 100%);
    }

    /* Alert styling */
    .alert {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 500;
    }

    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
    }

    .alert-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
    }

    .alert-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
    }

    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 12px;
        border: 3px dashed var(--primary-color);
        background: linear-gradient(135deg, var(--light-bg) 0%, #ffffff 100%);
        padding: 2rem;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, #ffffff 0%, var(--light-bg) 100%);
    }

    /* Enhanced form styling */
    .stSelectbox > div > div, .stNumberInput > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }

    .stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 10px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: var(--light-bg);
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(31, 119, 180, 0.1);
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }

    /* Custom insight cards */
    .insight-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 5px solid var(--info-color);
    }

    .insight-card h4 {
        color: var(--info-color);
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Risk level indicators */
    .risk-low { color: var(--success-color); font-weight: 600; }
    .risk-medium { color: var(--warning-color); font-weight: 600; }
    .risk-high { color: var(--danger-color); font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'clusters' not in st.session_state:
        st.session_state.clusters = None
    if 'pca_data' not in st.session_state:
        st.session_state.pca_data = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'pca_model' not in st.session_state:
        st.session_state.pca_model = None
    if 'cluster_model' not in st.session_state:
        st.session_state.cluster_model = None

# Data generation and preprocessing functions
@st.cache_data
def load_sample_data():
    """Load sample credit card dataset"""
    np.random.seed(42)
    n_samples = 1000

    data = {}
    data['BALANCE'] = np.random.exponential(scale=2000, size=n_samples)
    data['BALANCE'] = np.clip(data['BALANCE'], 0, 20000)
    data['BALANCE_FREQUENCY'] = np.random.beta(2, 2, size=n_samples)

    purchases_base = data['BALANCE'] * np.random.uniform(0.1, 2.0, size=n_samples)
    data['PURCHASES'] = purchases_base + np.random.exponential(scale=500, size=n_samples)
    data['PURCHASES'] = np.clip(data['PURCHASES'], 0, 50000)

    data['ONEOFF_PURCHASES'] = data['PURCHASES'] * np.random.uniform(0.1, 0.7, size=n_samples)
    data['INSTALLMENTS_PURCHASES'] = data['PURCHASES'] - data['ONEOFF_PURCHASES'] + np.random.exponential(scale=200, size=n_samples)
    data['INSTALLMENTS_PURCHASES'] = np.clip(data['INSTALLMENTS_PURCHASES'], 0, None)

    cash_advance_users = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    data['CASH_ADVANCE'] = cash_advance_users * np.random.exponential(scale=1000, size=n_samples)
    data['CASH_ADVANCE'] = np.clip(data['CASH_ADVANCE'], 0, 15000)

    data['PURCHASES_FREQUENCY'] = np.random.beta(3, 2, size=n_samples)
    data['ONEOFF_PURCHASES_FREQUENCY'] = data['PURCHASES_FREQUENCY'] * np.random.uniform(0.3, 0.8, size=n_samples)
    data['PURCHASES_INSTALLMENTS_FREQUENCY'] = data['PURCHASES_FREQUENCY'] * np.random.uniform(0.2, 0.9, size=n_samples)
    data['CASH_ADVANCE_FREQUENCY'] = (data['CASH_ADVANCE'] > 0).astype(float) * np.random.beta(1, 4, size=n_samples)
    data['CASH_ADVANCE_TRX'] = np.random.poisson(lam=data['CASH_ADVANCE_FREQUENCY'] * 10, size=n_samples)
    data['PURCHASES_TRX'] = np.random.poisson(lam=data['PURCHASES_FREQUENCY'] * 20, size=n_samples)

    data['CREDIT_LIMIT'] = data['BALANCE'] * np.random.uniform(1.5, 5.0, size=n_samples) + np.random.normal(loc=5000, scale=2000, size=n_samples)
    data['CREDIT_LIMIT'] = np.clip(data['CREDIT_LIMIT'], 1000, 30000)

    data['PAYMENTS'] = (data['BALANCE'] + data['PURCHASES'] * 0.3) * np.random.uniform(0.8, 1.2, size=n_samples)
    data['PAYMENTS'] = np.clip(data['PAYMENTS'], 0, None)

    data['MINIMUM_PAYMENTS'] = data['PAYMENTS'] * np.random.uniform(0.05, 0.3, size=n_samples)
    data['PRC_FULL_PAYMENT'] = np.random.beta(2, 3, size=n_samples)
    data['TENURE'] = np.random.randint(6, 13, size=n_samples)

    df = pd.DataFrame(data)
    df.insert(0, 'CUSTOMER_ID', [f'CUST_{i:06d}' for i in range(1, len(df) + 1)])

    # Add some missing values
    missing_cols = ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df

def validate_data(df):
    """Validate uploaded data"""
    required_columns = [
        'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
        'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
        'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
        'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    if df.shape[0] < 10:
        return False, "Dataset must have at least 10 rows"

    return True, "Data validation successful"

def preprocess_data(df):
    """Preprocess data for clustering"""
    if 'CUSTOMER_ID' in df.columns:
        df_processed = df.drop('CUSTOMER_ID', axis=1)
    else:
        df_processed = df.copy()

    df_processed = df_processed.fillna(df_processed.median())
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed = df_processed[numeric_columns]

    return df_processed

# Clustering functions
def perform_kmeans_clustering(data, n_clusters=4):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters, dbscan

def find_optimal_clusters(data, max_clusters=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, clusters))

    return K_range, inertias, silhouette_scores

# PCA functions
def perform_pca(data, n_components=3):
    """Perform PCA analysis"""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca_data, pca

# Visualization functions
def create_cluster_3d_plot(pca_data, clusters, title="3D Cluster Visualization"):
    """Create 3D scatter plot of clusters"""
    fig = go.Figure(data=go.Scatter3d(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        z=pca_data[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=clusters,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Cluster", tickmode="linear"),
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        text=[f'Cluster {c}' for c in clusters],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color='#1f77b4')),
        scene=dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white", showbackground=True, zerolinecolor="white")
        ),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig

def create_elbow_plot(K_range, inertias, silhouette_scores):
    """Create elbow method plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method for Optimal K', 'Silhouette Score Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(
            x=list(K_range), 
            y=inertias, 
            mode='lines+markers', 
            name='Inertia',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(K_range), 
            y=silhouette_scores, 
            mode='lines+markers', 
            name='Silhouette Score',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8, color='#ff7f0e')
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig.update_layout(height=500, showlegend=False, title_x=0.5)

    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    corr_matrix = data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text='Feature Correlation Matrix', x=0.5, font=dict(size=18, color='#1f77b4')),
        width=900,
        height=700,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )

    return fig

def create_pca_explained_variance_plot(pca):
    """Create PCA explained variance plot"""
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        y=explained_variance_ratio,
        name='Individual Variance',
        marker_color='lightblue',
        text=[f'{val:.1%}' for val in explained_variance_ratio],
        textposition='auto'
    ))

    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumulative_variance_ratio))],
        y=cumulative_variance_ratio,
        mode='lines+markers',
        name='Cumulative Variance',
        yaxis='y2',
        line=dict(color='red', width=3),
        marker=dict(size=8, color='red')
    ))

    fig.update_layout(
        title=dict(text='PCA Explained Variance Analysis', x=0.5, font=dict(size=18, color='#1f77b4')),
        xaxis_title='Principal Components',
        yaxis_title='Individual Explained Variance Ratio',
        yaxis2=dict(
            title='Cumulative Explained Variance',
            overlaying='y',
            side='right',
            tickformat='.0%'
        ),
        height=500,
        showlegend=True
    )

    return fig

def create_cluster_distribution_plot(clusters):
    """Create cluster distribution plot"""
    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=[colors[i % len(colors)] for i in range(len(cluster_counts))],
            text=cluster_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Customers: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[count/sum(cluster_counts.values)*100 for count in cluster_counts.values]
        )
    ])

    fig.update_layout(
        title=dict(text='Customer Distribution Across Clusters', x=0.5, font=dict(size=18, color='#1f77b4')),
        xaxis_title='Clusters',
        yaxis_title='Number of Customers',
        height=500,
        showlegend=False
    )

    return fig

def create_silhouette_plot(data, clusters):
    """Create silhouette analysis plot"""
    silhouette_avg = silhouette_score(data, clusters)
    sample_silhouette_values = silhouette_samples(data, clusters)

    fig = go.Figure()

    y_lower = 10
    for i in range(len(np.unique(clusters))):
        cluster_silhouette_values = sample_silhouette_values[clusters == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=list(range(y_lower, y_upper)),
            fill='tozeroy',
            mode='none',
            name=f'Cluster {i}',
            fillcolor=color,
            hovertemplate=f'<b>Cluster {i}</b><br>Silhouette Score: %{{x:.3f}}<extra></extra>'
        ))

        y_lower = y_upper + 10

    fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red", 
                  annotation_text=f"Average Score: {silhouette_avg:.3f}")

    fig.update_layout(
        title=dict(text='Silhouette Analysis for Cluster Validation', x=0.5, font=dict(size=18, color='#1f77b4')),
        xaxis_title='Silhouette Coefficient Values',
        yaxis_title='Cluster Label',
        height=600,
        showlegend=True
    )

    return fig

def create_feature_importance_plot(data, feature_names):
    """Create feature importance plot based on variance"""
    feature_variance = data.var().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=feature_variance.values,
        y=feature_variance.index,
        orientation='h',
        marker_color='lightcoral',
        text=[f'{val:.2f}' for val in feature_variance.values],
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(text='Feature Variance Analysis', x=0.5, font=dict(size=18, color='#1f77b4')),
        xaxis_title='Variance',
        yaxis_title='Features',
        height=600,
        showlegend=False
    )

    return fig

def create_cluster_comparison_radar(profiles):
    """Create radar chart comparing cluster profiles"""
    features = ['avg_balance', 'avg_purchases', 'avg_cash_advance', 'purchase_frequency', 'cash_advance_frequency']
    feature_labels = ['Balance', 'Purchases', 'Cash Advance', 'Purchase Freq', 'Cash Advance Freq']

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (cluster_id, profile) in enumerate(profiles.items()):
        values = []
        for feature in features:
            # Normalize values to 0-1 scale for radar chart
            if feature in profile:
                val = profile[feature]
                if feature == 'avg_balance':
                    val = min(val / 10000, 1)  # Normalize balance
                elif feature == 'avg_purchases':
                    val = min(val / 5000, 1)   # Normalize purchases
                elif feature == 'avg_cash_advance':
                    val = min(val / 3000, 1)   # Normalize cash advance
                else:
                    val = min(val, 1)          # Frequencies already 0-1
                values.append(val)
            else:
                values.append(0)

        values += values[:1]  # Complete the circle
        labels = feature_labels + [feature_labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f'Cluster {cluster_id}',
            line_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=dict(text='Cluster Profile Comparison', x=0.5, font=dict(size=18, color='#1f77b4')),
        height=600,
        showlegend=True
    )

    return fig

def create_customer_journey_plot(data, clusters):
    """Create customer journey visualization based on tenure and activity"""
    df_plot = data.copy()
    df_plot['Cluster'] = clusters

    fig = px.scatter(
        df_plot, 
        x='TENURE', 
        y='PURCHASES_FREQUENCY',
        color='Cluster',
        size='BALANCE',
        hover_data=['PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT'],
        title='Customer Journey: Tenure vs Activity',
        labels={
            'TENURE': 'Customer Tenure (months)',
            'PURCHASES_FREQUENCY': 'Purchase Frequency',
            'BALANCE': 'Account Balance'
        }
    )

    fig.update_layout(
        title=dict(text='Customer Journey Analysis', x=0.5, font=dict(size=18, color='#1f77b4')),
        height=600,
        showlegend=True
    )

    return fig

# Business insights functions
def generate_cluster_profiles(data, clusters):
    """Generate cluster profiles and insights"""
    df_with_clusters = data.copy()
    df_with_clusters['Cluster'] = clusters

    profiles = {}
    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue

        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]

        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_with_clusters) * 100,
            'avg_balance': cluster_data['BALANCE'].mean(),
            'avg_purchases': cluster_data['PURCHASES'].mean(),
            'avg_cash_advance': cluster_data['CASH_ADVANCE'].mean(),
            'avg_credit_limit': cluster_data['CREDIT_LIMIT'].mean(),
            'avg_payments': cluster_data['PAYMENTS'].mean(),
            'purchase_frequency': cluster_data['PURCHASES_FREQUENCY'].mean(),
            'cash_advance_frequency': cluster_data['CASH_ADVANCE_FREQUENCY'].mean(),
            'tenure': cluster_data['TENURE'].mean(),
            'balance_frequency': cluster_data['BALANCE_FREQUENCY'].mean(),
            'prc_full_payment': cluster_data['PRC_FULL_PAYMENT'].mean(),
            'avg_oneoff_purchases': cluster_data['ONEOFF_PURCHASES'].mean(),
            'avg_installments_purchases': cluster_data['INSTALLMENTS_PURCHASES'].mean(),
            'avg_purchases_trx': cluster_data['PURCHASES_TRX'].mean(),
            'avg_cash_advance_trx': cluster_data['CASH_ADVANCE_TRX'].mean(),
            'avg_minimum_payments': cluster_data['MINIMUM_PAYMENTS'].mean()
        }

        profiles[cluster_id] = profile

    return profiles

def generate_business_insights(profiles):
    """Generate business insights and recommendations"""
    insights = {}

    for cluster_id, profile in profiles.items():
        if profile['avg_balance'] > 5000 and profile['avg_purchases'] > 3000:
            segment_type = "💎 High-Value Customers"
            description = "Premium customers with high balance and spending power"
            recommendations = [
                "🎯 Offer premium credit card products with exclusive benefits",
                "🏆 Provide VIP customer service and dedicated relationship managers",
                "💰 Focus on retention strategies with loyalty rewards",
                "📈 Cross-sell investment and wealth management products",
                "🎁 Exclusive access to premium events and experiences"
            ]
            risk_level = "Low"
            marketing_strategy = "Premium positioning with personalized services"

        elif profile['avg_cash_advance'] > 2000 and profile['cash_advance_frequency'] > 0.3:
            segment_type = "⚠️ Cash-Dependent Customers"
            description = "Customers heavily reliant on cash advances - potential financial stress"
            recommendations = [
                "🔍 Monitor closely for signs of financial distress",
                "💡 Offer financial counseling and budgeting services",
                "⚖️ Consider credit limit adjustments based on risk assessment",
                "🏦 Provide alternative lending products with better terms",
                "📞 Proactive outreach for financial wellness programs"
            ]
            risk_level = "High"
            marketing_strategy = "Supportive approach with financial education focus"

        elif profile['purchase_frequency'] > 0.8 and profile['avg_purchases'] > 1000:
            segment_type = "🛍️ Active Spenders"
            description = "Frequent purchasers with consistent spending habits"
            recommendations = [
                "💳 Offer attractive cashback and rewards programs",
                "🏪 Promote merchant partnerships and exclusive discounts",
                "📊 Target with personalized offers based on spending patterns",
                "⬆️ Gradually increase credit limits to support spending growth",
                "🎯 Focus on category-specific rewards (dining, shopping, travel)"
            ]
            risk_level = "Medium"
            marketing_strategy = "Engagement-focused with rewards optimization"

        elif profile['avg_balance'] < 1000 and profile['avg_purchases'] < 500:
            segment_type = "😴 Low-Activity Customers"
            description = "Customers with minimal card usage and low engagement"
            recommendations = [
                "🚀 Implement activation campaigns with incentives",
                "🎁 Offer introductory bonuses and welcome offers",
                "📚 Provide financial education and card usage benefits",
                "💸 Consider account maintenance fees for inactive accounts",
                "📱 Promote digital banking features and convenience"
            ]
            risk_level = "Medium"
            marketing_strategy = "Activation and engagement campaigns"

        elif profile['prc_full_payment'] > 0.7 and profile['avg_balance'] > 2000:
            segment_type = "🏛️ Responsible Users"
            description = "Customers who pay in full regularly with good balances"
            recommendations = [
                "🎖️ Reward responsible behavior with better terms",
                "📈 Offer credit limit increases and premium products",
                "💼 Cross-sell other financial products (loans, investments)",
                "🤝 Use as referral sources for new customer acquisition",
                "🏆 Provide loyalty benefits and exclusive offers"
            ]
            risk_level = "Low"
            marketing_strategy = "Relationship building and cross-selling"

        else:
            segment_type = "⚖️ Balanced Customers"
            description = "Customers with moderate and balanced usage patterns"
            recommendations = [
                "📊 Maintain current service levels and monitor trends",
                "🔍 Watch for usage pattern changes and opportunities",
                "🎯 Offer targeted promotions based on spending categories",
                "😊 Focus on customer satisfaction and service quality",
                "📈 Gradual engagement increase through relevant offers"
            ]
            risk_level = "Low"
            marketing_strategy = "Steady relationship maintenance with growth opportunities"

        insights[cluster_id] = {
            'segment_type': segment_type,
            'description': description,
            'recommendations': recommendations,
            'risk_level': risk_level,
            'marketing_strategy': marketing_strategy,
            'profile': profile
        }

    return insights

def generate_individual_prediction(customer_data, scaler, pca_model, cluster_model):
    """Generate prediction for individual customer"""
    try:
        # Convert to DataFrame if it's a dictionary
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()

        # Preprocess the data
        customer_processed = preprocess_data(customer_df)

        # Scale the data
        customer_scaled = scaler.transform(customer_processed)

        # Apply PCA
        customer_pca = pca_model.transform(customer_scaled)

        # Predict cluster
        cluster_prediction = cluster_model.predict(customer_scaled)[0]

        return {
            'cluster': cluster_prediction,
            'pca_coordinates': customer_pca[0],
            'processed_data': customer_processed.iloc[0].to_dict()
        }

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Download and utility functions
def create_download_link(df, filename, link_text):
    """Create download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none; background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600;">{link_text}</a>'
    return href

def create_report_download(insights, profiles):
    """Create comprehensive report for download"""

     # Convert keys to strings to ensure JSON serialization compatibility
    insights_str_keys = {str(k): v for k, v in insights.items()}
    profiles_str_keys = {str(k): v for k, v in profiles.items()}
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_customers': sum(profile['size'] for profile in profiles.values()),
        'number_of_clusters': len(profiles),
        'cluster_insights': insights_str_keys,
        'cluster_profiles': profiles_str_keys
    }

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(report_json.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="customer_segmentation_report.json" style="text-decoration: none; background-color: #ff7f0e; color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600;">📊 Download Full Report (JSON)</a>'
    return href

def predict_customer_segment(customer_features, scaler, cluster_model):
    """Predict customer segment for new data"""
    try:
        # Ensure all required features are present
        required_features = [
            'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
            'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
            'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
        ]

        # Create feature vector
        feature_vector = []
        for feature in required_features:
            if feature in customer_features:
                feature_vector.append(customer_features[feature])
            else:
                feature_vector.append(0)  # Default value for missing features

        # Scale features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        scaled_features = scaler.transform(feature_vector)

        # Predict cluster
        cluster_prediction = cluster_model.predict(scaled_features)[0]

        return cluster_prediction

    except Exception as e:
        return None

def format_currency(value):
    """Format currency values"""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage values"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'Low': '#2ca02c',
        'Medium': '#ff7f0e', 
        'High': '#d62728'
    }
    return colors.get(risk_level, '#1f77b4')

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Main application --------------------------------------------------
def main():
    load_css()
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced Customer Segmentation Dashboard</h1>
        <p>Comprehensive analytics platform for customer behavior analysis and business intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["📊 Dashboard", "📁 Data Upload", "👤 Individual Prediction", "🔄 Batch Processing", "📈 Visualizations", "💡 Business Insights"],
        icons=["graph-up", "cloud-upload", "person", "gear", "bar-chart", "lightbulb"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1f77b4", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )
    
    # Dashboard Overview
    if selected == "📊 Dashboard":
        st.markdown('<div class="section-header"><h2>📊 Dashboard Overview</h2></div>', unsafe_allow_html=True)
        
        # Load sample data if no data exists
        if st.session_state.data is None:
            st.session_state.data = load_sample_data()
            st.session_state.processed_data = preprocess_data(st.session_state.data)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <div class="metric-value">{len(st.session_state.data):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_balance = st.session_state.data['BALANCE'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Balance</h3>
                <div class="metric-value">{format_currency(avg_balance)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_purchases = st.session_state.data['PURCHASES'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Purchases</h3>
                <div class="metric-value">{format_currency(avg_purchases)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_credit_limit = st.session_state.data['CREDIT_LIMIT'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Credit Limit</h3>
                <div class="metric-value">{format_currency(avg_credit_limit)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick clustering analysis
        st.markdown("### 🚀 Quick Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔄 Perform Quick Clustering Analysis", key="quick_analysis"):
                with st.spinner("Performing clustering analysis..."):
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(st.session_state.processed_data)
                    
                    # Perform PCA
                    pca_data, pca_model = perform_pca(scaled_data, n_components=3)
                    
                    # Perform clustering
                    clusters, kmeans_model = perform_kmeans_clustering(scaled_data, n_clusters=4)
                    
                    # Store in session state
                    st.session_state.clusters = clusters
                    st.session_state.pca_data = pca_data
                    st.session_state.scaler = scaler
                    st.session_state.pca_model = pca_model
                    st.session_state.cluster_model = kmeans_model
                    
                    st.success("✅ Quick analysis completed! Explore other sections for detailed insights.")
        
        with col2:
            st.markdown("""
            <div class="alert alert-info">
                <strong>💡 Quick Start:</strong><br>
                Click the analysis button to automatically segment your customers and unlock insights!
            </div>
            """, unsafe_allow_html=True)
        
        # Display results if analysis has been performed
        if st.session_state.clusters is not None:
            st.markdown("### 📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_3d = create_cluster_3d_plot(st.session_state.pca_data, st.session_state.clusters, "Customer Segments (3D View)")
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with col2:
                fig_dist = create_cluster_distribution_plot(st.session_state.clusters)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
    
    # Data Upload Section
    elif selected == "📁 Data Upload":
        st.markdown('<div class="section-header"><h2>📁 Data Upload & Validation</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a CSV file with customer data",
                type="csv",
                help="Upload a CSV file containing customer credit card data with 17 required features"
            )
            
            if uploaded_file is not None:
                try:
                    # Read uploaded file
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate data
                    is_valid, message = validate_data(df)
                    
                    if is_valid:
                        st.markdown(f'<div class="alert alert-success">✅ {message}</div>', unsafe_allow_html=True)
                        
                        # Store data in session state
                        st.session_state.data = df
                        st.session_state.processed_data = preprocess_data(df)
                        
                        # Display data info
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("#### 📊 Dataset Information")
                            st.write(f"**Shape:** {df.shape}")
                            st.write(f"**Columns:** {len(df.columns)}")
                            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
                        
                        with col_b:
                            st.markdown("#### 🔍 Data Types")
                            st.write(df.dtypes)
                        
                        # Data preview
                        st.markdown("#### 👀 Data Preview")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Data statistics
                        st.markdown("#### 📈 Statistical Summary")
                        st.dataframe(df.describe(), use_container_width=True)
                        
                    else:
                        st.markdown(f'<div class="alert alert-warning">⚠️ {message}</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f'<div class="alert alert-warning">❌ Error reading file: {str(e)}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-card">
                <h4>📋 Required Features</h4>
                <p>Your CSV file must contain these 17 columns:</p>
                <ul style="font-size: 0.9rem;">
                    <li>BALANCE</li>
                    <li>BALANCE_FREQUENCY</li>
                    <li>PURCHASES</li>
                    <li>ONEOFF_PURCHASES</li>
                    <li>INSTALLMENTS_PURCHASES</li>
                    <li>CASH_ADVANCE</li>
                    <li>PURCHASES_FREQUENCY</li>
                    <li>ONEOFF_PURCHASES_FREQUENCY</li>
                    <li>PURCHASES_INSTALLMENTS_FREQUENCY</li>
                    <li>CASH_ADVANCE_FREQUENCY</li>
                    <li>CASH_ADVANCE_TRX</li>
                    <li>PURCHASES_TRX</li>
                    <li>CREDIT_LIMIT</li>
                    <li>PAYMENTS</li>
                    <li>MINIMUM_PAYMENTS</li>
                    <li>PRC_FULL_PAYMENT</li>
                    <li>TENURE</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample data option
        st.markdown("### 🎯 Use Sample Data")
        if st.button("📊 Load Sample Dataset", key="load_sample"):
            st.session_state.data = load_sample_data()
            st.session_state.processed_data = preprocess_data(st.session_state.data)
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    
    # Individual Prediction Section
    elif selected == "👤 Individual Prediction":
        st.markdown('<div class="section-header"><h2>👤 Individual Customer Prediction</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.markdown('<div class="alert alert-warning">⚠️ Please upload data or load sample data first!</div>', unsafe_allow_html=True)
            return
        
        # Check if clustering has been performed
        if st.session_state.clusters is None:
            st.markdown('<div class="alert alert-info">💡 Please perform clustering analysis first in the Dashboard or Visualizations section.</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Perform Clustering Analysis", key="individual_clustering"):
                with st.spinner("Performing clustering analysis..."):
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(st.session_state.processed_data)
                    pca_data, pca_model = perform_pca(scaled_data, n_components=3)
                    clusters, kmeans_model = perform_kmeans_clustering(scaled_data, n_clusters=4)
                    
                    st.session_state.clusters = clusters
                    st.session_state.pca_data = pca_data
                    st.session_state.scaler = scaler
                    st.session_state.pca_model = pca_model
                    st.session_state.cluster_model = kmeans_model
                    
                    st.success("✅ Clustering analysis completed!")
                    st.rerun()
            return
        
        st.markdown("### 📝 Enter Customer Information")
        
        # Create input form
        with st.form("customer_input_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                balance = st.number_input("Balance ($)", min_value=0.0, value=2000.0, step=100.0)
                balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.8, 0.1)
                purchases = st.number_input("Total Purchases ($)", min_value=0.0, value=1500.0, step=100.0)
                oneoff_purchases = st.number_input("One-off Purchases ($)", min_value=0.0, value=800.0, step=50.0)
                installments_purchases = st.number_input("Installments Purchases ($)", min_value=0.0, value=700.0, step=50.0)
                cash_advance = st.number_input("Cash Advance ($)", min_value=0.0, value=0.0, step=100.0)
            
            with col2:
                purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.7, 0.1)
                oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.5, 0.1)
                purchases_installments_frequency = st.slider("Installments Purchases Frequency", 0.0, 1.0, 0.6, 0.1)
                cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0, 0.1)
                cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, value=0, step=1)
                purchases_trx = st.number_input("Purchase Transactions", min_value=0, value=15, step=1)
            
            with col3:
                credit_limit = st.number_input("Credit Limit ($)", min_value=1000.0, value=8000.0, step=500.0)
                payments = st.number_input("Payments ($)", min_value=0.0, value=2200.0, step=100.0)
                minimum_payments = st.number_input("Minimum Payments ($)", min_value=0.0, value=200.0, step=50.0)
                prc_full_payment = st.slider("Percentage Full Payment", 0.0, 1.0, 0.3, 0.1)
                tenure = st.selectbox("Tenure (months)", options=list(range(6, 13)), index=6)
            
            submitted = st.form_submit_button("🔮 Predict Customer Segment", use_container_width=True)
        
        if submitted:
            # Create customer data dictionary
            customer_data = {
                'BALANCE': balance,
                'BALANCE_FREQUENCY': balance_frequency,
                'PURCHASES': purchases,
                'ONEOFF_PURCHASES': oneoff_purchases,
                'INSTALLMENTS_PURCHASES': installments_purchases,
                'CASH_ADVANCE': cash_advance,
                'PURCHASES_FREQUENCY': purchases_frequency,
                'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchases_frequency,
                'PURCHASES_INSTALLMENTS_FREQUENCY': purchases_installments_frequency,
                'CASH_ADVANCE_FREQUENCY': cash_advance_frequency,
                'CASH_ADVANCE_TRX': cash_advance_trx,
                'PURCHASES_TRX': purchases_trx,
                'CREDIT_LIMIT': credit_limit,
                'PAYMENTS': payments,
                'MINIMUM_PAYMENTS': minimum_payments,
                'PRC_FULL_PAYMENT': prc_full_payment,
                'TENURE': tenure
            }
            
            # Make prediction
            prediction_result = generate_individual_prediction(
                customer_data, 
                st.session_state.scaler, 
                st.session_state.pca_model, 
                st.session_state.cluster_model
            )
            
            if prediction_result:
                predicted_cluster = prediction_result['cluster']
                
                # Generate insights for this customer
                profiles = generate_cluster_profiles(st.session_state.processed_data, st.session_state.clusters)
                insights = generate_business_insights(profiles)
                
                # Display prediction results
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Cluster</h3>
                        <div class="metric-value">Cluster {predicted_cluster}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if predicted_cluster in insights:
                        insight = insights[predicted_cluster]
                        risk_color = get_risk_color(insight['risk_level'])
                        
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>{insight['segment_type']}</h4>
                            <p><strong>Description:</strong> {insight['description']}</p>
                            <p><strong>Risk Level:</strong> <span style="color: {risk_color};">{insight['risk_level']}</span></p>
                            <p><strong>Marketing Strategy:</strong> {insight['marketing_strategy']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display recommendations
                if predicted_cluster in insights:
                    st.markdown("### 💡 Personalized Recommendations")
                    recommendations = insights[predicted_cluster]['recommendations']
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
                
                # Visualize customer position
                st.markdown("### 📊 Customer Position Visualization")
                
                # Create a plot showing where this customer fits
                fig = create_cluster_3d_plot(st.session_state.pca_data, st.session_state.clusters, "Your Customer Position")
                
                # Add the new customer point
                customer_pca = prediction_result['pca_coordinates']
                fig.add_trace(go.Scatter3d(
                    x=[customer_pca[0]],
                    y=[customer_pca[1]],
                    z=[customer_pca[2]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name='Your Customer',
                    hovertemplate='<b>Your Customer</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
                ))
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Batch Processing Section
    elif selected == "🔄 Batch Processing":
        st.markdown('<div class="section-header"><h2>🔄 Batch Processing & Analysis</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.markdown('<div class="alert alert-warning">⚠️ Please upload data or load sample data first!</div>', unsafe_allow_html=True)
            return
        
        # Clustering Configuration
        st.markdown("### ⚙️ Clustering Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### K-Means Settings")
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            
            st.markdown("#### PCA Settings")
            n_components = st.slider("PCA Components", 2, 5, 3)
        
        with col2:
            st.markdown("#### DBSCAN Settings")
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.4, 0.1)
            min_samples = st.slider("Minimum Samples", 3, 20, 6)
        
        # Processing Options
        st.markdown("### 🎯 Processing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Run K-Means Analysis", key="kmeans_batch"):
                with st.spinner("Running K-Means clustering..."):
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(st.session_state.processed_data)
                    
                    # Perform PCA
                    pca_data, pca_model = perform_pca(scaled_data, n_components=n_components)
                    
                    # Perform K-means clustering
                    clusters, kmeans_model = perform_kmeans_clustering(scaled_data, n_clusters=n_clusters)
                    
                    # Store results
                    st.session_state.clusters = clusters
                    st.session_state.pca_data = pca_data
                    st.session_state.scaler = scaler
                    st.session_state.pca_model = pca_model
                    st.session_state.cluster_model = kmeans_model
                    
                    st.success(f"✅ K-Means clustering completed with {n_clusters} clusters!")
        
        with col2:
            if st.button("🔄 Run DBSCAN Analysis", key="dbscan_batch"):
                with st.spinner("Running DBSCAN clustering..."):
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(st.session_state.processed_data)
                    
                    # Perform PCA
                    pca_data, pca_model = perform_pca(scaled_data, n_components=n_components)
                    
                    # Perform DBSCAN clustering
                    clusters, dbscan_model = perform_dbscan_clustering(pca_data, eps=eps, min_samples=min_samples)
                    
                    # Store results
                    st.session_state.clusters = clusters
                    st.session_state.pca_data = pca_data
                    st.session_state.scaler = scaler
                    st.session_state.pca_model = pca_model
                    st.session_state.cluster_model = dbscan_model
                    
                    n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
                    n_noise = list(clusters).count(-1)
                    
                    st.success(f"✅ DBSCAN clustering completed! Found {n_clusters_found} clusters and {n_noise} noise points.")
        
        with col3:
            if st.button("📊 Find Optimal Clusters", key="optimal_clusters"):
                with st.spinner("Finding optimal number of clusters..."):
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(st.session_state.processed_data)
                    
                    # Find optimal clusters
                    K_range, inertias, silhouette_scores = find_optimal_clusters(scaled_data, max_clusters=10)
                    
                    # Create elbow plot
                    fig = create_elbow_plot(K_range, inertias, silhouette_scores)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommend optimal number
                    optimal_k = K_range[np.argmax(silhouette_scores)]
                    st.success(f"✅ Recommended number of clusters: {optimal_k} (highest silhouette score)")
        
        # Display results if clustering has been performed
        if st.session_state.clusters is not None:
            st.markdown("### 📊 Batch Processing Results")
            
            # Generate profiles and insights
            profiles = generate_cluster_profiles(st.session_state.processed_data, st.session_state.clusters)
            insights = generate_business_insights(profiles)
            
            # Cluster summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Cluster Distribution")
                fig_dist = create_cluster_distribution_plot(st.session_state.clusters)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Silhouette Analysis")
                fig_silhouette = create_silhouette_plot(
                    st.session_state.scaler.transform(st.session_state.processed_data), 
                    st.session_state.clusters
                )
                st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Detailed results table
            st.markdown("#### 📋 Detailed Cluster Profiles")
            
            profile_data = []
            for cluster_id, profile in profiles.items():
                profile_data.append({
                    'Cluster': f'Cluster {cluster_id}',
                    'Size': profile['size'],
                    'Percentage': f"{profile['percentage']:.1f}%",
                    'Avg Balance': format_currency(profile['avg_balance']),
                    'Avg Purchases': format_currency(profile['avg_purchases']),
                    'Avg Cash Advance': format_currency(profile['avg_cash_advance']),
                    'Purchase Frequency': format_percentage(profile['purchase_frequency']),
                    'Risk Level': insights[cluster_id]['risk_level'] if cluster_id in insights else 'N/A'
                })
            
            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)
            
            # Download options
            st.markdown("### 📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create clustered dataset
                clustered_data = st.session_state.data.copy()
                clustered_data['Cluster'] = st.session_state.clusters
                
                csv_link = create_download_link(clustered_data, "clustered_customers.csv", "📊 Download Clustered Data")
                st.markdown(csv_link, unsafe_allow_html=True)
            
            with col2:
                # Create profiles summary
                profiles_df = pd.DataFrame(profile_data)
                profiles_link = create_download_link(profiles_df, "cluster_profiles.csv", "📈 Download Cluster Profiles")
                st.markdown(profiles_link, unsafe_allow_html=True)
            
            with col3:
                # Create comprehensive report
                report_link = create_report_download(insights, profiles)
                st.markdown(report_link, unsafe_allow_html=True)
    
    # Visualizations Section
    elif selected == "📈 Visualizations":
        st.markdown('<div class="section-header"><h2>📈 Advanced Visualizations</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.markdown('<div class="alert alert-warning">⚠️ Please upload data or load sample data first!</div>', unsafe_allow_html=True)
            return
        
        # Perform clustering if not done
        if st.session_state.clusters is None:
            st.markdown('<div class="alert alert-info">💡 Performing automatic clustering analysis for visualizations...</div>', unsafe_allow_html=True)
            
            with st.spinner("Preparing visualizations..."):
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(st.session_state.processed_data)
                pca_data, pca_model = perform_pca(scaled_data, n_components=3)
                clusters, kmeans_model = perform_kmeans_clustering(scaled_data, n_clusters=4)
                
                st.session_state.clusters = clusters
                st.session_state.pca_data = pca_data
                st.session_state.scaler = scaler
                st.session_state.pca_model = pca_model
                st.session_state.cluster_model = kmeans_model
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Cluster Analysis", "🔍 PCA Analysis", "🌡️ Correlation Analysis", "📊 Feature Analysis", "🚀 Advanced Plots"])
        
        with tab1:
            st.markdown("### 🎯 Cluster Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 3D cluster plot
                fig_3d = create_cluster_3d_plot(st.session_state.pca_data, st.session_state.clusters, "3D Customer Segments")
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with col2:
                # Cluster distribution
                fig_dist = create_cluster_distribution_plot(st.session_state.clusters)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Silhouette analysis
            st.markdown("### 🎯 Silhouette Analysis")
            fig_silhouette = create_silhouette_plot(
                st.session_state.scaler.transform(st.session_state.processed_data), 
                st.session_state.clusters
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Cluster comparison radar
            profiles = generate_cluster_profiles(st.session_state.processed_data, st.session_state.clusters)
            fig_radar = create_cluster_comparison_radar(profiles)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔍 PCA Analysis")
            
            # PCA explained variance
            fig_pca = create_pca_explained_variance_plot(st.session_state.pca_model)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # PCA components interpretation
            st.markdown("### 📊 PCA Components")
            
            components_df = pd.DataFrame(
                st.session_state.pca_model.components_.T,
                columns=[f'PC{i+1}' for i in range(st.session_state.pca_model.n_components_)],
                index=st.session_state.processed_data.columns
            )
            
            st.dataframe(components_df.style.background_gradient(cmap='RdBu', axis=0), use_container_width=True)
            
            # Feature loadings plot
            st.markdown("### 🎯 Feature Loadings")
            
            fig_loadings = go.Figure()
            
            for i in range(min(3, st.session_state.pca_model.n_components_)):
                fig_loadings.add_trace(go.Bar(
                    name=f'PC{i+1}',
                    x=components_df.index,
                    y=components_df[f'PC{i+1}'],
                    opacity=0.8
                ))
            
            fig_loadings.update_layout(
                title='PCA Feature Loadings',
                xaxis_title='Features',
                yaxis_title='Loading Value',
                barmode='group',
                height=500,
                xaxis_tickangle=45
            )
            
            st.plotly_chart(fig_loadings, use_container_width=True)
        
        with tab3:
            st.markdown("### 🌡️ Correlation Analysis")
            
            # Correlation heatmap
            fig_corr = create_correlation_heatmap(st.session_state.processed_data)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature correlation insights
            st.markdown("### 🔍 Key Correlations")
            corr_matrix = st.session_state.processed_data.corr()
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔺 Strongest Positive Correlations")
                positive_corr = corr_df[corr_df['Correlation'] > 0].head(5)
                for _, row in positive_corr.iterrows():
                    st.write(f"**{row['Feature 1']}** ↔ **{row['Feature 2']}**: {row['Correlation']:.3f}")
            
            with col2:
                st.markdown("#### 🔻 Strongest Negative Correlations")
                negative_corr = corr_df[corr_df['Correlation'] < 0].head(5)
                for _, row in negative_corr.iterrows():
                    st.write(f"**{row['Feature 1']}** ↔ **{row['Feature 2']}**: {row['Correlation']:.3f}")
        
        with tab4:
            st.markdown("### 📊 Feature Analysis")
            
            # Feature importance based on variance
            fig_importance = create_feature_importance_plot(st.session_state.processed_data, st.session_state.processed_data.columns)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature distribution analysis
            st.markdown("### 📈 Feature Distributions")
            
            # Select features for distribution analysis
            selected_features = st.multiselect(
                "Select features to analyze:",
                options=list(st.session_state.processed_data.columns),
                default=list(st.session_state.processed_data.columns)[:4]
            )
            
            if selected_features:
                n_cols = min(2, len(selected_features))
                n_rows = (len(selected_features) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=n_cols,
                    subplot_titles=selected_features,
                    specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
                )
                
                for i, feature in enumerate(selected_features):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(
                            x=st.session_state.processed_data[feature],
                            name=feature,
                            showlegend=False,
                            opacity=0.7
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(height=300 * n_rows, title_text="Feature Distribution Analysis")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(st.session_state.processed_data[selected_features].describe(), use_container_width=True)
        
        with tab5:
            st.markdown("### 🚀 Advanced Visualizations")
            
            # Customer journey plot
            fig_journey = create_customer_journey_plot(st.session_state.processed_data, st.session_state.clusters)
            st.plotly_chart(fig_journey, use_container_width=True)
            
            # Advanced scatter plot matrix
            st.markdown("### 🔍 Scatter Plot Matrix")
            
            # Select features for scatter matrix
            scatter_features = st.multiselect(
                "Select features for scatter matrix:",
                options=list(st.session_state.processed_data.columns),
                default=['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']
            )
            
            if len(scatter_features) >= 2:
                # Create scatter matrix
                df_scatter = st.session_state.processed_data[scatter_features].copy()
                df_scatter['Cluster'] = st.session_state.clusters
                
                fig_scatter = px.scatter_matrix(
                    df_scatter,
                    dimensions=scatter_features,
                    color='Cluster',
                    title="Feature Relationships Scatter Matrix"
                )
                fig_scatter.update_layout(height=600)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Parallel coordinates plot
            st.markdown("### 🎯 Parallel Coordinates")
            
            # Normalize data for parallel coordinates
            from sklearn.preprocessing import MinMaxScaler
            scaler_viz = MinMaxScaler()
            normalized_data = pd.DataFrame(
                scaler_viz.fit_transform(st.session_state.processed_data),
                columns=st.session_state.processed_data.columns
            )
            normalized_data['Cluster'] = st.session_state.clusters
            
            # Select features for parallel coordinates
            parallel_features = st.multiselect(
                "Select features for parallel coordinates:",
                options=list(st.session_state.processed_data.columns),
                default=['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'CREDIT_LIMIT']
            )
            
            if parallel_features:
                fig_parallel = go.Figure(data=
                    go.Parcoords(
                        line=dict(color=normalized_data['Cluster'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Cluster")),
                        dimensions=list([
                            dict(range=[0, 1],
                                constraintrange=[0, 1],
                                label=feature,
                                values=normalized_data[feature]) for feature in parallel_features
                        ])
                    )
                )
                
                fig_parallel.update_layout(
                    title='Parallel Coordinates Plot - Customer Segments',
                    height=500
                )
                st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Business Insights Section
    elif selected == "💡 Business Insights":
        st.markdown('<div class="section-header"><h2>💡 Business Insights & Recommendations</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.markdown('<div class="alert alert-warning">⚠️ Please upload data or load sample data first!</div>', unsafe_allow_html=True)
            return
        
        if st.session_state.clusters is None:
            st.markdown('<div class="alert alert-info">💡 Performing clustering analysis to generate insights...</div>', unsafe_allow_html=True)
            
            with st.spinner("Generating business insights..."):
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(st.session_state.processed_data)
                pca_data, pca_model = perform_pca(scaled_data, n_components=3)
                clusters, kmeans_model = perform_kmeans_clustering(scaled_data, n_clusters=4)
                
                st.session_state.clusters = clusters
                st.session_state.pca_data = pca_data
                st.session_state.scaler = scaler
                st.session_state.pca_model = pca_model
                st.session_state.cluster_model = kmeans_model
        
        # Generate profiles and insights
        profiles = generate_cluster_profiles(st.session_state.processed_data, st.session_state.clusters)
        insights = generate_business_insights(profiles)
        
        # Executive Summary
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = sum(profile['size'] for profile in profiles.values())
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            n_segments = len(profiles)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Customer Segments</h3>
                <div class="metric-value">{n_segments}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_value_customers = sum(1 for insight in insights.values() if 'High-Value' in insight['segment_type'])
            high_value_pct = (high_value_customers / len(insights)) * 100 if insights else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>High-Value Segments</h3>
                <div class="metric-value">{high_value_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high_risk_customers = sum(1 for insight in insights.values() if insight['risk_level'] == 'High')
            high_risk_pct = (high_risk_customers / len(insights)) * 100 if insights else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>High-Risk Segments</h3>
                <div class="metric-value">{high_risk_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Segment Analysis
        st.markdown("### 🎯 Detailed Segment Analysis")
        
        for cluster_id, insight in insights.items():
            profile = profiles[cluster_id]
            risk_color = get_risk_color(insight['risk_level'])
            
            with st.expander(f"{insight['segment_type']} - {profile['size']:,} customers ({profile['percentage']:.1f}%)", expanded=True):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>📋 Segment Profile</h4>
                        <p><strong>Description:</strong> {insight['description']}</p>
                        <p><strong>Size:</strong> {profile['size']:,} customers ({profile['percentage']:.1f}% of total)</p>
                        <p><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: 600;">{insight['risk_level']}</span></p>
                        <p><strong>Marketing Strategy:</strong> {insight['marketing_strategy']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 💡 Strategic Recommendations")
                    for i, rec in enumerate(insight['recommendations'], 1):
                        st.markdown(f"**{i}.** {rec}")
                
                with col2:
                    st.markdown("#### 📊 Key Metrics")
                    
                    metrics_data = {
                        'Metric': [
                            'Average Balance',
                            'Average Purchases', 
                            'Average Cash Advance',
                            'Average Credit Limit',
                            'Purchase Frequency',
                            'Cash Advance Frequency',
                            'Average Tenure'
                        ],
                        'Value': [
                            format_currency(profile['avg_balance']),
                            format_currency(profile['avg_purchases']),
                            format_currency(profile['avg_cash_advance']),
                            format_currency(profile['avg_credit_limit']),
                            format_percentage(profile['purchase_frequency']),
                            format_percentage(profile['cash_advance_frequency']),
                            f"{profile['tenure']:.1f} months"
                        ]
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Strategic Recommendations Summary
        st.markdown("### 🎯 Strategic Action Plan")
        
        # Prioritize segments by value and risk
        segment_priority = []
        for cluster_id, insight in insights.items():
            profile = profiles[cluster_id]
            
            # Calculate priority score (higher is more important)
            value_score = (profile['avg_balance'] + profile['avg_purchases']) / 10000
            size_score = profile['percentage'] / 100
            risk_multiplier = {'High': 1.5, 'Medium': 1.2, 'Low': 1.0}[insight['risk_level']]
            
            priority_score = (value_score + size_score) * risk_multiplier
            
            segment_priority.append({
                'segment': insight['segment_type'],
                'cluster_id': cluster_id,
                'priority_score': priority_score,
                'size': profile['size'],
                'percentage': profile['percentage'],
                'risk_level': insight['risk_level']
            })
        
        # Sort by priority
        segment_priority.sort(key=lambda x: x['priority_score'], reverse=True)
        
        st.markdown("#### 🏆 Priority Segments (Ranked by Strategic Importance)")
        
        for i, segment in enumerate(segment_priority, 1):
            risk_color = get_risk_color(segment['risk_level'])
            
            st.markdown(f"""
            **{i}. {segment['segment']}**
            - Size: {segment['size']:,} customers ({segment['percentage']:.1f}%)
            - Risk Level: <span style="color: {risk_color}; font-weight: 600;">{segment['risk_level']}</span>
            - Priority Score: {segment['priority_score']:.2f}
            """, unsafe_allow_html=True)
        
        # ROI Projections
        st.markdown("### 💰 ROI Projections & Business Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Revenue Opportunity Analysis")
            
            total_balance = sum(profile['avg_balance'] * profile['size'] for profile in profiles.values())
            total_purchases = sum(profile['avg_purchases'] * profile['size'] for profile in profiles.values())
            
            st.write(f"**Total Customer Balance:** {format_currency(total_balance)}")
            st.write(f"**Total Annual Purchases:** {format_currency(total_purchases)}")
            st.write(f"**Average Revenue per Customer:** {format_currency((total_balance + total_purchases) / sum(profile['size'] for profile in profiles.values()))}")
        
        with col2:
            st.markdown("#### ⚠️ Risk Management Priorities")
            
            high_risk_segments = [s for s in segment_priority if s['risk_level'] == 'High']
            medium_risk_segments = [s for s in segment_priority if s['risk_level'] == 'Medium']
            
            if high_risk_segments:
                st.write(f"**High-Risk Customers:** {sum(s['size'] for s in high_risk_segments):,}")
                st.write("**Immediate Action Required:** Financial counseling, credit monitoring")
            
            if medium_risk_segments:
                st.write(f"**Medium-Risk Customers:** {sum(s['size'] for s in medium_risk_segments):,}")
                st.write("**Proactive Monitoring:** Engagement campaigns, usage optimization")
        
        # Download comprehensive report
        st.markdown("### 📥 Export Business Intelligence Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create executive summary report
            executive_summary = {
                'timestamp': datetime.now().isoformat(),
                'total_customers': sum(profile['size'] for profile in profiles.values()),
                'segments': len(profiles),
                'high_value_segments': sum(1 for insight in insights.values() if 'High-Value' in insight['segment_type']),
                'high_risk_segments': sum(1 for insight in insights.values() if insight['risk_level'] == 'High'),
                'segment_priorities': segment_priority,
                'key_insights': {cluster_id: {
                    'segment_type': insight['segment_type'],
                    'size': profiles[cluster_id]['size'],
                    'percentage': profiles[cluster_id]['percentage'],
                    'risk_level': insight['risk_level'],
                    'top_recommendations': insight['recommendations'][:3]
                } for cluster_id, insight in insights.items()}
            }
            
            # Convert all numpy types to native Python types before JSON serialization
            executive_summary_native = convert_to_native_types(executive_summary)


            summary_json = json.dumps(executive_summary_native, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(summary_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="executive_summary.json" style="text-decoration: none; background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600;">📊 Executive Summary</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Create detailed insights report
            report_link = create_report_download(insights, profiles)
            st.markdown(report_link, unsafe_allow_html=True)
        
        with col3:
            # Create action plan document
            action_plan = {
                'timestamp': datetime.now().isoformat(),
                'priority_segments': segment_priority,
                'immediate_actions': [],
                'medium_term_strategies': [],
                'long_term_goals': []
            }
           
            for segment in segment_priority[:3]:  # Top 3 priority segments
                cluster_id = segment['cluster_id']
                if cluster_id in insights:
                    action_plan['immediate_actions'].extend(insights[cluster_id]['recommendations'][:2])
            
            action_plan_native= convert_to_native_types(action_plan)
            action_json = json.dumps(action_plan_native, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(action_json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="action_plan.json" style="text-decoration: none; background-color: #2ca02c; color: white; padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600;">🎯 Action Plan</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()