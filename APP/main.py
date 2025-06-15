import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
import pathlib

# Load and clean data
@st.cache_data
def get_clean_data():
    data_path = (pathlib.Path(__file__).parent / "../DATA/data.csv").resolve()
    data = pd.read_csv(data_path)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Sidebar with sliders
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    features = [
        ("Radius Mean", "radius_mean"),
        ("Texture Mean", "texture_mean"),
        ("Perimeter Mean", "perimeter_mean"),
        ("Area Mean", "area_mean"),
        ("Smoothness Mean", "smoothness_mean"),
        ("Compactness Mean", "compactness_mean"),
        ("Concavity Mean", "concavity_mean"),
        ("Concave Points Mean", "concave points_mean"),
        ("Symmetry Mean", "symmetry_mean"),
        ("Fractal Dimension Mean", "fractal_dimension_mean"),
        ("Radius SE", "radius_se"),
        ("Texture SE", "texture_se"),
        ("Perimeter SE", "perimeter_se"),
        ("Area SE", "area_se"),
        ("Smoothness SE", "smoothness_se"),
        ("Compactness SE", "compactness_se"),
        ("Concavity SE", "concavity_se"),
        ("Concave Points SE", "concave points_se"),
        ("Symmetry SE", "symmetry_se"),
        ("Fractal Dimension SE", "fractal_dimension_se"),
        ("Radius Worst", "radius_worst"),
        ("Texture Worst", "texture_worst"),
        ("Perimeter Worst", "perimeter_worst"),
        ("Area Worst", "area_worst"),
        ("Smoothness Worst", "smoothness_worst"),
        ("Compactness Worst", "compactness_worst"),
        ("Concavity Worst", "concavity_worst"),
        ("Concave Points Worst", "concave points_worst"),
        ("Symmetry Worst", "symmetry_worst"),
        ("Fractal Dimension Worst", "fractal_dimension_worst"),
    ]

    user_input = {}
    for label, key in features:
        min_val = float(data[key].min())
        max_val = float(data[key].max())
        mean_val = float(data[key].mean())
        user_input[key] = st.sidebar.slider(label, min_value=min_val, max_value=max_val, value=mean_val)

    return pd.DataFrame([user_input])

# Radar Chart with normalization
def get_radar_chart(input_df):
    data = get_clean_data()
    row = input_df.iloc[0]
    
    categories = [
        'Radius', 'Texture', 'Perimeter', 'Area',
        'Smoothness', 'Compactness',
        'Concavity', 'Concave Points',
        'Symmetry', 'Fractal Dimension'
    ]
    
    feature_map = {
        'Radius': ('radius_mean', 'radius_se', 'radius_worst'),
        'Texture': ('texture_mean', 'texture_se', 'texture_worst'),
        'Perimeter': ('perimeter_mean', 'perimeter_se', 'perimeter_worst'),
        'Area': ('area_mean', 'area_se', 'area_worst'),
        'Smoothness': ('smoothness_mean', 'smoothness_se', 'smoothness_worst'),
        'Compactness': ('compactness_mean', 'compactness_se', 'compactness_worst'),
        'Concavity': ('concavity_mean', 'concavity_se', 'concavity_worst'),
        'Concave Points': ('concave points_mean', 'concave points_se', 'concave points_worst'),
        'Symmetry': ('symmetry_mean', 'symmetry_se', 'symmetry_worst'),
        'Fractal Dimension': ('fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst')
    }

    def normalize(col, val):
        col_min = data[col].min()
        col_max = data[col].max()
        return (val - col_min) / (col_max - col_min) if col_max != col_min else 0.5

    mean_values = []
    se_values = []
    worst_values = []

    for cat, cols in feature_map.items():
        mean_values.append(normalize(cols[0], row[cols[0]]))
        se_values.append(normalize(cols[1], row[cols[1]]))
        worst_values.append(normalize(cols[2], row[cols[2]]))

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=mean_values,
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=se_values,
        theta=categories,
        fill='toself',
        name='Standard Error Values'
    ))

    fig.add_trace(go.Scatterpolar(
        r=worst_values,
        theta=categories,
        fill='toself',
        name='Worst Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
    )

    return fig

# Model Prediction
def add_predictions(input_df):
    model_path = (pathlib.Path(__file__).parent / "../MODEL/model.pkl").resolve()
    scaler_path = (pathlib.Path(__file__).parent / "../MODEL/scaler.pkl").resolve()

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

    input_array_selected = scaler.transform(input_df)
    prediction = model.predict(input_array_selected)

    st.write("Prediction:", "Benign" if prediction[0] == 0 else "Malignant")
    st.write(f"Probability of being benign: {model.predict_proba(input_array_selected)[0][0]:.2f}")
    st.write(f"Probability of being malignant: {model.predict_proba(input_array_selected)[0][1]:.2f}")

# Main App
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩‍⚕",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_df = add_sidebar()

    with st.container():
        st.title("🔬 Breast Cancer Predictor")
        st.subheader("Your Input Features")
        st.write(input_df)

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown("### Radar Chart")
            st.plotly_chart(get_radar_chart(input_df), use_container_width=True)

        with col2:
            st.markdown("### The Cell Cluster Prediction")
            st.write("The Cell Cluster is:")
            add_predictions(input_df)
            
    with st.expander("ℹ️ About this App"):
        st.write("""
        This application allows you to predict the likelihood of:
        1. Breast Cancer: Using various cell nuclei measurements, you can predict whether a tumor is benign or malignant.
        2. Breast Cancer Risk: Assess your risk based on medical history and lifestyle factors.
        3. Breast Cancer Screening: Get recommendations for screening based on your age and risk factors.
        4. Breast Cancer Treatment: Understand treatment options based on your diagnosis.
        5. Breast Cancer Support: Find resources and support groups in your area.

        The models used in this app are trained on the Breast Cancer Wisconsin (Diagnostic) dataset, which includes various features of cell nuclei from breast cancer biopsies. The app provides a user-friendly interface to input measurements and visualize predictions through a radar chart.
        
        This app was developed by Mohammed Asbar as part of his ML portfolio.
        """)


if __name__ == '__main__':
    main()
