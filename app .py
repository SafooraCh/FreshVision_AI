"""
FreshVision — Streamlit inference app (polished edition)
----------------------------------------------------------
Loads the trained multi-task CNN (freshvision_cnn_final.keras) plus its
companion artifact bundle (freshvision_artifacts.pkl) and provides:
  - A styled hero header + custom theme
  - Single & batch image upload with drag-and-drop
  - Animated confidence gauges (Plotly) for food type + freshness
  - Top-5 food type prediction bar chart
  - A "Model Insights" tab: training curves, confusion matrices,
    per-task metrics pulled straight from the saved artifact bundle
  - Downloadable CSV of batch predictions

Run with:
    streamlit run app.py

Expected files in the same directory (produced by the training pipeline):
    freshvision_cnn_final.keras
    freshvision_artifacts.pkl
"""

import os
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "freshvision_cnn_final.keras"
ARTIFACTS_PATH = "freshvision_artifacts.pkl"

st.set_page_config(
    page_title="FreshVision · AI Food Freshness Classifier",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1DB954"
DANGER = "#E5484D"
DARK_BG = "#0E1117"
CARD_BG = "#171B22"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top left, #182028 0%, {DARK_BG} 55%);
    }}
    .hero {{
        padding: 2.2rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(29,185,84,0.18), rgba(29,185,84,0.02));
        border: 1px solid rgba(29,185,84,0.25);
        margin-bottom: 1.5rem;
    }}
    .hero h1 {{
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1DB954, #7CE0A0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero p {{
        color: #B7C0C9;
        font-size: 1.05rem;
        margin-top: 0;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }}
    .metric-card h2 {{
        margin: 0;
        font-size: 1.9rem;
    }}
    .metric-card span {{
        color: #8A93A0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .badge-fresh {{
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(29,185,84,0.15);
        color: {PRIMARY};
        border: 1px solid rgba(29,185,84,0.4);
        font-weight: 600;
    }}
    .badge-rotten {{
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(229,72,77,0.15);
        color: {DANGER};
        border: 1px solid rgba(229,72,77,0.4);
        font-weight: 600;
    }}
    div[data-testid="stFileUploader"] section {{
        border: 2px dashed rgba(29,185,84,0.35);
        border-radius: 14px;
        background: rgba(29,185,84,0.03);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_artifacts(artifacts_path):
    if not os.path.exists(artifacts_path):
        return None
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def make_gauge(value, title, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "white"}},
            title={"text": title, "font": {"size": 15, "color": "#B7C0C9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3A4351", "tickfont": {"color": "#8A93A0"}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,255,255,0.04)"},
                    {"range": [50, 80], "color": "rgba(255,255,255,0.07)"},
                    {"range": [80, 100], "color": "rgba(255,255,255,0.1)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def predict(model, bundle, pil_image):
    img_size = bundle["img_size"]
    idx_to_food_type = bundle["idx_to_food_type"]

    img = pil_image.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    food_probs, fresh_prob = model.predict(arr, verbose=0)

    food_idx = int(np.argmax(food_probs[0]))
    food_label = idx_to_food_type[food_idx]
    food_confidence = float(food_probs[0][food_idx])

    fresh_raw = float(fresh_prob[0][0])
    freshness_label = "Fresh" if fresh_raw > 0.5 else "Rotten"
    freshness_confidence = fresh_raw if freshness_label == "Fresh" else 1 - fresh_raw

    top_k = min(5, len(idx_to_food_type))
    top_indices = np.argsort(food_probs[0])[::-1][:top_k]
    top_food = pd.DataFrame(
        {
            "food_type": [idx_to_food_type[i] for i in top_indices],
            "confidence": [float(food_probs[0][i]) for i in top_indices],
        }
    )

    return {
        "food_type": food_label,
        "food_confidence": food_confidence,
        "freshness": freshness_label,
        "freshness_confidence": freshness_confidence,
        "top_food_predictions": top_food,
    }


# ---------------------------------------------------------------------------
# Load model/artifacts
# ---------------------------------------------------------------------------
model = load_model(MODEL_PATH)
bundle = load_artifacts(ARTIFACTS_PATH)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🥦 FreshVision</h1>
        <p>AI-powered food type recognition & freshness detection, in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or bundle is None:
    st.error(
        "Could not find the trained model and/or artifact bundle in this directory.\n\n"
        f"Expected:\n- `{MODEL_PATH}`\n- `{ARTIFACTS_PATH}`\n\n"
        "Place both files (produced by the training pipeline) next to `app.py` before running."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Top metric strip
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
test_results = bundle.get("test_results", {})
food_acc = test_results.get("food_type_output_accuracy")
fresh_acc = test_results.get("freshness_output_accuracy")

with m1:
    st.markdown(
        f'<div class="metric-card"><span>Food Classes</span><h2>{bundle["num_food_classes"]}</h2></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><span>Food Accuracy</span><h2>{food_acc:.1%}</h2></div>'
        if food_acc is not None
        else '<div class="metric-card"><span>Food Accuracy</span><h2>—</h2></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><span>Freshness Accuracy</span><h2>{fresh_acc:.1%}</h2></div>'
        if fresh_acc is not None
        else '<div class="metric-card"><span>Freshness Accuracy</span><h2>—</h2></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><span>Image Size</span><h2>{bundle["img_size"][0]}×{bundle["img_size"][1]}</h2></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔍 Predict", "🗂️ Batch Mode", "📊 Model Insights", "ℹ️ About"]
)

# ---------------- Predict tab ----------------
with tab_predict:
    left, right = st.columns([1, 1.2])

    with left:
        uploaded_file = st.file_uploader(
            "Drop an image here or click to browse", type=["jpg", "jpeg", "png"], key="single"
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, use_container_width=True, caption="Uploaded image")

    with right:
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                result = predict(model, bundle, pil_image)

            badge_class = "badge-fresh" if result["freshness"] == "Fresh" else "badge-rotten"
            st.markdown(
                f"### {result['food_type'].title()} &nbsp; "
                f'<span class="{badge_class}">{result["freshness"]}</span>',
                unsafe_allow_html=True,
            )

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    make_gauge(result["food_confidence"], "Food Type Confidence", PRIMARY),
                    use_container_width=True,
                )
            with g2:
                gauge_color = PRIMARY if result["freshness"] == "Fresh" else DANGER
                st.plotly_chart(
                    make_gauge(result["freshness_confidence"], "Freshness Confidence", gauge_color),
                    use_container_width=True,
                )

            st.markdown("**Top-5 food type predictions**")
            bar_fig = px.bar(
                result["top_food_predictions"].sort_values("confidence"),
                x="confidence",
                y="food_type",
                orientation="h",
                text_auto=".1%",
                color="confidence",
                color_continuous_scale=["#3A4351", PRIMARY],
            )
            bar_fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                xaxis={"tickformat": ".0%", "showgrid": False},
                yaxis_title=None,
                xaxis_title=None,
                coloraxis_showscale=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Upload an image on the left to see live predictions here.")

# ---------------- Batch tab ----------------
with tab_batch:
    st.write("Upload multiple images to classify them all at once, then export the results.")
    batch_files = st.file_uploader(
        "Drop multiple images here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
    )

    if batch_files:
        rows = []
        progress = st.progress(0, text="Running batch inference...")
        thumb_cols = st.columns(6)

        for i, f in enumerate(batch_files):
            pil_img = Image.open(f)
            res = predict(model, bundle, pil_img)
            rows.append(
                {
                    "filename": f.name,
                    "food_type": res["food_type"],
                    "food_confidence": res["food_confidence"],
                    "freshness": res["freshness"],
                    "freshness_confidence": res["freshness_confidence"],
                }
            )
            with thumb_cols[i % 6]:
                st.image(pil_img, use_container_width=True, caption=res["freshness"])
            progress.progress((i + 1) / len(batch_files), text=f"Processed {i + 1}/{len(batch_files)}")

        progress.empty()
        results_df = pd.DataFrame(rows)

        st.markdown("#### Batch results")
        st.dataframe(
            results_df.style.format(
                {"food_confidence": "{:.1%}", "freshness_confidence": "{:.1%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        fresh_counts = results_df["freshness"].value_counts()
        pie_fig = px.pie(
            values=fresh_counts.values,
            names=fresh_counts.index,
            color=fresh_counts.index,
            color_discrete_map={"Fresh": PRIMARY, "Rotten": DANGER},
            hole=0.55,
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"freshvision_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No images uploaded yet.")

# ---------------- Insights tab ----------------
with tab_insights:
    hist = bundle.get("history")
    if hist:
        st.markdown("#### Training curves")
        c1, c2 = st.columns(2)
        with c1:
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=hist["loss"], name="Train loss", line=dict(color=PRIMARY)))
            loss_fig.add_trace(go.Scatter(y=hist["val_loss"], name="Val loss", line=dict(color=DANGER)))
            loss_fig.update_layout(
                title="Total loss", height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(loss_fig, use_container_width=True)
        with c2:
            acc_fig = go.Figure()
            if "val_food_type_output_accuracy" in hist:
                acc_fig.add_trace(go.Scatter(y=hist["food_type_output_accuracy"], name="Food train acc", line=dict(color=PRIMARY)))
                acc_fig.add_trace(go.Scatter(y=hist["val_food_type_output_accuracy"], name="Food val acc", line=dict(color="#7CE0A0", dash="dot")))
            if "val_freshness_output_accuracy" in hist:
                acc_fig.add_trace(go.Scatter(y=hist["freshness_output_accuracy"], name="Freshness train acc", line=dict(color=DANGER)))
                acc_fig.add_trace(go.Scatter(y=hist["val_freshness_output_accuracy"], name="Freshness val acc", line=dict(color="#F2A6A9", dash="dot")))
            acc_fig.update_layout(
                title="Accuracy", height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(acc_fig, use_container_width=True)
    else:
        st.info("No training history found in the artifact bundle.")

    st.markdown("#### Confusion matrices")
    cm1, cm2 = st.columns(2)
    if "confusion_matrix_food" in bundle:
        with cm1:
            cm_fig = px.imshow(
                bundle["confusion_matrix_food"],
                x=bundle["food_types"],
                y=bundle["food_types"],
                color_continuous_scale="Greens",
                labels=dict(color="Count"),
            )
            cm_fig.update_layout(
                title="Food type", height=450, paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(cm_fig, use_container_width=True)
    if "confusion_matrix_freshness" in bundle:
        with cm2:
            cm_fig2 = px.imshow(
                bundle["confusion_matrix_freshness"],
                x=["Rotten", "Fresh"],
                y=["Rotten", "Fresh"],
                color_continuous_scale="Reds",
                labels=dict(color="Count"),
            )
            cm_fig2.update_layout(
                title="Freshness", height=450, paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(cm_fig2, use_container_width=True)

    if "summary_table" in bundle:
        st.markdown("#### Per-task summary metrics")
        st.dataframe(bundle["summary_table"], use_container_width=True, hide_index=True)

# ---------------- About tab ----------------
with tab_about:
    st.markdown(
        """
        **FreshVision** is a multi-task CNN that jointly predicts:
        1. **Food type** — which produce item is in the image
        2. **Freshness** — whether that item is fresh or rotten

        The model shares a convolutional backbone between both tasks and
        branches into two heads: a softmax classifier for food type and a
        sigmoid classifier for freshness. It was trained with class-balanced
        weighting to handle uneven category sizes, and evaluated with
        per-task accuracy, precision, recall, and F1.

        This app loads the trained `.keras` model and a companion `.pkl`
        artifact bundle (label mappings, training history, and evaluation
        metrics) so predictions and the *Model Insights* tab stay in sync
        with whatever was produced by the training pipeline.
        """
    )
