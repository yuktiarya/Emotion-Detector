"""Streamlit frontend for the emotion-prediction FastAPI service."""

import os

import requests
import streamlit as st


API_URL = os.getenv("EMOTION_API_URL", "https://emotion-detector-1-17h4.onrender.com").rstrip("/")
EMOTION_EMOJIS = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "💛",
    "sadness": "😢",
    "surprise": "😮",
}

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }
    .main .block-container {
        max-width: 820px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent);
        border-radius: 14px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Emotion Detector")
st.write("Wondering what emotion a sentence expresses? Enter it below.")
st.caption("A simple text-classification demo built with machine learning.")

with st.container(border=True):
    st.subheader("Try a sentence")
    text = st.text_area(
        "Your text",
        placeholder="For example: I had a wonderful day with my friends!",
        max_chars=500,
        height=130,
        label_visibility="collapsed",
    )
    st.caption("Enter one sentence or a short message (up to 500 characters).")
    analyze = st.button("Analyze emotion", type="primary", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Type a sentence first, then select **Analyze emotion**.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=10,
            )
        except requests.RequestException as error:
            st.error(
                "Could not reach the prediction service. Start FastAPI in another "
                "terminal, then try again."
            )
            st.caption(f"Connection details: {error}")
        else:
            if not response.ok:
                st.error(f"Prediction service error: {response.text}")
            else:
                result = response.json()
                emotion = result["emotion"]

                st.subheader("Result")
                result_column, confidence_column = st.columns(2)
                with result_column:
                    emoji = EMOTION_EMOJIS.get(emotion.lower(), "💭")
                    st.metric("Detected emotion", f"{emoji} {emotion.title()}")
                with confidence_column:
                    st.metric("Model confidence", f"{result['confidence']:.0%}")

                with st.expander("See other possible emotions"):
                    for alternative in result["alternatives"]:
                        label = alternative["emotion"]
                        st.write(
                            f"{EMOTION_EMOJIS.get(label.lower(), '💭')} "
                            f"{label.title()}"
                        )
                        st.progress(alternative["confidence"])

st.divider()
st.caption(
    "This is a learning project. Its prediction is not a diagnosis and may not "
    "reflect how a person actually feels."
)
