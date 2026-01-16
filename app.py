import streamlit as st
import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# -------------------------------
# Load tokenizer and model
# -------------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("sentiment_model.keras")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Text Sentiment Analysis",
    page_icon="⭐",
    layout="centered"
)

st.title("🌟 Text Sentiment Analysis 🌟")
st.write("Enter a sentence below to analyze its sentiment.")

user_input = st.text_area("Enter your text:", height=200)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():

        # -------------------------------
        # Preprocessing
        # -------------------------------
        text = user_input.lower().strip()
        text = re.sub(r"[^a-z\s]", "", text)

        rule_applied = False

        # -------------------------------
        # Sentiment vocab
        # -------------------------------
        positive_words = [
            "amazing", "awesome", "fantastic", "excellent", "brilliant",
            "wonderful", "outstanding", "beautiful", "breathtaking",
            "stunning", "loved", "worth", "masterpiece"
        ]

        negative_words = [
            "worst", "waste", "terrible", "awful", "horrible",
            "bad", "boring", "downgraded", "dont watch",
            "do not watch", "waste of time", "waste of money"
        ]

        # Count sentiment words
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        # -------------------------------
        # Rule 1: Strong NEGATIVE dominance
        # -------------------------------
        if neg_count >= 2 and neg_count > pos_count:
            sentiment = "Negative 😞"
            sentiment_color = "color:#e74c3c; font-size:26px;"
            confidence = 90.0
            rule_applied = True

        # -------------------------------
        # Rule 2: Strong POSITIVE dominance
        # -------------------------------
        if not rule_applied and pos_count >= 2 and pos_count > neg_count:
            sentiment = "Positive 😊"
            sentiment_color = "color:#2ecc71; font-size:26px;"
            confidence = 85.0
            rule_applied = True

        # -------------------------------
        # Rule 3: Negation handling
        # -------------------------------
        if not rule_applied:
            negation_rules = {
                "not bad": "Neutral 😐",
                "not good": "Negative 😞",
                "dont like": "Negative 😞",
                "do not like": "Negative 😞",
                "dont hate": "Neutral 😐",
                "do not hate": "Neutral 😐"
            }

            for phrase, forced_sentiment in negation_rules.items():
                if phrase in text:
                    sentiment = forced_sentiment
                    sentiment_color = (
                        "color:#2ecc71;" if "Positive" in forced_sentiment else
                        "color:#f1c40f;" if "Neutral" in forced_sentiment else
                        "color:#e74c3c;"
                    )
                    confidence = 65.0
                    rule_applied = True
                    break

        # -------------------------------
        # Rule 4: Neutral keywords
        # -------------------------------
        if not rule_applied:
            neutral_keywords = ["okay", "average", "fine", "decent", "normal"]
            if any(word in text for word in neutral_keywords):
                sentiment = "Neutral 😐"
                sentiment_color = "color:#f1c40f; font-size:26px;"
                confidence = 50.0
                rule_applied = True

        # -------------------------------
        # Rule 5: Model fallback
        # -------------------------------
        if not rule_applied:
            tokenized_input = tokenizer.texts_to_sequences([text])
            padded_input = pad_sequences(tokenized_input, maxlen=200)

            prediction = model.predict(padded_input, verbose=0)
            score = prediction[0][0]

            if score >= 0.6:
                sentiment = "Positive 😊"
                sentiment_color = "color:#2ecc71; font-size:26px;"
                confidence = score * 100
            elif score <= 0.4:
                sentiment = "Negative 😞"
                sentiment_color = "color:#e74c3c; font-size:26px;"
                confidence = (1 - score) * 100
            else:
                sentiment = "Neutral 😐"
                sentiment_color = "color:#f1c40f; font-size:26px;"
                confidence = 50.0

        # -------------------------------
        # Display
        # -------------------------------
        st.markdown(
            f"<p style='{sentiment_color}'><b>Predicted Sentiment:</b> {sentiment}</p>",
            unsafe_allow_html=True
        )
        st.write(f"**Confidence:** {confidence:.2f}%")

    else:
        st.warning("❗ Please enter some text to analyze.")
