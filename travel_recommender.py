import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = ['DestinationName', 'State', 'Category', 'BestTimeToVisit', 
            'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

# Load destinations data
destinations_df = pd.read_csv("Destinations.csv")
destinations_df.rename(columns={'Type': 'Category'}, inplace=True)

# Add image URLs (manually mapped or loaded from a CSV)
image_map = {
    'Jaipur City': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg/1024px-East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg',
    'Taj Mahal': 'https://upload.wikimedia.org/wikipedia/commons/d/da/Taj-Mahal.jpg',
    'Kerala Backwaters': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Backwaters_in_Kerala.jpg',
    'Goa Beaches': 'https://upload.wikimedia.org/wikipedia/commons/4/48/Palolem_beach_sunset_Goa.jpg',
    'Leh Ladakh': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Leh_Ladakh_-_Pangong_Lake.jpg',
}
destinations_df['ImageURL'] = destinations_df['DestinationName'].map(image_map)

# Prediction function
def predict_popularity(user_input):
    encoded = {}
    for f in features:
        if f in label_encoders:
            encoded[f] = label_encoders[f].transform([user_input[f]])[0]
        else:
            encoded[f] = user_input[f]
    df = pd.DataFrame([encoded])
    score = model.predict(df)[0]
    return round(score, 3)

# Streamlit UI
st.set_page_config(page_title="Travel Buddy", layout="wide")
st.title("🧭 Travel Buddy – Personalized Destination Recommender")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        dest = st.selectbox("Destination", destinations_df['DestinationName'].unique())
        cat = st.selectbox("Category", ['City', 'Historical', 'Beach', 'Nature', 'Adventure'])
        state = st.selectbox("State", ['Rajasthan', 'Uttar Pradesh', 'Kerala', 'Goa', 'Ladakh'])
        time = st.selectbox("Best Time to Visit", ['Oct-Mar', 'Nov-Feb', 'Dec-May', 'Apr-Jun', 'May-Sep'])

    with col2:
        prefs = st.multiselect("Preferences (Select up to 2)", 
                               ['Beaches', 'Historical', 'Nature', 'Adventure', 'City'], max_selections=2)
        gender = st.radio("Gender", ['Male', 'Female', 'Other'], horizontal=True)
        adults = st.slider("Number of Adults", 1, 10, 2)
        children = st.slider("Number of Children", 0, 10, 1)

    submit = st.form_submit_button("Recommend Destinations")

# Handle submission
if submit:
    preferences = ", ".join(prefs)
    user_input = {
        'DestinationName': dest,
        'Category': cat,
        'State': state,
        'BestTimeToVisit': time,
        'Preferences': preferences,
        'Gender': gender,
        'NumberOfAdults': adults,
        'NumberOfChildren': children,
    }

    # Predict popularity for all destinations
    scores = []
    for name in destinations_df['DestinationName'].unique():
        user_input['DestinationName'] = name
        try:
            score = predict_popularity(user_input)
        except Exception:
            score = 0
        scores.append(score)

    destinations_df['PredictedPopularity'] = scores
    top_destinations = destinations_df.sort_values(by='PredictedPopularity', ascending=False).head(3)

    st.success("✅ Top 3 Recommended Destinations for You:")

    for _, row in top_destinations.iterrows():
        st.markdown(f"### {row['DestinationName']} ({row['PredictedPopularity']:.2f})")
        cols = st.columns([1.5, 3])
        with cols[0]:
            st.image(row['ImageURL'], width=200, caption=row['DestinationName'])
        with cols[1]:
            st.markdown(f"**Location:** {row['State']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"**Best Time to Visit:** {row['BestTimeToVisit']}")
            st.markdown(f"**Popularity Score:** {row['PredictedPopularity']:.3f}")
        st.markdown("---")

st.sidebar.markdown("🚀 Powered by Machine Learning")
