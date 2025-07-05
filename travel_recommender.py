import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = ['DestinationName', 'State', 'Category', 'BestTimeToVisit',
            'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

# Load destination data
destinations_df = pd.read_csv("Destinations.csv")
destinations_df.rename(columns={'Type': 'Category'}, inplace=True)

# Add destination images
image_map = {
    'Jaipur City': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg/1024px-East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg',
    'Taj Mahal': 'https://upload.wikimedia.org/wikipedia/commons/d/da/Taj-Mahal.jpg',
    'Kerala Backwaters': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Backwaters_in_Kerala.jpg',
    'Goa Beaches': 'https://upload.wikimedia.org/wikipedia/commons/4/48/Palolem_beach_sunset_Goa.jpg',
    'Leh Ladakh': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Leh_Ladakh_-_Pangong_Lake.jpg',
}
destinations_df['ImageURL'] = destinations_df['DestinationName'].map(image_map)

# Streamlit UI
st.set_page_config(page_title="Travel Recommender", layout="wide")
st.title("✈️ Personalized Travel Recommendation System")
st.markdown("Get the best travel recommendations based on your preferences!")

# User input form
with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        preferences = st.selectbox("Preferences", ['Beaches, Historical', 'Nature, Adventure', 'City, Historical'])
        num_adults = st.slider("Number of Adults", 1, 10, 2)
    with col2:
        num_children = st.slider("Number of Children", 0, 5, 0)
        best_time = st.selectbox("Preferred Travel Time", destinations_df['BestTimeToVisit'].unique())
        state = st.selectbox("Preferred State", destinations_df['State'].unique())
    
    submit = st.form_submit_button("Get Recommendations")

if submit:
    predictions = []

    for _, row in destinations_df.iterrows():
        input_dict = {
            'DestinationName': row['DestinationName'],
            'State': state,
            'Category': row['Category'],
            'BestTimeToVisit': best_time,
            'Preferences': preferences,
            'Gender': gender,
            'NumberOfAdults': num_adults,
            'NumberOfChildren': num_children
        }

        # Encode input
        input_encoded = input_dict.copy()
        for col in label_encoders:
            input_encoded[col] = label_encoders[col].transform([input_encoded[col]])[0]

        input_df = pd.DataFrame([input_encoded])
        predicted_score = model.predict(input_df)[0]

        predictions.append({
            'DestinationName': row['DestinationName'],
            'Location': row['State'],
            'Category': row['Category'],
            'BestTimeToVisit': row['BestTimeToVisit'],
            'Popularity Score': round(predicted_score, 2),
            'ImageURL': row.get('ImageURL', '')
        })

    # Convert to DataFrame and sort
    result_df = pd.DataFrame(predictions).sort_values(by='Popularity Score', ascending=False).head(3)

    st.success("✅ Top 3 Recommendations based on your preferences:")
    for _, row in result_df.iterrows():
        st.subheader(row['DestinationName'])
        if pd.notna(row['ImageURL']):
            st.image(row['ImageURL'], use_column_width=True)
        st.markdown(f"📍 **Location**: {row['Location']}")
        st.markdown(f"📅 **Best Time to Visit**: {row['BestTimeToVisit']}")
        st.markdown(f"⭐ **Predicted Popularity Score**: {row['Popularity Score']}")
        st.markdown("---")

