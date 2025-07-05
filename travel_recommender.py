import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = ['DestinationName', 'State', 'Category', 'BestTimeToVisit',
            'Preferences', 'NumberOfAdults', 'NumberOfChildren']

# Load destination data
destinations_df = pd.read_csv("Destinations.csv")
destinations_df.rename(columns={'Type': 'Category'}, inplace=True)

# Add image URLs
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
st.markdown("Get top destination suggestions based on your travel preferences!")

# User input form
with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        preferences = st.selectbox("Your Travel Preferences", [
            'Beaches, Historical', 'Nature, Adventure', 'City, Historical'
        ])
        best_time = st.selectbox("Preferred Travel Time", sorted(destinations_df['BestTimeToVisit'].unique()))
    with col2:
        num_adults = st.slider("Number of Adults", 1, 10, 2)
        num_children = st.slider("Number of Children", 0, 5, 0)

    submit = st.form_submit_button("Get Recommendations")

# Generate recommendations
if submit:
    predictions = []

    for _, row in destinations_df.iterrows():
        input_dict = {
            'DestinationName': row['DestinationName'],
            'State': row['State'],
            'Category': row['Category'],
            'BestTimeToVisit': best_time,
            'Preferences': preferences,
            'NumberOfAdults': num_adults,
            'NumberOfChildren': num_children
        }

        # Encode categorical features
        encoded_input = input_dict.copy()
        for col in label_encoders:
            if col in encoded_input:
                encoded_input[col] = label_encoders[col].transform([encoded_input[col]])[0]

        input_df = pd.DataFrame([encoded_input])
        predicted_score = model.predict(input_df)[0]

        predictions.append({
            'DestinationName': row['DestinationName'],
            'Location': row['State'],
            'Category': row['Category'],
            'BestTimeToVisit': row['BestTimeToVisit'],
            'Popularity Score': round(predicted_score, 2),
            'ImageURL': row.get('ImageURL', '')
        })

    # Convert to DataFrame and drop duplicate names
    results_df = pd.DataFrame(predictions).drop_duplicates(subset=['DestinationName'])

    # Get top 3 based on predicted popularity score
    top_df = results_df.sort_values(by='Popularity Score', ascending=False).head(3).reset_index(drop=True)

    if not top_df.empty:
                st.success("✅ Top 3 Destination Recommendations:")
                cols = st.columns(len(top_df))  # create columns equal to number of destinations
                for idx, row in top_df.iterrows():
                    with cols[idx]:
                        st.subheader(row['DestinationName'])
                        if pd.notna(row['ImageURL']):
                            st.image(row['ImageURL'], use_column_width=True)
                        st.markdown(f"📍 **Location**: {row['Location']}")
                        st.markdown(f"📅 **Best Time to Visit**: {row['BestTimeToVisit']}")
                        st.markdown(f"⭐ **Predicted Popularity Score**: {row['Popularity Score']}")
                        st.markdown("---")
    else:
        st.warning("⚠️ No suitable destinations found based on your preferences.")

