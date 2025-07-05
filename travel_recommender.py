import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model and encoders
features = ['Name_x', 'Type', 'BestTimeToVisit', 'Preferences', 'NumberOfAdults', 'NumberOfChildren']
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Load datasets
destinations_df = pd.read_csv("code and dataset/Expanded_Destinations.csv")
userhistory_df = pd.read_csv("code and dataset/Final_Updated_Expanded_UserHistory.csv")
df = pd.read_csv("code and dataset/final_df.csv")

# Create user-item matrix and compute user similarity
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')
user_item_matrix.fillna(0, inplace=True)
user_similarity = cosine_similarity(user_item_matrix)

# Collaborative filtering function
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    similar_users = user_similarity[user_id - 1]
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
        'DestinationID', 'Name', 'Type', 'Popularity', 'BestTimeToVisit'
    ]]
    return recommendations

# Popularity prediction function
def recommend_destinations(user_input, model, label_encoders, features):
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]
    input_df = pd.DataFrame([encoded_input])
    predicted_popularity = model.predict(input_df)[0]
    return predicted_popularity

# Streamlit UI
st.set_page_config(page_title="Travel Recommender", layout="wide")
st.title("✈️ Personalized Travel Recommendation System")

st.markdown("Enter your details below to get travel destination recommendations.")

with st.form("user_input_form"):
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    name = st.selectbox("Select Destination Name", destinations_df['Name'].unique())
    type_ = st.selectbox("Select Destination Type", destinations_df['Type'].unique())
    best_time = st.selectbox("Best Time To Visit", destinations_df['BestTimeToVisit'].unique())
    preferences = st.selectbox("Your Preferences", df['Preferences'].unique())
    num_adults = st.slider("Number of Adults", 1, 10, 2)
    num_children = st.slider("Number of Children", 0, 5, 0)

    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    if user_id not in userhistory_df['UserID'].values:
        st.warning("User ID not found in user history data. Please enter a valid User ID.")
    else:
        user_input = {
            'Name_x': name,
            'Type': type_,
            'BestTimeToVisit': best_time,
            'Preferences': preferences,
            'NumberOfAdults': num_adults,
            'NumberOfChildren': num_children
        }

        # Get collaborative recommendations
        recommendations = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)
        
        # Get popularity prediction for user input
        predicted_popularity = recommend_destinations(user_input, model, label_encoders, features)

        st.success(f"✅ Predicted Popularity Score for selected destination: {predicted_popularity:.2f}")

        if not recommendations.empty:
            st.subheader(f"Top destination recommendations for User ID {user_id}:")
            for _, row in recommendations.iterrows():
                st.markdown(f"**{row['Name']}**")
                st.markdown(f"- Type: {row['Type']}")
                st.markdown(f"- Best Time to Visit: {row['BestTimeToVisit']}")
                st.markdown(f"- Popularity Score: {row['Popularity']:.2f}")
                st.markdown("---")
        else:
            st.info("No recommendations available.")



