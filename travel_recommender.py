import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load saved model & encoders ---
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = ['DestinationName', 'State', 'Category', 'BestTimeToVisit',
            'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

# --- Load all datasets ---
destinations_df = pd.read_csv("Destinations.csv")
reviews_df = pd.read_csv("Reviews.csv")
userhistory_df = pd.read_csv("UserHistory.csv")
users_df = pd.read_csv("Users.csv")

# Rename to match UI
destinations_df.rename(columns={'Type': 'Category'}, inplace=True)

# --- Add destination images ---
image_map = {
    'Jaipur City': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg/1024px-East_facade_Hawa_Mahal_Jaipur_from_ground_level_%28July_2022%29_-_img_01.jpg',
    'Taj Mahal': 'https://upload.wikimedia.org/wikipedia/commons/d/da/Taj-Mahal.jpg',
    'Kerala Backwaters': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Backwaters_in_Kerala.jpg',
    'Goa Beaches': 'https://upload.wikimedia.org/wikipedia/commons/4/48/Palolem_beach_sunset_Goa.jpg',
    'Leh Ladakh': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Leh_Ladakh_-_Pangong_Lake.jpg',
}
destinations_df['ImageURL'] = destinations_df['DestinationName'].map(image_map)

# --- Merge data for content-based filtering ---
df = reviews_df.merge(destinations_df, on='DestinationID') \
               .merge(userhistory_df, on='UserID') \
               .merge(users_df, on='UserID')
df.drop_duplicates(inplace=True)
df['features'] = df['Category'] + ' ' + df['State'] + ' ' + df['BestTimeToVisit'] + " " + df['Preferences']

# --- TF-IDF & cosine similarity for content-based ---
vectorizer = TfidfVectorizer(stop_words='english')
destination_features = vectorizer.fit_transform(df['features'])
cosine_sim = cosine_similarity(destination_features)

# --- Collaborative filtering: user similarity ---
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)

# --- Content-based Recommendation ---
def recommend_content(user_id, top_n=3):
    visited = userhistory_df[userhistory_df['UserID'] == user_id]['DestinationID'].values
    if len(visited) == 0:
        return pd.DataFrame()

    similarity_scores = np.sum(cosine_sim[visited - 1], axis=0)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []
    for idx in sorted_indices:
        dest_id = destinations_df.iloc[idx]['DestinationID']
        if dest_id not in visited:
            recommendations.append(destinations_df.iloc[idx])
        if len(recommendations) == top_n:
            break
    return pd.DataFrame(recommendations)

# --- Collaborative filtering Recommendation ---
def recommend_collab(user_id, top_n=3):
    if user_id - 1 not in range(user_similarity.shape[0]):
        return pd.DataFrame()

    sim_scores = user_similarity[user_id - 1]
    sim_users = np.argsort(sim_scores)[::-1]
    sim_users = sim_users[sim_users != (user_id - 1)][:top_n]

    avg_ratings = user_item_matrix.iloc[sim_users].mean(axis=0)
    top_dest_ids = avg_ratings.sort_values(ascending=False).head(top_n).index
    return destinations_df[destinations_df['DestinationID'].isin(top_dest_ids)]

# --- Streamlit App UI ---
st.set_page_config(page_title="Travel Recommendation App", layout="wide")
st.title("🏝️ Personalized Travel Recommendation System")
st.markdown("Get destination suggestions based on your preferences and travel behavior.")

user_id = st.number_input("🔢 Enter your User ID:", min_value=1, max_value=users_df['UserID'].max(), value=1, step=1)

if st.button("🔍 Recommend Destinations"):
    content_df = recommend_content(user_id)
    collab_df = recommend_collab(user_id)

    # Combine and deduplicate
    hybrid_df = pd.concat([content_df, collab_df]).drop_duplicates(subset=['DestinationID'])
    hybrid_df = hybrid_df.sort_values(by='Popularity', ascending=False).head(3).reset_index(drop=True)

    if not hybrid_df.empty:
        st.success(f"✅ Top {len(hybrid_df)} Destination Recommendations for User {user_id}:\n")
        for _, row in hybrid_df.iterrows():
            with st.container():
                st.subheader(f"📍 {row['DestinationName']}")
                if pd.notna(row['ImageURL']):
                    st.image(row['ImageURL'], use_column_width=True)
                st.markdown(f"**Location:** {row['State']}")
                st.markdown(f"**Best Time to Visit:** {row['BestTimeToVisit']}")
                st.markdown(f"**Popularity Score:** ⭐ {round(row['Popularity'], 2)}")
                st.markdown("---")
    else:
        st.warning("😕 No recommendations available for this user.")

