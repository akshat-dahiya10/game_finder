import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GameFinder",
    page_icon="🎮",
    layout="wide"
)

st.title("GameFinder")
st.write("A content-based video game recommendation system.")
st.divider()


# --------------------------------------------------
# LOAD DATASET AND BUILD RECOMMENDER
# --------------------------------------------------

@st.cache_resource
def load_recommender(file_path):

    df = pd.read_csv(file_path, low_memory=False)

    # Required columns
    required_columns = [
        "Name",
        "Genres",
        "Tags",
        "About the game"
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    # Handle missing values
    for column in required_columns:
        df[column] = df[column].fillna("").astype(str)

    # Remove rows without game names
    df = df[df["Name"].str.strip() != ""].copy()

    # Remove duplicate games
    if "AppID" in df.columns:
        df["AppID"] = df["AppID"].astype(str)
        df = df.drop_duplicates(subset=["AppID"])
    else:
        df = df.drop_duplicates(subset=["Name"])

    df = df.reset_index(drop=True)

    # Combine game metadata into one feature
    # Genres and Tags receive additional weight
    df["combined_features"] = (
        (df["Genres"] + " ") * 2
        + (df["Tags"] + " ") * 2
        + df["About the game"]
    )

    # Convert text into numerical TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=150000,
        min_df=1,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_features"]
    )

    # Train nearest-neighbor model
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(tfidf_matrix)

    return df, tfidf_matrix, model


# --------------------------------------------------
# INITIALIZE MODEL
# --------------------------------------------------

DATASET_PATH = "games_filtered_40mb.csv"

try:
    df, tfidf_matrix, model = load_recommender(DATASET_PATH)

except FileNotFoundError:
    st.error(
        "Dataset not found. Please place "
        "games_filtered_40mb.csv in the project folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the recommendation system: {error}")
    st.stop()


st.success(f"Loaded {len(df):,} games successfully.")


# --------------------------------------------------
# GAME SEARCH
# --------------------------------------------------

st.subheader("Find a game")

search_query = st.text_input(
    "Enter a game title",
    placeholder="Search for a game..."
)

if search_query.strip():

    matching_games = df[
        df["Name"].str.contains(
            search_query,
            case=False,
            regex=False,
            na=False
        )
    ]

    if matching_games.empty:
        st.warning("No matching games found.")

    else:

        # Limit dropdown size
        matching_games = matching_games.head(100)

        game_options = {
            f"{row['Name']} ({row.name})": row.name
            for _, row in matching_games.iterrows()
        }

        selected_game = st.selectbox(
            "Select a game",
            options=list(game_options.keys())
        )

        selected_index = game_options[selected_game]

        # --------------------------------------------------
        # RECOMMENDATION SETTINGS
        # --------------------------------------------------

        number_of_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=10,
            value=5
        )

        if st.button("Find Similar Games", type="primary"):

            # Find games closest to the selected game
            distances, indices = model.kneighbors(
                tfidf_matrix[selected_index],
                n_neighbors=len(df)
            )

            recommendations = []
            selected_name = df.iloc[selected_index]["Name"]

            for distance, index in zip(
                distances[0],
                indices[0]
            ):

                # Skip the selected game
                if index == selected_index:
                    continue

                game = df.iloc[index]

                # Avoid duplicate game titles
                if game["Name"] == selected_name:
                    continue

                similarity = max(0, 1 - distance) * 100

                recommendations.append({
                    "game": game,
                    "similarity": similarity
                })

                if len(recommendations) >= number_of_recommendations:
                    break

            # --------------------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------------------

            st.divider()
            st.subheader("Recommended Games")

            if not recommendations:
                st.warning("No recommendations could be generated.")

            else:

                for i, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    game = recommendation["game"]
                    similarity = recommendation["similarity"]

                    with st.container(border=True):

                        st.markdown(
                            f"### {i}. {game['Name']}"
                        )

                        st.caption(
                            f"Metadata similarity: {similarity:.2f}%"
                        )

                        if game["Genres"].strip():
                            st.write(
                                f"**Genres:** {game['Genres']}"
                            )

                        if game["Tags"].strip():
                            st.write(
                                f"**Tags:** {game['Tags']}"
                            )

                        if "Price" in df.columns:
                            price = game["Price"]

                            if pd.notna(price):
                                st.write(f"**Price:** {price}")

                        description = game["About the game"]

                        if description.strip():
                            with st.expander("Game description"):
                                st.write(description)

            st.caption(
                "Recommendations are based on similarities in game "
                "metadata, including genres, tags and descriptions. "
                "Similarity scores do not represent user ratings "
                "or guaranteed enjoyment."
            )
