# GameFinder – Video Game Recommendation System

## 1. Project Overview

GameFinder is a content-based video game recommendation system built using Python and Streamlit.

The application recommends games based on similarities in their genres, tags, and descriptions.

Users can search for a game, select it, and receive a list of similar games.

## 2. Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Cosine Similarity

## 3. Recommendation Methodology

The project uses content-based filtering.

### Step 1: Data Preprocessing

The dataset is loaded using Pandas.

Missing values are handled, duplicate games are removed, and relevant metadata is extracted.

### Step 2: Feature Combination

Three attributes are combined to represent each game:

- Genres
- Tags
- About the game

Genres and tags are given additional weight because they provide useful information about game categories and characteristics.

### Step 3: TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts textual information into numerical vectors.

This allows the system to compare games mathematically.

### Step 4: Similarity Calculation

The system uses cosine distance through Scikit-learn's NearestNeighbors algorithm.

Games with smaller cosine distances have more similar metadata.

### Step 5: Recommendations

When a user selects a game, the system identifies the nearest games in the feature space and displays the most similar results.

## 4. Dataset

The project uses the Steam video game dataset.

The following dataset columns are used:

- Name
- Genres
- Tags
- About the game
- Price (optional)

The dataset file must be named:

`games_filtered_40mb.csv`

Place the dataset in the same directory as `app.py`.

## 5. Installation

Install Python 3.9 or later.

Open a terminal in the project directory and execute:

```bash
pip install -r requirements.txt
```

## 6. Running the Application

Execute the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 7. Project Structure

```text
GameFinder/
│
├── app.py
├── requirements.txt
├── README.md
└── games_filtered_40mb.csv
```

## 8. Limitations

- Recommendations depend on available game metadata.
- The system does not use individual user preferences or gameplay history.
- Similarity scores represent metadata similarity, not game quality.
- Recommendations may be affected by incomplete or inaccurate dataset information.

## 9. Future Improvements

- Incorporate user ratings and reviews.
- Add collaborative filtering.
- Improve search functionality.
- Include additional game attributes.

## 10. Conclusion

GameFinder demonstrates how content-based filtering can be used to recommend video games.

By combining TF-IDF vectorization with cosine-distance-based nearest-neighbor search, the application identifies games with similar characteristics and presents them through a simple user interface.
