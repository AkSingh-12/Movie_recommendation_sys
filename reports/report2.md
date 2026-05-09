# Status Report 2: Implementation & Evaluation

## Methodology
**Overview:** End-to-end pipeline from raw TMDB data to personalized recs, integrating emotion signals.

1. **Data Collection**: TMDB scraper (`src/scraper.py`): Start with popular movies API → fetch details (id,title,genres,release_date,director,cast,overview,rating,vote_count,popularity). Dedupe by tmdb_id/title fuzzy. Append-only to movies.csv (~300 rows currently).

2. **Feature Engineering**: Movie 'soup' = title + '|'.join(genres) + director + '|'.join(cast[:3]) + overview. TF-IDF vectorization (scikit-learn TfidfVectorizer, max_features=20000, ngram_range=(1,2), stop_words='english') → sparse matrix saved as .npz.

   **TF-IDF Formula Recap:**
   $$
   tfidf(t,d) = tf(t,d) \times \log\left(\frac{N}{df(t)}\right)
   $$

3. **Similarity Computation**: Precompute cosine similarity matrix (linear_kernel on TF-IDF). Genre centroids: mean vector per genre (from movies.csv genre parsing).

4. **Recommendation Flow**:
   | Mode | Steps |
   |------|-------|
   | Title Search | fuzzywuzzy token_sort → top TF-IDF sim |
   | Genre Rec | Filter movies by genre → sim to centroid |
   | Mood Rec | mood → genres → genre rec + personalize |

**Mood-to-Genre Mapping (Empirical):**
| Mood | Primary Genres | Secondary |
|------|----------------|-----------|
| happy | Comedy, Romance | Animation, Family |
| angry | Action, Thriller | Crime, Horror |
| calm | Documentary, Drama | History |
| sad | Romance, Drama | Biography |

```python
# ex: src/recommender.py (via wrapper api/recomender)
def recommend_by_mood(mood, user_profile, top_n=10):
    genres = MOOD_GENRES[mood]
    candidates = filter_movies_by_genres(df_movies, genres)
    sim_scores = cosine_sim[genre_centroid_idx[genres[0]]]
    boosts = personalization_model.predict(user_features(mood, profile))
    final_scores = sim_scores + np.clip(boosts, -1.2, 1.2)
    return top_n_movies(candidates, final_scores)
```


## Data Warehouse Architecture
File-based (no DB for simplicity/local-first deployment):

```
data/
├── movies.csv          # TMDB schema: tmdb_id,title,genres (pipe-delim),release_date,director,cast (pipe-delim),overview,rating,vote_count,popularity (~350 rows, scraped incrementally)
├── feedback_events.jsonl # Append-only: {\"timestamp\":\"2024-...\",\"detected_mood\":\"angry\",\"signal\":1.0,\"movie_title\":\"Avengers: Endgame\",\"genres\":[\"Action\",\"Adventure\"]} (16 events so far)
├── user_learning.json # Profile state: {\"mood_counts\":{\"angry\":8,\"calm\":6,...},\"mood_genre_weights\":{\"angry-thriller\":3.1},\"title_prefs\":{\"Avengers Endgame\":0.6}}
├── personalization_model.pkl # SGDRegressor (Huber loss, online partial_fit)
├── cache/             # tfidf_matrix.npz (scipy.sparse), genre_centroids.npy, full_sim.npy (precomputed, ~100ms load)
└── emotion_*.csv/png  # Mini-XC confusion matrix, test predictions
├── video_sources.json # Fallback for voice/STT testing
```
**Advantages:**
- Atomic jsonl appends, pickle dumps.
- Version control friendly.
- No DB server deps.

**ETL Workflows:**
1. `src/scraper.py` → movies.csv
2. UI feedback → `src/user_store.py` → jsonl → `src/personalization_model.py train`
3. `src/api.py /rebuild_index` → caches

## Implementation
**Core Modules Detailed:**

- **Emotion Detection (`src/emotion_detection.py`)**: 
  - OpenCV Haar cascade face detect → CLAHE histogram eq → resize(48x48 grayscale) → Mini-XCEPTION Keras predict (7 softmax: angry/disgust/fear/happy/neutral/sad/surprise).
  - Confidence threshold 0.6 → map to 4 moods (happy/angry/calm/sad).
  ```python
  def detect_mood_from_frame(frame):  # BGR frame → 'happy' or None
      faces = detect_faces(frame)
      if faces:
          face_crop = preprocess_face(faces[0])
          probs = model.predict(face_crop)[0]
          emotion = EMOTION_MAP[np.argmax(probs)]
          if max(probs) > CONF_THRESH:
              return map_emotion_to_mood(emotion)
      return None
  ```

- **Multimodal Mood (`src/multimodal_mood.py`)**: Weighted fuse face_mood (0.7) + voice_mood (0.3). Voice: librosa RMS energy/ZCR, threshold rules (high RMS→angry).
  - Fallback: browser SpeechRecognition API for keyword moods.

- **Feedback & Personalization (`src/user_store.py`, `src/personalization_model.py`)**: 
  - Append jsonl: timestamp, detected_mood, signal (+1/-1), movie_title/genres.
  - Profile: mood counts, learned weights (genre/mood pairs).
  - Rerank: base_sim + rule_boost (genre match) + model_boost (SGD.predict([mood_onehot, genre_onehot, past_signal_avg])).

- **API (`src/api.py` FastAPI)**: 
  - POST /recommend: {"query":str, "mood":str, "top_n":int} → list[dict(title,poster_url,score)]
  - POST /feedback: {"movie":str, "signal":float}
  - GET /refresh → rebuild caches.

- **UI (`web/app_streamlit.py`)**: Real-time webcam capture → mood display → 5 rec cards (TMDB posters) → thumbs → auto-feedback.

**Testing & Deploy:**
- pytest: 90% coverage (unit: emotion batch, rec accuracy; integration: api).
- Playwright UI tests (`tests/playwright_ui_test.py`): mood→rec→feedback cycle.
- Docker-compose: api+ui+scraper services; systemd units for prod.

## RESULTS / OUTPUTS
- **Dataset**: ~350 TMDB movies (Action/Thriller dominant ~40%; full genre dist below).
  Genre Distribution (top 10):
  | Genre | Count | % |
  |-------|-------|---|
  | Action | 85 | 24% |
  | Drama | 62 | 18% |
  | Comedy | 45 | 13% |

- **Feedback Events (16 total, `data/feedback_events.jsonl`)**:
  | Mood | Count | + Signals | - Signals | Top Genres (weighted) |
  |------|-------|-----------|-----------|-----------------------|
  | calm | 6 | 4 | 2 | documentary(2.4), comedy(1.2) |
  | angry | 8 | 7 | 1 | thriller(3.0), action(3.2) |
  | happy | 1 | 1 | 0 | action(0.6) |
  | sad | 1 | 0 | 1 | romance(-0.4) |

- **User Learning (`data/user_learning.json`)**: Example weights: angry-adventure(1.8), calm-documentary(2.1); title prefs "Avengers: Endgame"(+0.6), "Inside Out"(-0.2).

- **Personalization Model**: SGDRegressor(Huber), 19 samples → MAE 0.57 (train). Features: mood/genre onehot, past avg signal. Progress ~10% to target (200 events).

**Emotion Model Eval (`data/emotion_confusion_matrix.csv`)**: Mini-XC accuracy 62% on 100 test frames (conf >0.6 filter).

- **Outputs & Demos**:
  - Real-time: angry mood → "John Wick", "Mad Max" (scores 0.92, 0.88).
  - UI: http://localhost:8501 (webcam live).
  - Query latency: <800ms (caches hit).


## Performance Benchmarks
| Component | Latency (ms) | Accuracy/Notes |
|-----------|--------------|----------------|
| Emotion Detect | 150-250 | 62% on test frames |
| TF-IDF Query | <100 | Cache hit |
| Full Rec | 400-800 | End-to-end |
| Model Predict | 5 | Vectorized |

Text-based Emotion Confusion (top errors):
```
Predicted \ True | happy | angry | calm | sad
happy            | 15    | 2     | 1    | 0
angry            | 3     | 18    | 0    | 1
...
(From data/emotion_confusion_matrix.csv)
```

## Limitations & Future Work
**Current Limits:**
- Small feedback dataset (16 → model immature).
- No collaborative filtering.
- Webcam-only (mobile?).

**Planned Enhancements:**
1. Collect 200+ events via UI sessions.
2. Voice STT (Whisper local) for richer multimodal.
3. User segmentation (multiple profiles).
4. Cloud deploy (TMDB rate limits handled).

Progress: Learning stage (target 200 events); functional prototype achieved.

