# SmartList 🎵  
*Intelligent Spotify playlist generation through temporal listening behavior clustering*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)  [![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org)

---

## 🚀 Overview

SmartList transforms your Spotify “Extended Streaming History” into context‑aware Daylists by clustering **when** you listen, not just **what**. Instead of generic algorithms, it builds playlists that mirror your real weekly rhythms.

---

## 🧠 Technical Approach

### 1. Data Extraction (`scripts/data_extraction.py`)
- **Loads** raw JSON files exported from Spotify.  
- **Aggregates** play sessions into `full_history` (time features + metadata) and `song_history` (track‑level stats).

### 2. Temporal Feature Engineering
- **Day‑of‑week** & **time‑of‑day** converted into **cyclical features**:  
  ```python
  dow_sin = sin(2π × day_of_week / 7)
  dow_cos = cos(2π × day_of_week / 7)
  time_sin = sin(2π × minutes_since_midnight / 1440)
  time_cos = cos(2π × minutes_since_midnight / 1440)
  ```

### 3. Clustering (`scripts/clustering.py`)
- **K‑Means** on 4‑dimensional cyclical time space (n_clusters=50).  
- **Silhouette Score** ≈ 0.53, indicating well‑separated listening patterns.  
- **Summarization**: maps clusters to a dominant day, time range, and descriptive name.

### 4. Playlist Generation (`scripts/clustering.py`)
- **Popularity Score**:  
  ```python
  popularity_score = 0.6 * play_count + 0.3 * minutes_listened - 0.1 * skip_count
  ```  
  *Balances total plays, listening depth, and penalizes skips.*

- **Three‑Stage Sampling**:  
  1. **Stage 1: Curated Favorites**  
     - **Action**: Randomly select **10** tracks from `top_songs[cluster_id]` (your top played songs for that cluster).  
     - **Reason**: Ensures each playlist anchors around the cluster’s signature tracks—your core favorites.
  2. **Stage 2: Cluster Hits**  
     - **Action**: Randomly sample **10** tracks from the **top 100** by `popularity_score` in the full cluster.  
     - **Reason**: Introduces global favorites and deep cuts that rank highly but didn’t make the cluster-based top list.
  3. **Stage 3: Contextual Fill**  
     - **Action**: Randomly sample **10** additional tracks from the cluster where `popularity_score ≥ min_popularity`.  
     - **Reason**: Fills out the playlist with contextually appropriate tracks, maintaining a quality threshold.

- **Uniqueness & Reproducibility**  
  - Enforce **no duplicate** URIs or artists across all 30 tracks.  
  - Shuffles freshly on each run, so you get a new random playlist every time

### 5. Web App (`main.py`)
- **Flask** app serving:
  - **Dashboard**: cluster cards organized by day, shows silhouette tag.  
  - **Cluster pages**: playlist preview, top artists/songs, CSV download.  
- Clean, Spotify‑inspired design with responsive layout.

### 6. Interactive Notebook (`pipeline.ipynb`)
- Step‑by‑step EDA:  
  - Data loading → clustering → summary tables → playlist samples.

## 🛠️ Getting Started

1. **Install dependencies**  
   ```bash
   pip install flask pandas scikit-learn numpy
   ```
2. **Prepare data**  
   - Download “Extended Streaming History” JSON from Spotify.  
   - Place files in `data/Raw Data/`.  
3. **Run notebook**  
   ```bash
   jupyter notebook pipeline.ipynb
   ```
4. **Launch web app**  
   ```bash
   python main.py
   ```
   Visit `http://localhost:5001`.

---

## 🔮 Future Directions

- **Audio‑feature integration** (tempo, valence, energy) for mood enrichment.  
- **Real‑time Spotify API sync** for live updates.  
- **Advanced clustering** (e.g., DBSCAN) to capture irregular listening patterns.

---

*Built with Python, scikit‑learn, and Flask.*  
