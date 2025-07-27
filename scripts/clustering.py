from sklearn.cluster import KMeans
import random
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def get_top_artists_per_cluster(clustered_df: pd.DataFrame, top_n: int = 5) -> dict:
    """Returns the top artists by total listens per cluster."""
    top_artists = {}

    for cluster_id, group in clustered_df.groupby('cluster'):
        artist_counts = group['artist'].value_counts().nlargest(top_n)
        df = artist_counts.reset_index()
        df.columns = ['artist', 'listen_count']
        top_artists[cluster_id] = df

    return top_artists

def get_top_songs_per_cluster(clustered_df: pd.DataFrame, top_n: int = 50) -> dict:
    """Returns the top songs by total listens per cluster."""
    top_songs = {}

    for cluster_id, group in clustered_df.groupby('cluster'):
        song_counts = (
            group.groupby(['track', 'artist', 'uri', 'album'])
            .size()
            .reset_index(name='listen_count')
            .sort_values(by='listen_count', ascending=False)
            .head(top_n)
        )
        top_songs[cluster_id] = song_counts

    return top_songs

def calculate_silhouette_score(full_history: pd.DataFrame) -> float:
    """Computes the silhouette score for the clustered data."""
    features = full_history[['dow_sin', 'dow_cos', 'time_sin', 'time_cos']]
    labels = full_history['cluster']
    return silhouette_score(features, labels)

def k_means_clustering(full_history: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Performs K-Means clustering on cyclical time features and returns DataFrame with cluster assignments."""
    features = full_history[['dow_sin', 'dow_cos', 'time_sin', 'time_cos']]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    full_history['cluster'] = kmeans.fit_predict(features)

    return full_history

def add_popularity_scores(song_history: pd.DataFrame) -> pd.DataFrame:
    """Adds a custom popularity score to the song aggregation data."""
    song_history['popularity_score'] = (
        0.6 * song_history['count'] +
        0.3 * song_history['min_listened'] -
        0.1 * song_history['skip']
    )
    return song_history


def generate_cluster_playlists(clustered_df: pd.DataFrame,
                               popularity_df: pd.DataFrame,
                               top_songs: dict,
                               top_n: int = 30,
                               min_popularity: float = 10.0) -> dict:
    """
    Three‑stage playlist (random each run):
      1) 10 random from top_songs[cluster_id]
      2) 10 random from top 100 popular in cluster
      3) 10 random from cluster with pop_score ≥ min_popularity
    Ensures unique URIs and artists, but shuffles differently each execution.
    """
    playlists = {}

    for cluster_id in clustered_df['cluster'].unique():
        used_uris = set()
        used_artists = set()
        selected = []

        ts = top_songs.get(cluster_id, pd.DataFrame())
        if not ts.empty:
            for _, track in ts.sample(frac=1).head(10).iterrows():
                if track['uri'] in used_uris or track['artist'] in used_artists:
                    continue
                selected.append(track)
                used_uris.add(track['uri'])
                used_artists.add(track['artist'])

        cluster_uris = set(clustered_df.loc[clustered_df['cluster']==cluster_id, 'uri'])
        cluster_pop = popularity_df[popularity_df['uri'].isin(cluster_uris)]
        top100 = cluster_pop.nlargest(100, 'popularity_score')
        for _, track in top100.sample(frac=1).iterrows():
            if len(selected) >= 20:
                break
            if track['uri'] in used_uris or track['artist'] in used_artists:
                continue
            selected.append(track)
            used_uris.add(track['uri'])
            used_artists.add(track['artist'])

        eligible = cluster_pop[cluster_pop['popularity_score'] >= min_popularity]
        for _, track in eligible.sample(frac=1).iterrows():
            if len(selected) >= 30:
                break
            if track['uri'] in used_uris or track['artist'] in used_artists:
                continue
            selected.append(track)
            used_uris.add(track['uri'])
            used_artists.add(track['artist'])

        if selected:
            df = pd.DataFrame(selected).head(top_n).reset_index(drop=True)
            playlists[cluster_id] = df[['track','artist','album','uri','popularity_score']]
        else:
            playlists[cluster_id] = pd.DataFrame(
                columns=['track','artist','album','uri','popularity_score']
            )

    return playlists

def summarize_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary of each cluster's dominant day, time range, and a Daylist-style descriptive name."""
    import random

    clustered_df = clustered_df.copy()

    clustered_df['dow'] = np.arctan2(clustered_df['dow_sin'], clustered_df['dow_cos']) * (7 / (2 * np.pi))
    clustered_df['dow'] = clustered_df['dow'].apply(lambda x: x if x >= 0 else x + 7).round().astype(int)
    clustered_df['dow_name'] = clustered_df['dow'].apply(lambda d: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][d % 7])

    clustered_df['time'] = np.arctan2(clustered_df['time_sin'], clustered_df['time_cos']) * (1440 / (2 * np.pi))
    clustered_df['time'] = clustered_df['time'].apply(lambda x: x if x >= 0 else x + 1440).astype(int)

    def mins_to_str(mins):
        return f"{mins // 60:02}:{mins % 60:02}"

    mood_descriptors = {
        "Morning": ["mellow", "hopeful", "slow", "quiet", "cozy", "fresh"],
        "Afternoon": ["sunny", "focused", "lazy", "casual", "bright", "wandering"],
        "Evening": ["chill", "moody", "breezy", "romantic", "cool", "hazy"],
        "Late Night": ["restless", "chaotic", "soft", "dreamy", "lonely", "electric"],
        "All Day": ["flowy", "familiar", "nostalgic", "rhythmic", "mixed", "steady"]
    }

    def get_time_label(min_time, max_time):
        mid = (min_time + max_time) // 2
        if max_time - min_time >= 1200:
            return "All Day"
        elif 300 <= mid < 720:
            return "Morning"
        elif 720 <= mid < 1020:
            return "Afternoon"
        elif 1020 <= mid < 1320:
            return "Evening"
        else:
            return "Late Night"

    summaries = []

    for cluster_id, group in clustered_df.groupby('cluster'):
        day_counts = group['dow_name'].value_counts()
        dominant_day = day_counts.idxmax()

        min_time = group['time'].min()
        max_time = group['time'].max()
        time_range = f"{mins_to_str(min_time)}–{mins_to_str(max_time)}"

        time_of_day = get_time_label(min_time, max_time)
        mood = random.choice(mood_descriptors[time_of_day])

        name = f"{dominant_day} {mood} {time_of_day.lower()}"

        summaries.append({
            'cluster': cluster_id,
            'day': dominant_day,
            'time_range': time_range,
            'count': len(group),
            'name': name
        })

    return pd.DataFrame(summaries)