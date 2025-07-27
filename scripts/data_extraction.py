import os
import pandas as pd
from typing import Tuple
import numpy as np

def time_and_date_aggregation(history: pd.DataFrame) -> pd.DataFrame:
    """Converts raw listening history to cyclically encoded time features for clustering."""
    date_time = pd.to_datetime(history['ts'], utc=True).dt.tz_convert('US/Eastern')
    dow = date_time.dt.weekday
    time_minutes = date_time.dt.hour * 60 + date_time.dt.minute

    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    time_sin = np.sin(2 * np.pi * time_minutes / 1440)
    time_cos = np.cos(2 * np.pi * time_minutes / 1440)

    new = pd.DataFrame({
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'time_sin': time_sin,
        'time_cos': time_cos,
        'track': history['master_metadata_track_name'],
        'artist': history['master_metadata_album_artist_name'],
        'album': history['master_metadata_album_album_name'],
        'uri': history['spotify_track_uri']
    })

    return new

def song_aggregation(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregates listening data by unique track URI with metrics for popularity scoring."""
    aggregated = history.groupby('spotify_track_uri').agg(
        count=('spotify_track_uri', 'count'),
        track=('master_metadata_track_name', 'first'),
        artist=('master_metadata_album_artist_name', 'first'),
        album=('master_metadata_album_album_name', 'first'),
        min_listened=('ms_played', 'sum'),
        shuffle=('shuffle', 'sum'),
        skip=('skipped', 'sum'),
        uri=('spotify_track_uri', 'first')
    ).reset_index()
    aggregated['min_listened'] = aggregated['min_listened'] / 60000
    aggregated.drop('spotify_track_uri', axis=1, inplace=True)

    return aggregated.sort_values(by='count', ascending=False).reset_index(drop=True)

def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and processes all JSON files from the specified path into time-encoded and aggregated datasets."""
    history = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            file_path = os.path.join(path, file)
            df = pd.read_json(file_path)
            history.append(df)
    
    if history:
        history = pd.concat(history, ignore_index=True)
    else:
        history = pd.DataFrame()

    full_history = time_and_date_aggregation(history)
    song_history = song_aggregation(history)
    
    return full_history, song_history