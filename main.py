from flask import Flask, render_template_string, jsonify, send_file
import pandas as pd
import os
import random
import hashlib
from scripts.data_extraction import load_data
from scripts.clustering import (
    k_means_clustering,
    add_popularity_scores,
    generate_cluster_playlists,
    summarize_clusters,
    calculate_silhouette_score,
    get_top_artists_per_cluster,
    get_top_songs_per_cluster
)

app = Flask(__name__)

def get_dynamic_gradient(cluster_id, time_label):
    """Generates unique color gradients for each cluster based on time period and cluster ID."""
    random.seed(cluster_id)
    
    gradient_sets = {
        'morning': [
            ['#FFE5B4', '#FFADAD'],
            ['#FFEAA7', '#DDA0DD'],
            ['#FFF2CC', '#FFB6C1'],
            ['#E6E6FA', '#FFE4E1'],
            ['#F0E68C', '#FFA07A'],
        ],
        'afternoon': [
            ['#FFD76E', '#FF9671'],
            ['#74B9FF', '#0984E3'],
            ['#FDCB6E', '#E17055'],
            ['#00CEC9', '#55EFC4'],
            ['#A29BFE', '#6C5CE7'],
        ],
        'evening': [
            ['#828AB9', '#4E54C8'],
            ['#FD79A8', '#E84393'],
            ['#FDCB6E', '#E17055'],
            ['#00B894', '#00A085'],
            ['#6C5CE7', '#A29BFE'],
        ],
        'late_night': [
            ['#2E335A', '#232536'],
            ['#1A1A2E', '#16213E'],
            ['#0F3460', '#533483'],
            ['#2C2C54', '#40407A'],
            ['#1B1464', '#663399'],
        ],
        'all_day': [
            ['#89F7FE', '#66A6FF'],
            ['#FDBB2D', '#22C1C3'],
            ['#FF9A9E', '#FECFEF'],
            ['#A8EDEA', '#FED6E3'],
            ['#D299C2', '#FEF9D7'],
        ]
    }
    
    available_gradients = gradient_sets.get(time_label, gradient_sets['all_day'])
    selected_gradient = available_gradients[cluster_id % len(available_gradients)]
    
    return selected_gradient

def process_time_label(time_range):
    """Converts time range string to categorical time period label."""
    start_time = int(time_range.split('–')[0].split(':')[0]) * 60 + int(time_range.split('–')[0].split(':')[1])
    if 300 <= start_time < 720:
        return 'morning'
    elif 720 <= start_time < 1020:
        return 'afternoon'
    elif 1020 <= start_time < 1320:
        return 'evening'
    else:
        return 'late_night'

# Data Loading & Processing Flow
data_path = '/Users/zoehightower/Desktop/SmartList/data/Raw Data'
history_data, song_data = load_data(data_path)  # Load raw JSON files
clustered_df = k_means_clustering(history_data, n_clusters=50)  # Cluster listening sessions
sil_score = calculate_silhouette_score(clustered_df)  # Evaluate clustering quality

# Generate cluster summaries with visual styling
summaries = summarize_clusters(clustered_df)
summaries['time_label'] = summaries['time_range'].apply(process_time_label)
summaries['start_minutes'] = summaries['time_range'].apply(lambda tr: int(tr.split('–')[0].split(':')[0]) * 60 + int(tr.split('–')[0].split(':')[1]))

# Add dynamic gradients to each cluster
for idx, row in summaries.iterrows():
    gradient = get_dynamic_gradient(row['cluster'], row['time_label'])
    summaries.loc[idx, 'gradient_start'] = gradient[0]
    summaries.loc[idx, 'gradient_end'] = gradient[1]

# Group clusters by day for UI organization
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
clusters_by_day = {
    d: sorted(
        [r for r in summaries.to_dict(orient='records') if r['day']==d],
        key=lambda r: r['start_minutes']
    ) for d in days
}

# Generate playlists and top lists for each cluster
song_data_scored = add_popularity_scores(song_data)
top_artists = get_top_artists_per_cluster(clustered_df)
top_songs = get_top_songs_per_cluster(clustered_df, top_n=50)
playlists = generate_cluster_playlists(clustered_df, song_data_scored, top_songs, top_n=30)

INDEX_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartList</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      margin: 0; 
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
      background: #121212; 
      color: #FFF; 
      line-height: 1.4;
    }
    .container { 
      padding: 1.5rem; 
      max-width: 1200px; 
      margin: 0 auto; 
    }
    .header { 
      display: flex; 
      align-items: center; 
      justify-content: space-between; 
      margin-bottom: 2rem; 
      padding-bottom: 1rem;
      border-bottom: 1px solid #333;
    }
    .header h1 { 
      color: #1DB954; 
      font-size: 2.5rem; 
      font-weight: 800;
      margin: 0;
    }
    .tag { 
      background: linear-gradient(135deg, #1DB954, #1ed760); 
      padding: 0.5rem 1rem; 
      border-radius: 20px; 
      font-size: 0.9rem; 
      font-weight: 600;
      box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
    }
    .day-section { 
      margin-bottom: 3rem; 
    }
    .day-title { 
      font-size: 1.5rem; 
      font-weight: 700;
      margin-bottom: 1rem; 
      color: #FFF;
    }
    .scroll { 
      display: flex; 
      overflow-x: auto; 
      gap: 1.2rem; 
      padding-bottom: 0.5rem;
    }
    .scroll::-webkit-scrollbar {
      height: 6px;
    }
    .scroll::-webkit-scrollbar-track {
      background: #333;
      border-radius: 3px;
    }
    .scroll::-webkit-scrollbar-thumb {
      background: #1DB954;
      border-radius: 3px;
    }
    .card {
      background: #181818; 
      border-radius: 12px; 
      width: 200px; 
      flex-shrink: 0; 
      cursor: pointer;
      transition: all 0.3s ease;
      overflow: hidden;
      box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .card:hover { 
      transform: translateY(-5px);
      box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }
    .cover { 
      height: 160px; 
      position: relative;
      overflow: hidden;
    }
    .cover::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(135deg, var(--start), var(--end));
      opacity: 0.9;
    }
    .cover::after {
      content: '🎵';
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 3rem;
      z-index: 2;
      text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .info { 
      padding: 1rem; 
    }
    .info .name { 
      font-weight: 600; 
      font-size: 1.1rem; 
      margin-bottom: 0.4rem; 
      line-height: 1.3;
    }
    .info .meta { 
      font-size: 0.9rem; 
      color: #B3B3B3; 
    }
    @media (max-width: 768px) {
      .container { padding: 1rem; }
      .header h1 { font-size: 2rem; }
      .card { width: 160px; }
      .cover { height: 120px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>SmartList</h1>
      <span class="tag">🔍 Silhouette: {{ sil_score }}</span>
    </div>
    {% for day, clusters in clusters_by_day.items() %}
      {% if clusters %}
      <div class="day-section">
        <div class="day-title">{{ day }}</div>
        <div class="scroll">
          {% for row in clusters %}
          <div class="card" onclick="location.href='/cluster/{{ row.cluster }}'" style="--start: {{ row.gradient_start }}; --end: {{ row.gradient_end }};">
            <div class="cover"></div>
            <div class="info">
              <div class="name">{{ row.name }}</div>
              <div class="meta">{{ row.time_range }}</div>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endif %}
    {% endfor %}
  </div>
</body>
</html>
'''

CLUSTER_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ summary.name }} - SmartList</title>
  <style>
    * { box-sizing: border-box; }
    body { 
      margin: 0; 
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
      background: #121212; 
      color: #FFF; 
    }
    .hero {
      height: 60vh;
      background: linear-gradient(180deg, {{ summary.gradient_start }}, {{ summary.gradient_end }});
      display: flex;
      justify-content: center;
      align-items: flex-end;
      padding: 2rem;
      position: relative;
    }
    .hero-content {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      max-width: 600px;
      width: 100%;
    }
    .hero::before {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 50px;
      background: linear-gradient(to bottom, transparent, #121212);
    }
    .back-btn {
      position: absolute;
      top: 2rem;
      left: 2rem;
      color: rgba(255,255,255,0.9);
      text-decoration: none;
      font-size: 1.8rem;
      background: rgba(0,0,0,0.3);
      width: 45px;
      height: 45px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }
    .back-btn:hover {
      background: rgba(0,0,0,0.5);
      transform: scale(1.1);
    }
    .download-btn {
      position: absolute;
      top: 2rem;
      right: 2rem;
      background: rgba(255,255,255,0.2);
      color: rgba(255,255,255,0.9);
      text-decoration: none;
      font-size: 1rem;
      padding: 0.75rem 1.5rem;
      border-radius: 25px;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
      border: 1px solid rgba(255,255,255,0.2);
      font-weight: 600;
    }
    .download-btn:hover {
      background: rgba(255,255,255,0.3);
      transform: scale(1.05);
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .playlist-cover {
      width: 200px;
      height: 200px;
      background: rgba(255,255,255,0.15);
      border-radius: 12px;
      margin-bottom: 1.5rem;
      backdrop-filter: blur(20px);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 4rem;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .playlist-title {
      font-size: 2.5rem;
      font-weight: 800;
      margin-bottom: 0.5rem;
      text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .playlist-meta {
      font-size: 1rem;
      opacity: 0.9;
      text-shadow: 0 1px 5px rgba(0,0,0,0.5);
    }
    .content {
      padding: 2rem;
      background: #121212;
      max-width: 600px;
      margin: 0 auto;
    }
    .stats-section {
      display: flex;
      flex-direction: column;
      gap: 2rem;
      margin-bottom: 3rem;
    }
    .stat-card {
      background: #181818;
      border-radius: 12px;
      padding: 1.5rem;
      border-left: 4px solid {{ summary.gradient_start }};
    }
    .stat-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: {{ summary.gradient_start }};
    }
    .stat-item {
      display: flex;
      align-items: center;
      margin-bottom: 0.8rem;
      padding: 0.5rem;
      background: rgba(255,255,255,0.05);
      border-radius: 8px;
    }
    .stat-rank {
      background: {{ summary.gradient_start }};
      color: #000;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 700;
      margin-right: 1rem;
    }
    .playlist-section {
      margin-top: 2rem;
    }
    .section-title {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 1.5rem;
      color: #FFF;
    }
    .track-list {
      width: 100%;
    }
    .track-item {
      display: flex;
      align-items: center;
      padding: 0.8rem;
      margin-bottom: 0.5rem;
      background: #181818;
      border-radius: 8px;
      transition: background 0.2s ease;
    }
    .track-item:hover {
      background: #282828;
    }
    .track-number {
      color: #B3B3B3;
      font-size: 0.9rem;
      width: 30px;
      text-align: center;
      margin-right: 1rem;
    }
    .track-cover {
      width: 50px;
      height: 50px;
      background: linear-gradient(135deg, {{ summary.gradient_start }}, {{ summary.gradient_end }});
      border-radius: 6px;
      margin-right: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      flex-shrink: 0;
    }
    .track-info {
      flex: 1;
      min-width: 0;
    }
    .track-name {
      font-weight: 600;
      font-size: 1rem;
      margin-bottom: 0.2rem;
      color: #FFF;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .track-artist {
      font-size: 0.9rem;
      color: #B3B3B3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .play-btn {
      background: {{ summary.gradient_start }};
      color: #000;
      border: none;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 1.2rem;
      transition: all 0.2s ease;
      margin-left: 1rem;
    }
    .play-btn:hover {
      transform: scale(1.1);
      box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    @media (max-width: 768px) {
      .hero {
        height: 50vh;
        padding: 1.5rem;
      }
      .download-btn {
        position: static;
        margin-top: 1rem;
        display: inline-block;
      }
      .playlist-cover {
        width: 150px;
        height: 150px;
        font-size: 3rem;
      }
      .playlist-title {
        font-size: 2rem;
      }
      .content {
        padding: 1.5rem;
      }
      .stats-section {
        gap: 1rem;
      }
      .track-cover {
        width: 40px;
        height: 40px;
        font-size: 1rem;
      }
      .play-btn {
        width: 35px;
        height: 35px;
        font-size: 1rem;
      }
    }
  </style>
</head>
<body>
  <div class="hero">
    <a href="/" class="back-btn">&larr;</a>
    <a href="/download/{{ cluster_id }}" class="download-btn">📥 Download</a>
    <div class="hero-content">
      <div class="playlist-cover">🎵</div>
      <div class="playlist-title">{{ summary.name }}</div>
      <div class="playlist-meta">{{ summary.time_range }} • {{ playlist|length }} songs</div>
    </div>
  </div>
  
  <div class="content">
    <div class="stats-section">
      <div class="stat-card">
        <div class="stat-title">Top Artists</div>
        {% for artist in top_artists[:3] %}
        <div class="stat-item">
          <div class="stat-rank">{{ loop.index }}</div>
          <div>
            <div style="font-weight: 600;">{{ artist.artist }}</div>
            <div style="font-size: 0.9rem; color: #B3B3B3;">{{ artist.listen_count }} plays</div>
          </div>
        </div>
        {% endfor %}
      </div>
      
      <div class="stat-card">
        <div class="stat-title">Top Songs</div>
        {% for song in top_songs[:3] %}
        <div class="stat-item">
          <div class="stat-rank">{{ loop.index }}</div>
          <div>
            <div style="font-weight: 600;">{{ song.track }}</div>
            <div style="font-size: 0.9rem; color: #B3B3B3;">{{ song.artist }}</div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    
    <div class="playlist-section">
      <div class="section-title">Your Mix</div>
      <div class="track-list">
        {% for track in playlist %}
        <div class="track-item">
          <div class="track-number">{{ loop.index }}</div>
          <div class="track-cover">🎵</div>
          <div class="track-info">
            <div class="track-name">{{ track.track }}</div>
            <div class="track-artist">{{ track.artist }}</div>
          </div>
          <button class="play-btn">▶</button>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
</body>
</html>
'''

# Flask Routes
@app.route('/')
def index():
    """Main dashboard showing all listening clusters organized by day of week."""
    return render_template_string(
        INDEX_HTML,
        clusters_by_day=clusters_by_day,
        sil_score=round(sil_score, 3)
    )

@app.route('/cluster/<int:cluster_id>')
def cluster_detail(cluster_id):
    """Individual cluster page showing playlist, top artists, and top songs."""
    summary = next(r for r in summaries.to_dict('records') if r['cluster'] == cluster_id)
    ta = top_artists[cluster_id].head(3).to_dict('records')
    ts = top_songs[cluster_id].head(3).to_dict('records')
    pl = playlists[cluster_id].to_dict('records')
    return render_template_string(
        CLUSTER_HTML,
        cluster_id=cluster_id,
        summary=summary,
        top_artists=ta,
        top_songs=ts,
        playlist=pl
    )

@app.route('/download/<int:cluster_id>')
def download(cluster_id):
    """Downloads the playlist for a specific cluster as a CSV file."""
    df = playlists[cluster_id]
    path = f'temp_{cluster_id}.csv'
    df.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)