import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import imageio
import os

# Load dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
df['target'] = data.target

# Use only 7 features (first 7 for simplicity)
features = df.columns[:7]

# Relevance
relevance = pd.Series(mutual_info_classif(df[features], df['target'], random_state=42), index=features)

# Redundancy as absolute correlation
corr = df[features].corr().abs()

# Create directory for frames
frames_dir = "mrmr_frames"
os.makedirs(frames_dir, exist_ok=True)
frame_files = []

# Greedy mRMR selection (select all 7 features)
selected, remaining = [], list(features)
for step in range(len(features)):
    scores = {}
    for f in remaining:
        red = 0 if not selected else corr.loc[f, selected].mean()
        scores[f] = relevance[f] - red
    best = max(scores, key=scores.get)
    selected.append(best)
    remaining.remove(best)

    # Plot step
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True)

    # Score bar chart
    score_series = pd.Series(scores).reindex(features)
    axs[0].bar(features, score_series.values, color='skyblue')
    axs[0].set_title(f"mRMR scores — Step {step + 1}: selecting '{best}'")
    axs[0].tick_params(axis='x', rotation=45)

    # Selected features
    selected_mask = [1 if f in selected else 0 for f in features]
    axs[1].bar(features, selected_mask, color='orange')
    axs[1].set_ylim(0, 1.2)
    axs[1].set_yticks([])
    axs[1].set_title("Selected features (1 = selected)")
    axs[1].tick_params(axis='x', rotation=45)

    # Save frame
    frame_path = os.path.join(frames_dir, f"frame_{step:02d}.png")
    plt.savefig(frame_path, dpi=100)
    plt.close(fig)
    frame_files.append(frame_path)

# Write GIF
gif_path = "mrmr_demo_7features.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.8) as writer:
    for fname in frame_files:
        image = imageio.imread(fname)
        writer.append_data(image)

# Cleanup frames
for fname in frame_files:
    os.remove(fname)

print("GIF saved to:", gif_path)
