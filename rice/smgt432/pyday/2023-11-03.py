# -*- coding: utf-8 -*-
"""pyday_2023-11-03.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZBR4lAxThmmndwHBH0ydYXpFzi40hWOi

Today we're taclking the second half of the Tracking Data section from Devin Pleuler's [Soccer Analytics Handbook](https://github.com/devinpleuler/analytics-handbook/blob/master/soccer_analytics_handbook.ipynb).
"""

!pip install kloppy
!pip install mplsoccer

import numpy as np
from kloppy import metrica
import mplsoccer as mpl

dataset = metrica.load_open_data(
    match_id=1,
    coordinates="metrica"
) # This takes about 60 seconds

df = dataset.to_pandas()

df.head(5)

frame_rate = 25
length, width = 105, 68
adjust = np.array([length, width])

metrica_attrs = {
    "pitch_type": "metricasports",
    "pitch_length": length,
    "pitch_width": width,
    "line_color": "black",
    "linewidth": 1,
    "goal_type": "circle"
}

hj = list(set([x.split("_")[1] for x in df.columns if "home" in x]))

start, stop = 2143, 2309 # Frame Range

pitch = mpl.Pitch(**metrica_attrs)
fig, ax = pitch.draw(figsize=(9,6))

for j in hj:
    path = df[['home_{}_x'.format(j), 'home_{}_y'.format(j)]].values[start:stop]
    pitch.plot(*path.T, lw=2, alpha=0.8, ax=ax)

path = df[['ball_x', 'ball_y']].values[start:stop]
pitch.plot(*path.T, lw=1, alpha=1, color='black', ax=ax)

from matplotlib.colors import ListedColormap, to_hex
def bulid_cmap(x, y):
    r,g,b = x
    r_, g_, b_ = y
    N = 256
    A = np.ones((N, 4))
    A[:, 0] = np.linspace(r, 1, N)
    A[:, 1] = np.linspace(g, 1, N)
    A[:, 2] = np.linspace(b, 1, N)
    cmp = ListedColormap(A)

    B = np.ones((N, 4))
    B[:, 0] = np.linspace(r_, 1, N)
    B[:, 1] = np.linspace(g_, 1, N)
    B[:, 2] = np.linspace(b_, 1, N)
    cmp_ = ListedColormap(B)

    newcolors = np.vstack((cmp(np.linspace(0, 1, 128)),
                            cmp_(np.linspace(1, 0, 128))))
    return ListedColormap(newcolors)

blue, red = (44,123,182), (215,25,28)
blue = [x/256 for x in blue]
red = [x/256 for x in red]
diverging = bulid_cmap(blue, red)
diverging_r = bulid_cmap(red, blue)


path = (
    df[['home_10_x', 'home_10_y', 'period_id']]
    .assign(
        home_10_x=lambda x: np.where(x['period_id'] == 2, -x['home_10_x'], x['home_10_x']),
        home_10_y=lambda x: np.where(x['period_id'] == 2, -x['home_10_y'], x['home_10_y']),
    )
    [['home_10_x', 'home_10_y']]
    .values * adjust
)

runs = []
running = False
speed_threshold = 6
sustained_frame_threshold = 10
for i, coord in enumerate(path):
    displacement = path[i-1]
    speed = np.linalg.norm(coord - displacement) * frame_rate
    if speed > speed_threshold:
        if not running:
            running = True
            frame_start = i
    else:
        if running:
            running = False
            frame_end = i
            if frame_end > frame_start + sustained_frame_threshold:
                runs.append((frame_start, frame_end))

pitch = mpl.Pitch(**metrica_attrs)
fig, ax = pitch.draw(figsize=(9,6))

for (start, stop) in runs:
    unadjust = np.array(path[start:stop]) * (1 / adjust)
    pitch.plot(*unadjust.T, lw=1, c=red, ax=ax, zorder=1)
    pitch.scatter(*unadjust[0].T, s=10, color=red, ax=ax)
    pitch.scatter(*unadjust[-1].T, s=15,
                  facecolor="white", edgecolor=red, zorder=2, ax=ax)

# EXERCISE #1: Flip the coordinates for the second half

n = 87470
frame = df.iloc[n:n+1]
frame_ = df.iloc[n+1:n+2]

bp = frame[['ball_x', 'ball_y']].values[0]

hj = np.unique([x.split("_")[1] for x in df.columns if "home" in x])
aj = np.unique([x.split("_")[1] for x in df.columns if "away" in x])

def team_vectors(f, f_, team, jerseys):
    p, v = [], []
    for j in jerseys:
        pp = f[['{}_{}_x'.format(team, j), '{}_{}_y'.format(team, j)]].values[0]
        pp_ = f_[['{}_{}_x'.format(team, j), '{}_{}_y'.format(team, j)]].values[0]
        if ~np.isnan(pp[0]):
            p.append(pp)
            v.append(pp_ - pp)
    return np.array(p), np.array(v) * frame_rate

hp, hv = team_vectors(frame, frame_, "home", hj)
ap, av = team_vectors(frame, frame_, "away", aj)

pitch = mpl.Pitch(**metrica_attrs)
fig, ax = pitch.draw(figsize=(12, 8))

pitch.arrows(*hp.T, *hp.T + hv.T, ax=ax, color='k', zorder=1,
             headaxislength=3, headlength=3, headwidth=4, width=1)

pitch.arrows(*ap.T, *ap.T + av.T, ax=ax, color='k', zorder=1,
             headaxislength=3, headlength=3, headwidth=4, width=1)

pitch.scatter(*hp.T, ax=ax, facecolor=red, s=100, edgecolor='k')
pitch.scatter(*ap.T, ax=ax, facecolor=blue, s=100, edgecolor='k')
pitch.scatter(*bp.T, ax=ax, facecolor='yellow', s=40, edgecolor='k')

# EXERCISE #2: Plot the frame for the second goal of the game (Game 1)

import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(0, length, length*2),
                     np.linspace(0, width, width*2))

indexes = np.stack([xx, yy], 2)

def tti_shaw(origin, destination, velocity,
               reaction_time=0.7, max_velocity=5.0):

    r_reaction = origin + velocity * reaction_time
    d = destination - r_reaction
    return reaction_time + np.linalg.norm(d, axis=-1) / max_velocity

def tti(origin, destination, velocity, reaction_time=0.7, max_velocity=5.0):
    u = (origin + velocity) - origin
    v = destination - origin
    u_mag = np.sqrt(np.sum(u**2, axis=-1))
    v_mag = np.sqrt(np.sum(v**2, axis=-1))
    dot_product = np.sum(u * v, axis=-1)
    angle = np.arccos(dot_product / (u_mag * v_mag))
    r_reaction = origin + velocity * reaction_time
    d = destination - r_reaction
    t = (u_mag * angle/np.pi +
         reaction_time +
         np.linalg.norm(d, axis=-1) / max_velocity)

    return t

def tti_surface(players, velocities, indexes, tti=tti):
    pvalues = np.empty((players.shape[0], indexes.shape[0], indexes.shape[1]))
    for k in range(players.shape[0]):
        pvalues[k, :, :] = tti(players[k], indexes, velocities[k])
    values = np.amin(pvalues, axis=0)

    return values

Z_home = tti_surface(hp * adjust, hv * adjust, indexes)
Z_away = tti_surface(ap * adjust, av * adjust, indexes)

Z = Z_home - Z_away

pitch = mpl.Pitch(**metrica_attrs)
fig, ax = pitch.draw(figsize=(12, 8))
min, max = -10, 10
levels = np.linspace(min, max, 11)
s = ax.contourf(Z, extent=(0,1,0,1), levels=levels,
            cmap=diverging_r, vmin=min, vmax=max, alpha=0.8,
            antialiased=True, extend="both")

pitch.arrows(*hp.T, *hp.T + hv.T, ax=ax, color='k', zorder=1,
             headaxislength=3, headlength=3, headwidth=4, width=1)

pitch.arrows(*ap.T, *ap.T + av.T, ax=ax, color='k', zorder=1,
             headaxislength=3, headlength=3, headwidth=4, width=1)

pitch.scatter(*hp.T, ax=ax, facecolor=red, s=100, edgecolor='k')
pitch.scatter(*ap.T, ax=ax, facecolor=blue, s=100, edgecolor='k')
pitch.scatter(*bp.T, ax=ax, facecolor='yellow', s=40, edgecolor='k')

cbar = plt.colorbar(s, shrink=0.6, ticks=levels)
l = cbar.ax.set_yticklabels(["{:.1f} sec.".format(t) for t in levels])

# EXERCISE #3: Find a way to toggle the uncertainty in the pitch control map