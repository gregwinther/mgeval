import numpy as np
from mgeval import core
import glob, os
from matplotlib import pyplot as plt
import matplotlib.animation as animation

files = sorted(glob.glob("../guitar/*.mid"), key=os.path.getmtime)
num_samples = len(files)

print(f"{num_samples} samples")

# Extracting pitch class transitions for each midi file
pct_matrices = []
for f in files:
    features = core.extract_feature(f)
    # This is the same as using the very cryptic "getattr()"
    pctm = core.metrics().pitch_class_transition_matrix(features)
    pct_matrices.append(pctm)

# Animation
fig = plt.figure()
im = plt.imshow(pct_matrices[0])
text = plt.text(1, -1, "...")

def animate(i):
    i = i % num_samples
    im.set_array(pct_matrices[i])
    text.set_text(files[i])
    return [im, text]

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=1000,
    interval=500,
)

plt.show()