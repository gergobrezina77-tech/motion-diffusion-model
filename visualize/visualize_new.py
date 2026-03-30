import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

# SMPL joint connections (parent -> child pairs)
JOINT_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),       # pelvis to hips and spine
    (1, 4), (2, 5), (3, 6),       # hips to knees, spine to chest
    (4, 7), (5, 8), (6, 9),       # knees to ankles, chest to upper spine
    (7, 10), (8, 11),              # ankles to feet
    (9, 12), (9, 13), (9, 14),    # upper spine to neck and shoulders
    (12, 15),                      # neck to head
    (13, 16), (14, 17),            # shoulders to elbows
    (16, 18), (17, 19),            # elbows to wrists
    (18, 20), (19, 21),            # wrists to hands
    (20, 22), (21, 23),            # hands to fingertips
]

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path', type=str, required=True, help='Path to results.npy')
parser.add_argument('--output_path', type=str, required=True, help='Path to output mp4')
args = parser.parse_args()

data = np.load(args.npy_path, allow_pickle=True).item()
motions = data['motion']

# pick the first generated example
motion = motions[0]   # shape: [num_joints, 3, num_frames]

# center around root joint
root = motion[0:1, :, :]
motion_centered = motion - root

max_range = np.ptp(motion_centered).max() / 2
mid = 0

fig = plt.figure(figsize=(6, 6), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

def draw_frame(t):
    ax.clear()
    ax.set_facecolor('black')
    frame = motion_centered[:, :, t]

    # draw joint connections
    for i, j in JOINT_CONNECTIONS:
        if i < len(frame) and j < len(frame):
            ax.plot(
                [frame[i, 0], frame[j, 0]],
                [frame[i, 1], frame[j, 1]],
                [frame[i, 2], frame[j, 2]],
                color='white', linewidth=2
            )

    # draw joints
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2],
               color='cyan', s=20, zorder=5)

    # hide all axes, grid, panes
    ax.set_xlim(mid - max_range, mid + max_range)
    ax.set_ylim(mid - max_range, mid + max_range)
    ax.set_zlim(mid - max_range, mid + max_range)
    ax.view_init(elev=90, azim=-60)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    plt.tight_layout(pad=0)

writer = imageio.get_writer(args.output_path, fps=20, macro_block_size=1)

num_frames = motion_centered.shape[-1]
for t in range(num_frames):
    draw_frame(t)
    fig.canvas.draw()
    if hasattr(fig.canvas, 'tostring_rgb'):
        buf = fig.canvas.tostring_rgb()
        h, w = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype='uint8').reshape(w, h, 3)
    else:
        buf = fig.canvas.tostring_argb()
        h, w = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype='uint8').reshape(w, h, 4)
        image = arr[:, :, [1, 2, 3]]
    writer.append_data(image)

writer.close()
print(f'wrote {args.output_path}')