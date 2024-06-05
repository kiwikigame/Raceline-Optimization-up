import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import yaml
import scipy
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import pandas as pd
import sys


# Define constants and load the map
MAP_NAME = "Melbourne_map"
TRACK_WIDTH_MARGIN = 0.0  # Extra Safety margin, in meters

if os.path.exists(f"maps/{MAP_NAME}.png"):
    map_img_path = f"maps/{MAP_NAME}.png"
elif os.path.exists(f"maps/{MAP_NAME}.pgm"):
    map_img_path = f"maps/{MAP_NAME}.pgm"
else:
    raise Exception("Map not found!")

map_yaml_path = f"maps/{MAP_NAME}.yaml"
raw_map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
raw_map_img = raw_map_img.astype(np.float64)

# Convert grayscale to binary
map_img = raw_map_img.copy()
map_img[map_img <= 210.] = 0
map_img[map_img > 210.] = 1

# Calculate Euclidean Distance Transform
dist_transform = scipy.ndimage.distance_transform_edt(map_img)

# Threshold the distance transform to create a binary image
THRESHOLD = 0.17  # Adjust as necessary
centers = dist_transform > THRESHOLD * dist_transform.max()

# Apply skeletonization
centerline = skeletonize(centers)

# Calculate the centerline distance
centerline_dist = np.where(centerline, dist_transform, 0)

map_height = map_img.shape[0]
LEFT_START_Y = map_height // 2
NON_EDGE = 0.0
left_start_y = LEFT_START_Y
left_start_x = 0

# Trouver le point de départ à gauche
while (centerline_dist[left_start_y][left_start_x] == NON_EDGE):
    left_start_x += 1

# Définir les limites de récursion
sys.setrecursionlimit(20000)

# Initialisation des structures de données
visited = set()
centerline_points = []
track_widths = []
DIRECTIONS = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)]

starting_point = (left_start_x, left_start_y)

# Vérification des limites
def is_within_bounds(point, max_x, max_y):
    return 0 <= point[0] < max_x and 0 <= point[1] < max_y

# Fonction DFS itérative
def dfs_iterative(starting_point, max_x, max_y):
    stack = [starting_point]
    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        centerline_points.append(np.array(point))
        track_widths.append(np.array([centerline_dist[point[1]][point[0]], centerline_dist[point[1]][point[0]]]))
        
        for direction in DIRECTIONS:
            next_point = (point[0] + direction[0], point[1] + direction[1])
            if (is_within_bounds(next_point, max_x, max_y) and
                    centerline_dist[next_point[1]][next_point[0]] != NON_EDGE and
                    next_point not in visited):
                stack.append(next_point)

# Définir les dimensions maximales
max_x = len(centerline_dist[0])
max_y = len(centerline_dist)

# Exécuter la fonction DFS
dfs_iterative(starting_point, max_x, max_y)

# Conversion en Pandas, transformation des pixels en mètres, et décalage par l'origine
track_widths_np = np.array(track_widths)
waypoints = np.array(centerline_points)

# Fusionner les largeurs de piste avec les waypoints
data = np.concatenate((waypoints, track_widths_np), axis=1)

# Conversion en DataFrame pour l'analyse ultérieure si nécessaire
df = pd.DataFrame(data, columns=['x', 'y', 'width1', 'width2'])

# Load map metadata
with open(map_yaml_path, 'r') as yaml_stream:
    try:
        map_metadata = yaml.safe_load(yaml_stream)
        map_resolution = map_metadata['resolution']
        origin = map_metadata['origin']
    except yaml.YAMLError as ex:
        print(ex)

# Calculate map parameters
orig_x = origin[0]
orig_y = origin[1]
orig_s = np.sin(origin[2])
orig_c = np.cos(origin[2])

# Transform the data
transformed_data = data
transformed_data *= map_resolution
transformed_data += np.array([orig_x, orig_y, 0, 0])

# Apply safety margin
transformed_data -= np.array([0, 0, TRACK_WIDTH_MARGIN, TRACK_WIDTH_MARGIN])

# Save to CSV
with open(f"inputs/tracks/{MAP_NAME}.csv", 'wb') as fh:
    np.savetxt(fh, transformed_data, fmt='%0.4f', delimiter=',', header='x_m,y_m,w_tr_right_m,w_tr_left_m')

# Load and transform the track data for plotting
raw_data = pd.read_csv(f"inputs/tracks/{MAP_NAME}.csv")
x = raw_data["# x_m"].values
y = raw_data["y_m"].values
wr = raw_data["w_tr_right_m"].values
wl = raw_data["w_tr_left_m"].values

x -= orig_x
y -= orig_y

x /= map_resolution
y /= map_resolution

# Plot all steps in a single figure
fig, axs = plt.subplots(2, 4, figsize=(24, 12))

# Original map
axs[0, 0].imshow(raw_map_img, cmap='gray', origin='lower')
axs[0, 0].set_title('Original Map')

# Binary map
axs[0, 1].imshow(map_img, cmap='gray', origin='lower')
axs[0, 1].set_title('Binary Map')

# Distance Transform
axs[0, 2].imshow(dist_transform, cmap='gray', origin='lower')
axs[0, 2].set_title('Distance Transform')

# Thresholded centers
axs[0, 3].imshow(centers, cmap='gray', origin='lower')
axs[0, 3].set_title('Thresholded Centers')

# Skeletonized map
axs[1, 0].imshow(centerline, cmap='gray', origin='lower')
axs[1, 0].set_title('Skeletonized Map')

# Centerline with distances
axs[1, 1].imshow(centerline_dist, cmap='gray', origin='lower')
axs[1, 1].set_title('Centerline with Distances')

# Empty subplot
axs[1, 2].axis('off')

# Final map with the track
axs[1, 3].imshow(map_img, cmap="gray", origin="upper")
axs[1, 3].plot(x, y, color='red')
axs[1, 3].set_title('Final Map with Track')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
