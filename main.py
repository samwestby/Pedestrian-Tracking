import os
import numpy as np
import json
import tracker_api
# from tracker_api import csv_to_list, convert_coordinates,sort_data, switch_frame_order_and_id_order,\
# 	centered_rolling_average, create_path_line_plot, fetch_all_hd, plot_hd

def save(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def load(path):
	with open(path, 'r') as f:
		json.load(f)

if __name__ == '__main__':
	# Get rected.csv from running a video with human_detection.py
	# rected.csv is a csv with each row corresponding to each frame of your video
	# and each cell is coordinates (x1, y1, x2, y2) or None
	csv_path = os.path.join('path', 'to', 'csv.csv')
	coord_list = tracker_api.csv_to_list(csv_path)

	# Source points should be the points in pixles of your measured area.
	# Destination points should be the points you want to convert to
	# You can use warp_perspective.py to test these out
	height = None
	width = None
	pts_src = np.float32([[top_left], [top_right], [bottom_left], [bottom_right]])
	pts_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
	# bounding =
	coord_list_transformed = tracker_api.convert_coordinates(coord_list, pts_src, pts_dst, bounding=None)
	# Currently index of data is frame number
	sorted_data = tracker_api.sort_data(coord_list_transformed, threshold=0.5)
	# Now index=id
	smoothed_data = tracker_api.centered_rolling_average(sorted_data, window=9, min_length=20)

	scale = None
	shift = None

	still_img_path = os.path.join('path', 'to', 'still_img.jpg')
	# Be sure index=id for this. If index=frame, use switch_frame_order_and_id_order()
	tracker_api.create_path_line_plot(still_img_path, smoothed_data, pts_src, pts_dst,
	                                  window=(15000, 14000), shift=shift, scale=scale)

	# This also takes data where index=id
	hd_data = tracker_api.fetch_all_hd(smoothed_data, min_people_per_frame=5)
	tracker_api.plot_hd(hd_data, radius=10, max_angle=45)