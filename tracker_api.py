import copy
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def warp_perspective(path, pts_src, pts_dst, shift=None, scale=None, window=(10000, 10000)):
	'''
	Warps view to be in flat coordinates

	:param path: Path to still image from video
	:param pts_src:
	:param pts_dst:
	:param shift:
	:param scale:
	:param window:
	:return:
	'''
	img = cv2.imread(path)
	if scale is not None:
		pts_dst = scale * pts_dst
	if shift is not None:
		pts_dst = shift + pts_dst
	M = cv2.getPerspectiveTransform(pts_src, pts_dst)
	dst = cv2.warpPerspective(img, M, window)
	plt.imshow(dst), plt.title('Transformed')
	plt.show()

def convert_box_string(cell):
	'''
	Helper function for :func:`~Dev_ped.tracker.api.csv_to_list'

	:param cell:
	:return:
	'''
	if pd.isnull(cell):
		return None
	# 0: x1, 1: y1, 2: x2, 3: y2, 4: Ped_id
	cords = np.empty(5)
	cords[4] = np.nan
	cords[:4] = cell.replace(' ', '').replace('(','').replace(')','').split(',')
	cords = [int(x) for x in cords[:4]]
	return cords

def csv_to_list(path):
	'''
	This function expects a csv where every row represents a video frame
	and each cell value is a bounding box coordinate.

	It expects each cell to be a string formatted as "(x1, y1, x2, y2)"

	:param path:
	:return:
	'''
	with open(path, mode='r', newline='') as f:
		reader = csv.reader(f, delimiter=',')
		data = list(reader)
		ped_coords = [None] * len(data)
		row = 0
		for line in data:
			line = [convert_box_string(x) for x in line if x != '']
			ped_coords[row] = line
			row += 1

		return ped_coords

def transform_cell(box, M, bounding_area=None):
	'''
	Helper function for :func:`~Dev_ped.tracker.api.convert_coordinates'

	:param box: len(box) >=4   top left: 0,1   top right: 2,1   bot left: 0, 3   bot right: 2,3
	:param M:
	:param bounding_area:
	:return:
	'''
	if box != box or box == '':
		return np.nan

	bottom_center = [int(np.round(np.mean([box[0], box[2]]))), box[3]]
	bottom_center.append(1)
	t_matrix = np.multiply(bottom_center, M)
	# tx = (h0 * x + h1 * y + h2)
	# ty = (h3 * x + h4 * y + h5)
	# tz = (h6 * x + h7 * y + h8)
	# x_transformed = tx/tz
	# y_transformed = ty/tz
	x = (sum(t_matrix[0]) / sum(t_matrix[2]))
	y = (sum(t_matrix[1]) / sum(t_matrix[2]))

	if bounding_area is not None:
		# if not(bounding_area[0] < x < bounding_area[2]
		# 		or bounding_area[1] < y < bounding_area[3]):
		if not(bounding_area[0] > x or bounding_area[2] < x
		        or bounding_area[1] > y or bounding_area[3] < y):
			return [x, y]
		else:
			return np.nan
	return [x, y]


def convert_coordinates(data, pts_src, pts_dst, bounding=None):
	'''
	Convert coordinates according to points determined in :func:'~Dev_ped.tracker.api.warp_perspective'

	:param data:
	:param pts_src:
	:param pts_dst:
	:param shift:
	:param scale:
	:param bounding: Area in video to focus on .(x1, y1, x2, y2).
		Use coordinates from the converted coordinate plane
	:return:
	'''
	# getPerspectiveTransform is good for when you know 4 points with certainty
	# M is a 3 x 3 array
	M = cv2.getPerspectiveTransform(pts_src, pts_dst)
	data_transformed = [[transform_cell(box, M, bounding) for box in frame] for frame in data]
	data_transformed = [[box for box in frame if box == box] for frame in data_transformed]

	return data_transformed


# TODO: Create a way to not use a global
ped_count = 0

# TODO: Find a more elegant implementation of this
def switch_frame_order_and_id_order(data, to_frame_index=False):
	'''
	Switch value of index and cell[1]. This is either index=frame and cell[1]=id
	or index=id and cell[1]=frame

	This function was written with variable names assuming we're going from
         index=frame to index=id
	but the function works for index=id too.

	:param data:
	:param to_frame_index: Set to True if going from index=id to index=frame
	:return:
	'''
	max_id = None

	if to_frame_index:
		# Maximum frame in data
		max_id = max([row[-1][1] for row in data if row is not None])
	else:
		local_max_id = 0
		for last_idx in range(-1, -len(data), -1):
			if not data[last_idx]:
				continue
			id_list = [cell[1] for cell in data[last_idx]]
			if max(id_list) > local_max_id:
				local_max_id = max(id_list)

			elif local_max_id not in id_list:
				max_id = local_max_id
				break

	if max_id is None:
		raise ValueError('No max_id found')
	id_ordered = [[np.nan] for _ in range(max_id + 1)]
	for frame in range(len(data)):
		if data[frame] is None:
			continue
		for person in range(len(data[frame])):
			# data[frame][person][1] is the id for that cell
			# We change it to frame number for id_ordered
			new_val = copy.deepcopy(data[frame][person])
			new_val[1] = frame
			id_ordered[ data[frame][person][1] ].append( new_val )

	for row in range(len(id_ordered)):
		del id_ordered[row][0]

	return id_ordered


def give_person_id(cell, new_person=False):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.sort_data'

	:param cell:
	:param new_person:
	:return:
	'''
	if cell != cell:
		return np.nan
	global ped_count
	#      [ bottom center coordinate, person id, vector to next frame coordinates ]
	cell = [cell, ped_count, np.nan]
	if new_person:
		ped_count += 1
	return cell


def resolve_duplicates(match_idxs, distance_matrix):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.sort_data'

	:param match_idxs:
	:param distance_matrix:
	:return:
	'''
	def get_instance_indices(idx_list, num):
		return list(filter(lambda x: idx_list[x]==num, range(len(idx_list))))
	for idx in match_idxs:
		if idx == -1:
			continue
		matches = get_instance_indices(match_idxs, idx)
		if len(matches) > 1:
			d = {match: distance_matrix[match][match_idxs[match]] for match in matches}
			actual_match = min(d, key=d.get)
			for match in matches:
				if match != actual_match:
					match_idxs[match] = -1
	match_idxs = [x if x != -1 else np.nan for x in match_idxs]
	return match_idxs

def sort_data(data, threshold):
	'''
	Go through each frame of data and match cells to cells in next row if
	they're within a certain distance threshold

	:param data:
	:return:
	'''
	frames_with_id = [np.nan] * len(data)
	frames_with_id[0] = [give_person_id(person, new_person=True) for person in data[0]]
	prev_frame = frames_with_id[0]
	# Give the same ID to between frame rectangles that are under a certain threshold
	frame = 0
	for cur_frame in data[1:]:
		frame += 1
		if not len(cur_frame):
			frames_with_id[frame] = []
			prev_frame = cur_frame
			continue
		if not len(prev_frame):
			cur_frame = [give_person_id(cell, new_person=True) for cell in cur_frame]
			frames_with_id[frame] = cur_frame
			prev_frame = cur_frame
			continue

		# In this section we create a distance matrix and find the minimums.
		#
		# After this line, cur_tile looks like:
		# [[1a, 1a, 1a, ...],
		#  [1b, 1b, 1b, ...],
		#  ...
		# ]
		# The shape of cur_tile: rows = len(cur_row), cols = len(prev_row)

		cur_tile = [[cell]*len(prev_frame) for cell in data[frame]]
		prev_tile = [data[frame-1]] * len(cur_frame)

		# After the transpose line, prev_tile looks like:
		# [[2a, 2b, 2c, ...],
		#  [2a, 2b, 2c, ...],
		#  ...
		# ]
		# The shape of prev_tile: rows = len(cur_row), cols = len(prev_row)
		#
		# Get the norms of all element-wise differences
		# [[norm(1a, 2a), norm(1a, 2b), norm(1a, 2c),
		#  [norm(1b, 2a), norm(1b, 2b), norm(1b, 2c),
		#  ...
		# ]

		distance_matrix = [[np.linalg.norm(dif) for dif in row] for row in np.subtract(cur_tile, prev_tile)]
		# For match_idxs, the key is the index of the current frame value that matches with previous frame value.
		# match_idxs[key] is the index of the previous frame match cell.
		match_idxs = np.argmin(distance_matrix, axis=1)
		match_idxs = resolve_duplicates(match_idxs, distance_matrix)
		frames_with_id[frame] = [np.nan] * len(cur_frame)
		for i in range(len(cur_frame)):
			if match_idxs[i] != match_idxs[i] or distance_matrix[i][match_idxs[i]] > threshold:
				frames_with_id[frame][i] = give_person_id(cur_frame[i], new_person=True)
			else:
				#                          [bottom center     id                vec to next frame]
				frames_with_id[frame][i] = [cur_frame[i], prev_frame[match_idxs[i]][1], np.nan]
				# Set vector to next frame
				prev_frame[match_idxs[i]][2] = list(np.subtract(cur_frame[i], prev_frame[match_idxs[i]][0]))

		prev_frame = frames_with_id[frame]
	return frames_with_id


def centered_rolling_average(data, window, min_length=None):
	'''
	This function smooths each pedestrian's path. This is essential for
	intelligible speed distributions and heading diretion calculations

	:param data: index=ID
	:param window:
	:return:
	'''
	if min_length is not None:
		if 2 * window + 1 > min_length:
			raise ValueError('2 * window + 1 = {} cannot be greater than min_length = {}'.format(window, min_length))

	# This was index=frame and now we're making it index=id
	data = switch_frame_order_and_id_order(data)
	smoothed_data = copy.deepcopy(data)
	if min_length is None:
		min_length = 2 * window + 2
	# Adding this window to front and back end of data point
	# Make a centered rolling average. Current coordinate is the average of the values in the window
	for id in range(len(data)):
		if len(data[id]) < min_length:
			smoothed_data[id] = None
			continue

		windowed_data_x = [None] * (2 * window + 1)
		windowed_data_y = [None] * (2 * window + 1)
		for frame in range(window, len(data[id]) - window):
			# get the x and y means of the window
			for c in range(0, 2 * window + 1):  # frame-window, frame+window):
				# [0][0] is the x coordinate, [0][1] is the y coordinate
				windowed_data_x[c] = data[id][c + frame - window][0][0]
				windowed_data_y[c] = data[id][c + frame - window][0][1]

			x_mean = np.convolve(windowed_data_x, np.ones(len(windowed_data_x), ) / len(windowed_data_x), 'valid')
			y_mean = np.convolve(windowed_data_y, np.ones(len(windowed_data_y), ) / len(windowed_data_y), 'valid')
			smoothed_data[id][frame][0] = [(x_mean[0]), (y_mean[0])]
			smoothed_data[id][frame][2] = None
			if frame == window:
				continue
			# set vec_to_next for previous values
			vec_from_prev_to_next = np.subtract(smoothed_data[id][frame][0],
			                                    smoothed_data[id][frame - 1][0])
			smoothed_data[id][frame - 1][2] = list(vec_from_prev_to_next)

		# beginning and end are still nan
		smoothed_data[id] = smoothed_data[id][window : len(smoothed_data[id]) - window]

	return smoothed_data



def get_color(cords):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.create_path_line_plot'

	:param cords:
	:return:
	'''
	# TODO make this divide colors in an X rather than a +
	# TODO change this back to cords[0] < 0
	if cords[0] < 0:
		# return tuple([255, 165, 0, 100])
		return tuple([255, 165, 0])
		# if cords[1] < 0:
		#     # -,- orange
		#     return tuple([255,165,0, 100])
		# else:
		#     # -,+ red
		#     return tuple([255,0,0, 100])
	else:
		# return tuple([0, 0, 255, 100])
		return tuple([0, 0, 255])
		# if cords[1] < 0:
		#     # +,- blue
		#     return tuple([0,0,255, 100])
		# else:
		#     # +,+ green
		#     return tuple([0,255,0, 100])


def draw_lines(img, data, shift, scale):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.create_path_line_plot'

	:param img:
	:param data:
	:param shift:
	:param scale:
	:return:
	'''

	# TODO: redo this to be more efficient
	for i in range(len(data)):
		# vec_to_next is null for final value, so we only go to len(data[i] - 1)
		if data[i] is None:
			continue
		for j in range(len(data[i]) - 1):
			if data[i][j][2] != data[i][j][2]:
				continue
			color = get_color(data[i][j][2])
			cur_center = scale_coordinate(data[i][j][0], shift=shift, scale=scale)
			next_center = scale_coordinate(data[i][j + 1][0], shift=shift, scale=scale)

			cv2.line(img, tuple(cur_center), tuple(next_center), color, 10)
			# list(np.add(data[i][j][0], data[i][j][2]))
			# line_coords = data[i][j][0] + next_center
			# line_coords = np.rint(line_coords).astype(int)
			# draw.line(scale_coordinate(line_coords), fill=color, width=10)

def scale_coordinate(coord, shift, scale):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.create_path_line_plot'

	:param coord:
	:param shift:
	:param scale:
	:return:
	'''

	if scale is not None:
		coord = np.multiply(scale, coord)
	if shift is not None:
		coord = shift + coord

	return [int(x) for x in coord]

def create_path_line_plot(still_im_path, data, pts_src, pts_dst, window=(10000, 10000), shift=None, scale=None):
	'''
	This function creates a plot which acts like a timelapse of all paths in data

	:param still_im_path: Path to still img from captured video
	:param data: index=id
	:param pts_src:
	:param pts_dst:
	:param window:
	:param shift: Found by guessing and checking with :func:'~Dev_ped.tracker.api.warp_perspective
	:param scale: Found by guessing and checking with :func:'~Dev_ped.tracker.api.warp_perspective
	:return:
	'''

	img = cv2.imread(still_im_path)
	if scale is not None:
		pts_dst = np.multiply(pts_dst, scale)
	if shift is not None:
		pts_dst = np.add(pts_dst, shift)
	M = cv2.getPerspectiveTransform(pts_src, pts_dst)
	im1 = cv2.warpPerspective(img, M, window)
	draw_lines(im1, data, scale=scale, shift=shift)
	plt.imshow(im1)
	plt.show()


def angle_between_vecs(v1, v2=(0, 1)):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.fetch_all_hd'

	:param v1:
	:param v2:
	:return: Absolute angle between two vectors
	'''
	# normalize vectors
	v1_norm = np.linalg.norm(v1)
	v2_norm = np.linalg.norm(v2)
	if v1_norm == 0 or v2_norm == 0:
		return None
	u1 = v1 / v1_norm
	u2 = v2 / v2_norm
	return np.rad2deg(np.arccos(np.clip(np.dot(u1, u2), -1, 1)))


def get_single_hd(fp, N):
	'''
	Helper function for :func:'~Dev_ped.tracker.api.fetch_all_hd'
	Get heading differences for one focal person and frame

	:param fp: Focal person
	:param N: Set of all neighbors per frame
	:return: List of mean heading difference (HD) for the focal person
			List of counts
	'''

	if fp[VEC_TO_NEXT_IDX] is None:
		return None
	N = [n for n in N if n[VEC_TO_NEXT_IDX] is not None]
	# Normalize vec lengths
	N = [ [ np.subtract(n[CENTER_IDX], fp[CENTER_IDX]), n[VEC_TO_NEXT_IDX] ] for n in N]
	# normalized location       how many degrees different is heading direction
	N = [[list(n[0]), angle_between_vecs(n[1], fp[VEC_TO_NEXT_IDX])] for n in N]
	return N


def fetch_all_hd(data, min_people_per_frame=5):
	'''
	Get the mean heading direction of the entire data set

	:param data: List of people and heading direction sorted by frame
	:param min_people_per_frame:
	:return: List of mean heading difference (HD) for every recorded person
	'''

	# Now it's index=frame
	data = switch_frame_order_and_id_order(data)
	data = [x for x in data if len(x) >= min_people_per_frame]

	# Create zeroed list of sectors for HD sorted by neighbors
	# Make ranges 0-2m, 2-4m, 4-6m, 6-8m, and, 8-10m
	# Use linear to polar conversion to represent a circle
	dif_count = 0
	for frame in data:
		dif_count += len(frame)
	heading_difs =  [None] * dif_count
	index = 0
	for frame in data:
		# Loop through every person as fp
		for fp in frame:
			# TODO: only loop through people that are surrounded by other people
			heading_difs[index] = get_single_hd(fp, frame)
			index += 1
	# flatten heading_difs
	heading_difs = [n for line in heading_difs if line is not None for n in line]
	return heading_difs

def plot_hd(hd, radius=20, max_angle=45):
	'''

	:param hd:
	:return: plot
	'''

	# Take hd point cloud, make a nice grid plot
	# hd = [n if n[1] < 99 else None for n in hd  ]
	# box_size is the length and width dims in meters of the heatmap boxes
	box_size = 1
	# radius is the radius size that we draw around the fp
	def get_grid_cords(neighbor):
		'''

		:param neighbor:
		:return:
		'''
		# shift the negative indices up so they're at zero
		return [int((x + radius) / box_size) for x in neighbor[0]]

	# List of mean_heading differences for each box to be plotted
	mean_heading_diff = np.zeros([int(np.ceil(2*radius / box_size)), int(np.ceil(2*radius / box_size))])
	count = np.zeros([int(np.ceil(2*radius / box_size)), int(np.ceil(2*radius / box_size))])
	# This loop sums up all of the heading diffs then takes the average of each cell in mhd
	for n in hd:
		if n is None:
			continue
		grid_cell = get_grid_cords(n)

		# Skip people that were farther than radius from focal person
		# Also skip if heading dif is greater than 75
		if max(grid_cell) >= (2 * radius / box_size) or min(grid_cell) < 0:
			continue

		mean_heading_diff[grid_cell[0], grid_cell[1]] = n[1]
		count[grid_cell[0], grid_cell[1]] += 1
	# The next two lines allow us to use np.divide for the element-wise average
	mean_heading_diff[count == 0] = None
	count[count == 0] = 1
	mean_heading_diff = np.divide(mean_heading_diff, count)
	# Dumb way to get around a runtime warning when running the next line
	mean_heading_diff[mean_heading_diff != mean_heading_diff] = max_angle + 1
	mean_heading_diff[mean_heading_diff > max_angle] = None
	# Plot each section of np.array onto a grid looking plot
	# filling in segments with their heading difference weighted color
	ax = sns.heatmap(mean_heading_diff, cmap='Spectral',vmax=max_angle, linewidth=0,
	                 xticklabels=[x for x in range(-radius, radius, box_size)],
	                 yticklabels=[-y for y in range(-radius, radius, box_size)])
	# ax.set_title("Generic Title")
	ax.set_ylabel('Back-Front (m)')
	ax2 = ax.twinx()
	ax2.tick_params(
		axis='y',      # changes apply to the x-axis
		which='both',  # both major and minor ticks are affected
		right=False)
	ax2.set_yticklabels([])
	ax2.set_ylabel("Mean Abs. Heading Diff.", rotation=270, labelpad=60)

	ax.set_xlabel("Left-Right (m)")
	# ax.axhline(0+radius+box_size/2, color='black')
	# ax.axvline(0+radius+box_size/2, color='black')
	plt.show()


CENTER_IDX = 0 # 'center'
FRAME_IDX = 1
VEC_TO_NEXT_IDX = 2 # 'vec_to_next'