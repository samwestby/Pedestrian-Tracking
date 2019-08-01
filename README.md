# Tracking Data Processor

Process object tracking data to be usable for research

Current application: track people walking and use the data to verify models of walking behavior

### Prerequisites

    python3 -m virtualenv .venv
    source .venv/bin/activate
    pip3 install numpy matplotlib cv2 pandas seaborn

`human_detection.py` requires an object detection model. I use the faster_rcnn_nas_coco_2018_01_28 [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


## Usage

1) Record video
2) Put video through `human_detection.py`
3) Get .xlsx file with bounding boxes
4) Convert .xlsx to .csv
5) Find desired coordinate transform -- See below
6) Changes values in 'main.py' to represent new values of *csv_path*, *pts_src*, *pts_dst*, *window*, *scale*, *shift*, and *path to still image*
7) Enjoy

__Step 5 in more detail__
  1. Get real-world measurements of a box from the recorded location. Larger area is better. This is pts_dst
  2. Find pixel coordinates of the same box from a still image of video. This is pts_src
  3. Use `tracker_api.warp_coordinates` to test coordinates
  4. Use `tracker_api.warp_coordinates` to find *window*, *shift*, *scale* values that make converted images display nicely

