#!/bin/bash

set -e

uv run yolo \
	predict \
	detect \
	model=$1 \
	source=$2 \
	project=inference \
	name=$3 \
	save_txt=true \
	save_frames=true \
	save_conf=true \
	show_labels=false \
	imgsz=1280 \
	classes=0,1,2,3,4


root_dir=inference/$3

labels_dir=$root_dir/labels
states_dir=$root_dir/states
smooth_dir=$root_dir/smooth_states
vis_dir=$root_dir/vis
frames_dir=$(find "$root_dir" -name "*_frames" -type d)

mkdir -p $states_dir
mkdir -p $vis_dir

# Parse the predictions from the model
uv run parse-yolo-predictions "$labels_dir" "$states_dir"

# Run smoothing on parseable frames
uv run smooth-frame-predictions "$states_dir" "$smooth_dir"

# Detect Discret state changes (moves)
# TODO

# Visualize the parsed states with the captured frames
uv run visualize-state "$states_dir" "$vis_dir" --frames "$frames_dir"

# Create a video visualization of the individual visualization frames
ffmpeg \
	-framerate 4 \
	-i $vis_dir/vis_%04d.jpg \
	-c:v libx264 \
	-pix_fmt yuv420p \
	$root_dir/visualization.mp4
