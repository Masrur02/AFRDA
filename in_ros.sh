#!/bin/bash

# MESH_DIR is the only argument passed
MESH_DIR=$1  # First argument for the 'mesh' or other folder name

# Fixed part of the path
BASE_PATH=$(find $HOME -type d -name "DA" 2>/dev/null | head -n 1)/work_dirs/local-basic

# Construct the full paths dynamically using the provided MESH_DIR
CONFIG_FILE="${BASE_PATH}/${MESH_DIR}/a.py"
CHECKPOINT_FILE="${BASE_PATH}/${MESH_DIR}/iter_40000.pth"
SHOW_DIR="${BASE_PATH}/${MESH_DIR}/preds"

echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE

# Run the python command with the constructed paths
python3 -m tools.inf_ros ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --opacity 0.5 --dataset "rugd"
### command for changing the navigability class
#rostopic pub /navigability std_msgs/String "data: '0,n'"
