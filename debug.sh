#!/bin/bash



# Build Docker Image

export IMAGE_NAME="real-robot-random-agent"
IMAGE_TAG=$(git rev-parse --short HEAD)
REPO2DOCKER="$(which aicrowd-repo2docker)"
sudo ${REPO2DOCKER} --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name ${IMAGE_NAME}:${IMAGE_TAG} \
  --debug .


# Run local_evaluation inside docker container
# The next step assumes that you have downloaded the goals.npy.npz file to the 
# data folder of your repository
sudo docker run \
    --gpus all \
    --net=host \
    -v `pwd`/data:/DATASET:z \
    -e AICROWD_DATASET_PATH=/DATASET/goals-REAL2020-s2020-50-1 \
    -it ${IMAGE_NAME}:${IMAGE_TAG} \
    /home/aicrowd/run.sh
