#!/bin/bash



# Build Docker Image

export IMAGE_NAME="real-robot-random-agent"
REPO2DOCKER="$(which aicrowd-repo2docker)"
sudo ${REPO2DOCKER} --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name ${IMAGE_NAME} \
  --debug .


# Run local_evaluation inside docker container
# The next step assumes that you have downloaded the goals.npy.npz file to the 
# data folder of your repository
sudo nvidia-docker run \
    --net=host \
    -v `pwd`/data:/DATASET:z \
    -e AICROWD_DATASET_PATH=/DATASET/goals.npy.npz \
    -it ${IMAGE_NAME}:${IMAGE_TAG} \
    /home/aicrowd/run.sh
