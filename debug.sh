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
sudo nvidia-docker run \
    --net=host \
    -v `pwd`/../scratch/:/DATASET:z \
    -e AICROWD_DATASET_PATH=/DATASET/goals.npy.npz \
    -it ${IMAGE_NAME}:${IMAGE_TAG} \
    /home/aicrowd/run.sh
