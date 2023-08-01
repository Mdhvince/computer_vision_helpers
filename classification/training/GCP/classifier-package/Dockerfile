# Dockerfile for building the classifier package on GCP based on Pytorch
# pre-built images can be found here: https://cloud.google.com/deep-learning-containers/docs/choosing-container

# base container image. We will build on top of this
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12.py310

# directory in the container where subsequent commands will be executed
WORKDIR /

# copy the trainer package into the container
COPY trainer /trainer

# Set up the entry point to invoke the trainer (exec the task.py file)
ENTRYPOINT ["python", "-m", "trainer.task"]

