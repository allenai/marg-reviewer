FROM python:3.11.3

# Uncomment the following lines to enable GPU support.
# See https://skiff.allenai.org/gpu.html for more details.
#
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y curl less vim sudo tmux git

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install cffi numpy nltk openai tiktoken

WORKDIR /api
COPY . .

# Kick things off
ENTRYPOINT [ "/api/dev.sh" ]
