FROM python:3.11.3

# Uncomment the following lines to enable GPU support.
# See https://skiff.allenai.org/gpu.html for more details.
#
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install -y nginx
RUN apt-get install -y openjdk-17-jre-headless openjdk-17-jdk-headless
RUN apt-get install -y curl less vim sudo tmux git

WORKDIR /grobid
RUN git clone https://github.com/kermitt2/grobid.git grobid
RUN git clone https://github.com/kermitt2/grobid_client_python.git grobid_client_python

WORKDIR /grobid/grobid
RUN git checkout 1c3efc247ba62ccd47d48ecc90901be88fe84865

RUN ./gradlew clean install

#RUN yes | unminimize

WORKDIR /tmp

# Install Python dependencies
RUN pip install grobid_client_python

# TODO: Need to install the actual doc2json module and the aries module
COPY requirements_doc2json.txt .
RUN pip install -r requirements_doc2json.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install cffi numpy nltk openai==0.28 tiktoken
RUN pip install --upgrade boto3


# Copy over the source code
COPY s2orc-doc2json /tmp/s2orc-doc2json
RUN pip install /tmp/s2orc-doc2json

WORKDIR /reviewgen
COPY . .

# Kick things off
ENTRYPOINT [ "/usr/local/bin/python", "run_reviewgen.py" ]
