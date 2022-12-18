FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean
RUN rm /etc/apt/sources.list.d/cuda.list && apt-get clean

ENV TZ=Brazil/East
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt install wget -y
RUN apt install python3 -y
RUN apt-get install python3-apt -y                                                              
RUN apt-get install python3-pip -y

RUN python3 -m pip install --upgrade pip

WORKDIR /home/project_ring_society
COPY ./requirements.txt /home/project_ring_society
RUN python3 -m pip install --default-timeout=900 -r ./requirements.txt

# Download anaconda and pytorch cuda package
RUN wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
RUN bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda
RUN rm Anaconda3-4.2.0-Linux-x86_64.sh
RUN ~/anaconda/bin/conda update conda
RUN ~/anaconda/bin/conda install pytorch torchvision torchaudio cudatoolkit -c pytorch

# CMD [ "python3", "./main.py"]

# Uncomment this line if you want the container stands up
ENTRYPOINT ["tail", "-f", "/dev/null"]
