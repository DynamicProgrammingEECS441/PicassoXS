# Train 
> Few common use instructions for training a model 


## Copy files into remote server 
```shell
# -r allow copy all file from directory 
scp -r /path/to/local/files <name>@<ip>:/path/to/server/files
scp -r $(pwd)/*.py  xiaosx@<ip>:~/train_model 
scp $(pwd)/faces.tgz  xiaosx@141.212.115.159:/home/xiaosx/dataset

# compress file for shipping
tar -czvf LotsOfFiles.tgz LotsOfFiles
tar -xvf faces.tgz
```

## Create conda virtual enviroment 
```shell
# Download Anaconda 
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh

# Create virtual enviroment using `.yml` file 
# Tested on ubuntu 18
conda env export > tf1-gpu.yml
conda env export > tf2-gpu.yml

conda env create -f tf2-gpu.yml 
conda env create -f tf2-gpu.yml 

# Create virtual enviroment from scratch 
# https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
conda create -n tf1-gpu python=3.6 tensorflow-gpu=1.12.0
conda create -n tf-gpu python=3.7.7 tensorflow-gpu=2.0

# Check if install enviroment correctly 
source  ~/.bashrc
conda activate tf2-gpu 
conda activate tf1-gpu
python
```
Inside python enviroment 
```python
import os 
import tensorflow as tf 

# check if this is the version you just installed 
tf.__version__

# set up GPU visibility 
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# check if GPU enviroment set up correctly 
tf.test.is_gpu_available() 
```


## Train Model & Monitor with Tensorboard 
```shell
# Link localhost port to server port 
ssh -N -f -L localhost:8004:localhost:8004 xiaosx@<ip>
ssh -N -f -L localhost:8004:localhost:8004 xiaosx@141.212.115.159

# Start tensorboard on server 
tensorboard --logdir="models/model_name/logs" --port=8000
tensorboard --logdir="models/model_francoise/logs" --port=8004

# Open monitor on your local device 
http://localhost:8000 
http://localhost:8004

# Run training model 
CUDA_VISIBLE_DEVICES=0 python main.py \
                 --model_name=francoise \
                 --phase=train \
                 --image_size=762 \
                 --ptcd=$(pwd)/places365 \
                 --ptad=$(pwd)/style/francoise \
                 --batch_size=3

CUDA_VISIBLE_DEVICES=9 python main.py \
                 --model_name=model_francoise \
                 --batch_size=1 \
                 --phase=train \
                 --image_size=768 \
                 --ptcd=/home/xiaosx/dataset/face \
                 --ptad=/home/xiaosx/dataset/style/francoise 

# Monitor GPU capacity 
watch -n 0.5 nvidia-smi
```

## Copy file back into local machine 
```shell
scp -r xiaosx@<ip>:~/train_model/models/ .
scp -r xiaosx@141.212.115.159:~/personal/models .
```