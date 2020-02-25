# Demo on TF Server 

> This directory is used to show how to build a small tf server on linux machine 

## Install TF Server on local linus machine 

* Download Docker 

[download docker instruction](https://docs.docker.com/install/)

[docker tutorial](https://yeasy.gitbooks.io/docker_practice/image/pull.html)


* Common use command 

```shell
# Find running docker 
docker ps -a

# Stop given docker image
docker stop <container_id>


```

## Use TF Server & Docker 

[tutorial 1](https://zhuanlan.zhihu.com/p/52096200)

[tutorial 2](https://zhuanlan.zhihu.com/p/96917543)

[tutorial 3](https://zhuanlan.zhihu.com/p/64413178)

[official document CH_zh](https://bookdown.org/leovan/TensorFlow-Learning-Notes/4-5-deploy-tensorflow-serving.html#serving-a-tensorflow-model--tensorflow-)



* Ideas 

use model outside docker 

log into docker & use tensorflow server 

pack model into part of the tf model 


```shell
# Run model by mount 
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving 


```