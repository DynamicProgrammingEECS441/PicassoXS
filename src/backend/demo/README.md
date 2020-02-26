# Demo 

* Reference 

[download docker instruction](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

[docker tutorial](https://yeasy.gitbooks.io/docker_practice/image/pull.html)

[tf server tutorial 1](https://zhuanlan.zhihu.com/p/52096200)

[tf server tutorial 2](https://zhuanlan.zhihu.com/p/96917543)

[tf server tutorial 3](https://zhuanlan.zhihu.com/p/64413178)

[official document CH_zh](https://bookdown.org/leovan/TensorFlow-Learning-Notes/4-5-deploy-tensorflow-serving.html#serving-a-tensorflow-model--tensorflow-)



## Docker

* Basic Ideas 

1. Repository : package of multiple images
2. Images : class 
3. Container : Instance of class 



* Docker 

```shell
# Remove old version 
sudo apt-get remove docker docker-engine docker.io containerd runc
 
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```



* Common use command 

```shell
# Check all docker images 
docker images 

# stop task 
docker kill <container_id>    # at terminal 

exit                          # inside docker 

# Check running task 
docker ps 

# Check all runnning task in unix 
ps -ef
ps -ef | grep <keyword_for_search>

# Kill running task in unix 
kill -9 <PID>
```



## TF Server 

* Prepare servable 





* Start TF Server outside docker 







* Start TF Server after running docker 




```shell
# Run model by mount 
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving 

# Run by going inside 
sudo nvidia-docker run -it tensorflow/serving:latest-devel-gpu bash
-ot 
```





