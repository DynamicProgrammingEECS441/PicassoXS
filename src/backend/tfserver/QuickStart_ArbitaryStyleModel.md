# Quick Start on using Docker to deploy user upload style model 

> This document is mainly for frontend developer 
>
> For backend developer, check `instruction.md` for more information  
>
> Compare to `QuickStart.md`, instruction in this file are fixed to arbitary style model 



## Process 

1. Download Docker Enviroment 

[docker download instruction](https://docs.docker.com/install/)




2. Docker pull packed image 

see currently supported model here [xiaosong99/servable](https://hub.docker.com/repository/docker/xiaosong99/servable)

```shell
>> docker pull xiaosong99/servable:arbitary_style  # Docker image only contain arbitary style model 
>> docker pull xiaosong99/servable:latest-skeleton # Docker image contain all model
>> docker images
REPOSITORY            TAG                 IMAGE ID            CREATED             SIZE
xiaosong99/servable   arbitary_style      98ea28dcbc2a        4 minutes ago       299MB

```




3. Start multiple docker image with different port 
   1. each docker image take 2 port
   2. you need to specify the machine port and docker port (8501, 8500 fixed)

Note :

1. tf server runs use port `8500`(gRPC), `8501`(REST) `inside docker`
2. `-p` command "link" your local computer / server's port with docker port 
3. `-p 0000:8500` means you have link your local computer `0000` port with docker port `8500` and you can send your gRPC request to `localhost:0000` and this request will be re-direct to `8500` inside docker, which is the port that tf server use. 

```shell
>> docker run -t -p ${MACHINE_PORT_FOR_gRPC}:8500 -p ${MACHINE_PORT_FOR_RESTfil}:8501 xiaosong99/servable:${MODEL_NAME}
>> docker run -t -p 8500:8500 -p 8500:8500 xiaosong99/servable:arbitary_style
>> docker run -t -p 8500:8500 -p 8500:8500 xiaosong99/servable:latest-skeleton

2020-03-14 15:44:42.811819: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:203] Restoring SavedModel bundle.
2020-03-14 15:44:42.917481: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:333] SavedModel load for tags { serve }; Status: success: OK. Took 156838 microseconds.
2020-03-14 15:44:42.919987: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:105] No warmup data file found at /models/model/1/assets.extra/tf_serving_warmup_requests
2020-03-14 15:44:42.923144: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: model version: 1}
2020-03-14 15:44:42.925616: I tensorflow_serving/model_servers/server.cc:358] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2020-03-14 15:44:42.926890: I tensorflow_serving/model_servers/server.cc:378] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...

```



4. Send gRPC request to Docker 

See `SendRequestArbitaryStyleModel_gRPC.py` for latest version 

