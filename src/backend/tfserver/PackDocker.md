# Pack Docker 
> This file intend to teach backend developer on how to pack `servable` into the docker we need. 
> Two formate of Docker are supported for fast development and deployment 
> 1. Docker Images with one model per each image 
> 2. Docker Images with all models inside one image

## Build Single-Model Docker from Scratch 
```shell
# 1. Start a docker container called "serving base"
# -d command allow docker run in background 
docker run -d --name serving_base tensorflow/serving:latest

# 2. Copy all servable into the "serving base" container you just run 
pwd 
>> src/backend/tfserver/
docker cp $(pwd)/servable/${MODEL_NAME} serving_base:/models/ 
docker cp $(pwd)/servable/arbitary_style serving_base:/models/ 

# 3. Get inside the "serving base" container
docker exec -it serving_base /bin/bash 

# 4. Change the layout of the `/models/` directory to have 
# /models/monet 
# /models/van-gogh 

# 5. Commit change 
docker commit --change "ENV MODEL_NAME ${MODEL_NAME}" serving_base xiaosong99/servable:${MODEL_NAME}
docker commit --change "ENV MODEL_NAME arbitary_style" serving_base xiaosong99/servable:arbitary_style

# 7. Run the new commit docker image to ensure everything works correct 
docker run -t -p 8500:8500 -p 8501:8501 --name test xiaosong99/servable:${MODEL_NAME}
docker run -t -p 8500:8500 -p 8501:8501 --name test xiaosong99/servable:arbitary_style

```
 For `SendRequestArbitaryStyleModel_gRPC.py` and `SendRequestGeneralModel_gRPC.py`, change line with `request.model_spec.name =` to the new model you just addes, change other required settings (e.g ip_port), run the python file to see if new docker work 

 **NOTICE: DO NOT `docker push` YOUR DOCKER WITHOUT TESTING** 

```shell
# 8. Upload your new docker image 
docker push xiaosong99/servable:${MODEL_NAME}
docker push xiaosong99/servable:arbitary_style

# 9. Tag the docker image you just build to your tensorflow serving project storage 
docker tag xiaosong99/servable:${MODEL_NAME} gcr.io/tensorflow-serving-9905/servable:${MODEL_NAME}
docker push gcr.io/tensorflow-serving-9905/servable:${MODEL_NAME}

docker tag xiaosong99/servable:arbitary_style gcr.io/tensorflow-serving-9905/servable:arbitary_style
docker push gcr.io/tensorflow-serving-9905/servable:arbitary_style

# 10. clean up enviroment 
docker kill serving_base  # stop the image 
docker rm serving_base    # remove the image temp save 
docker kill test 
docker rm test 
```


## Build Multi-Model Docker from Scratch 
```shell
# 1. Start a docker container called "serving base"
# -d command allow docker run in background 
docker run -d --name serving_base tensorflow/serving:latest

# 2. Copy all servable into the "serving base" container you just run 
pwd 
>> src/backend/tfserver/
docker cp $(pwd)/servable/ serving_base:/models/ 
docker cp $(pwd)/models.config serving_base:/models/

# 3. Get inside the "serving base" container
docker exec -it serving_base /bin/bash 

# 4. Change the layout of the `/models/` directory to have 
# /models/models.config 
# /models/monet 
# /models/van-gogh 

# 5. Change starting setting 
(inside docker container) cd usr/bin/
(inside docker container) apt-get update
(inside docker container) apt-get install vim
(inside docker container) vim tf_serving_entrypoint.sh
# change the `vim tf_serving_entrypoint.sh` file to this 
```
```shell
#!/bin/bash
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/models/models.config
``` 
```shell
# 6. Check if the modify you just make is correct 
(inside docker container) cd usr/bin/
(inside docker container) sh tf_serving_entrypoint.sh 
(inside docker container) ^C # control c to stop the service once you know the system work 

# 6. Commit change 
docker commit serving_base xiaosong99/servable:latest

# 7. Run the new commit docker image to ensure everything works correct 
docker run -t -p 8500:8500 -p 8501:8501 --name test xiaosong99/servable:latest
```
 For `SendRequestArbitaryStyleModel_gRPC.py` and `SendRequestGeneralModel_gRPC.py`, change line with `request.model_spec.name =` to the new model you just addes, change other required settings (e.g ip_port), run the python file to see if new docker work 

 **NOTICE: DO NOT `docker push` YOUR DOCKER WITHOUT TESTING** 

```shell
# 8. Upload your new docker image 
docker push xiaosong99/servable:latest

# 9. Tag the docker image you just build to your tensorflow serving project storage 
docker tag xiaosong99/servable:latest gcr.io/tensorflow-serving-9905/servable:latest
docker push gcr.io/tensorflow-serving-9905/servable:latest

# 10. clean up enviroment 
docker kill serving_base  # stop the image 
docker rm serving_base    # remove the image temp save 
docker kill test 
docker rm test 
```

