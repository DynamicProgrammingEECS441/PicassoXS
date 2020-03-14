# Quick Start on using Docker to deploy existing model

> This document is mainly for frontend developer 
>
> For backend developer, check `instruction.md` for more information  



## Process 

1. Download Docker Enviroment 

[docker download instruction](https://docs.docker.com/install/)




2. Docker pull packed image 

see currently supported model here [xiaosong99/servable](https://hub.docker.com/repository/docker/xiaosong99/servable)

```shell
>> docker pull xiaosong99/servable:${MODEL_NAME}
>> docker pull xiaosong99/servable:monet 

>> docker images
REPOSITORY            TAG                 IMAGE ID            CREATED             SIZE
xiaosong99/servable   morisot             98ea28dcbc2a        4 minutes ago       299MB
xiaosong99/servable   kandinsky           6b93f20b142b        5 minutes ago       299MB
xiaosong99/servable   van-gogh            1653250eb149        6 minutes ago       299MB

```




3. Start multiple docker image with different port 
   1. each docker image take 2 port
   2. you need to specify the machine port and docker port (8501, 8500 fixed)


```shell
>> docker run -t -p ${MACHINE_PORT_FOR_gRPC}:8500 -p ${MACHINE_PORT_FOR_RESTfil}:8501 xiaosong99/servable:${MODEL_NAME}
>> docker run -t -p 0000:8500 -p 0001:8501 xiaosong99/servable:morisot
>> docker run -t -p 0002:8500 -p 0003:8501 xiaosong99/servable:kandinsky

2020-03-14 15:44:42.811819: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:203] Restoring SavedModel bundle.
2020-03-14 15:44:42.917481: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:333] SavedModel load for tags { serve }; Status: success: OK. Took 156838 microseconds.
2020-03-14 15:44:42.919987: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:105] No warmup data file found at /models/model/1/assets.extra/tf_serving_warmup_requests
2020-03-14 15:44:42.923144: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: model version: 1}
2020-03-14 15:44:42.925616: I tensorflow_serving/model_servers/server.cc:358] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2020-03-14 15:44:42.926890: I tensorflow_serving/model_servers/server.cc:378] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...

```



4. Send RESTful request to Docker 


```python
# python file 
from PIL import Image
import json
import requests
import numpy as np 

# Load Image 
input_img = np.array(Image.open('path/to/source/img/file'))
input_img = np.expand_dims(input_img, axis=0)            # (1, H, W, 3)

# Prepare request
data = json.dumps({"signature_name": "predict_images", 
                   "instances": input_img.tolist()})
headers = {"content-type": "application/json"}

# Send Request
json_response = requests.post('http://localhost:${MACHINE_PORT_FOR_RESTfil}/v1/models/model:predict', \
                              data=data, headers=headers)
json_response = requests.post('http://localhost:0001/v1/models/model:predict', \
                              data=data, headers=headers)

# Load response 
output_img = json.loads(json_response.text)['predictions']
output_img = np.asarray(output_img)                     # (1, H, W, 3)
output_img = output_img[0]                              # (H, W, 3)
output_img = np.uint8(output_img)                       # (H, W, 3)

# Plot and check result 
plt.figure()
plt.axis('off')
plt.imshow(np.uint8(output_img))
plt.show()
plt.imshow()

# Save Image 
output_img_pil = Image.fromarray(output_img)
output_img_pil.save('path/to/target/img/file')
```


```shell
# Below line indicate docker successful execute out request
2020-03-14 16:00:19.743166: W external/org_tensorflow/tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 91750400 exceeds 10% of system memory.
```