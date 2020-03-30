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

> Currently multi-input model only support gRPC request 


```python
# pyton3.x 
# tensorflow 2.x

import grpc
from PIL import Image
import numpy as np
import tensorflow as tf 
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import matplotlib.pyplot as plt

# Load Image 
content_img = Image.open('./test_content_img1.jpg')
style_img = Image.open('./test_style_img3.jpg')

def img_preprocess(img):
    h, w = img.size
    IMG_LONG_SIZE = 700.   # during test time, 700 size is the largest supported size 

    if h > w : # h is the long side 
        h_new = int(IMG_LONG_SIZE)
        w_new = int(IMG_LONG_SIZE * (w * (1.0) ) / ( h  * (1.0) ) )
    else:      # w is the long side 
        w_new = int(IMG_LONG_SIZE)
        h_new = int(IMG_LONG_SIZE * (h * (1.0) ) / ( w  * (1.0) ) )
    img = img.resize((h_new, w_new), resample=Image.BILINEAR)
    
    img = np.array(img).astype(np.float32)
		img = np.expand_dims(img, axis=0) 
    
    return img 

content_img = img_preprocess(content_img)
style_img = img_preprocess(style_img)

print('input content image resized size', content_img_np.shape)
print('input style image resized size', style_img_np.shape)
print('input image dtype', style_img_np.dtype)  # foloat 32 

plt.figure()
plt.imshow(content_img[0].astype(np.uint8))
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(style_img[0].astype(np.uint8))
plt.axis('off')
plt.show()

# Send gRPC request 
ip_port = "0.0.0.0:8500"
channel = grpc.insecure_channel(ip_port)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = "arbitary_style"
request.model_spec.signature_name = "predict_images" 
request.inputs["content_img"].CopyFrom(tf.make_tensor_proto(content_img, shape=list(content_img.shape)))  
request.inputs["style_img"].CopyFrom(tf.make_tensor_proto(style_img, shape=list(style_img.shape)))  
response = stub.Predict(request, 10.0)  # 10 second for time out 
output_img = tf.make_ndarray(response.outputs['output_img'])  # np.float32 

# Save output 
output_img = output_img.astype(np.uint8)
output_img_pil = Image.fromarray(output_img)
output_img_pil.save('path/to/target/img/file')


```

