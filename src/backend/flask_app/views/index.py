"""
Insta485 index (main) view.

URLs include:
/
"""
import flask
import flask_app
import numpy as np
import cv2 
import json
import requests
from PIL import Image
import io
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2 # TODO you need to download tensorflow_serving on your server 
from tensorflow_serving.apis import prediction_service_pb2_grpc
from random import randint
import time
from typing import Optional, Tuple

class SleepingPolicy():
    def sleep(self, try_i: int):
        """
        How long to sleep in milliseconds.
        :param try_i: the number of retry (starting from zero)
        """
        assert try_i >= 0

class ExponentialBackoff(SleepingPolicy):
    def __init__(self, *, init_backoff_ms: int, max_backoff_ms: int, multiplier: int):
        self.init_backoff = randint(0, init_backoff_ms)
        self.max_backoff = max_backoff_ms
        self.multiplier = multiplier

    def sleep(self, try_i: int):
        sleep_range = min(
            self.init_backoff * self.multiplier ** try_i, self.max_backoff
        )
        sleep_ms = randint(0, sleep_range)
        print("sleeping for ", sleep_ms)
        time.sleep(sleep_ms / 1000)

class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor
):
    def __init__(
        self,
        *,
        max_attempts: int,
        sleeping_policy: SleepingPolicy,
        status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        self.max_attempts = max_attempts
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):

        for try_i in range(self.max_attempts):
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # Return if it was last attempt
                if try_i == (self.max_attempts - 1):
                    return response

                # If status code is not in retryable status codes
                if (
                    self.status_for_retry
                    and response.code() not in self.status_for_retry
                ):
                    return response

                self.sleeping_policy.sleep(try_i)
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        return self._intercept_call(continuation, client_call_details, request_iterator)

def img_resize(img):
    '''
    Input:
        img (PIL.Image) : <PIL.Image> class objecy represent image 
    '''
    h = img.shape[0]
    w = img.shape[1]
    IMG_LONG_SIZE = 1200.

    if h > w : # h is the long side 
        h_new = int(IMG_LONG_SIZE)
        w_new = int(IMG_LONG_SIZE * (w * (1.0) ) / ( h  * (1.0) ) )
    else:      # w is the long side 
        w_new = int(IMG_LONG_SIZE)
        h_new = int(IMG_LONG_SIZE * (h * (1.0) ) / ( w  * (1.0) ) )
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    return img 


def post_process(img):
    '''
    Input:
        img (np.array) :  value range : [-1, 1], dtype : float32
    Return:
        img (np.array) :  value range : [0, 255], dtype : uint8 
        # TODO change the return image having the same image range as the image you received 
    '''
    img = (img + 1.) * 127.5
    img = img.astype(np.uint8)
    return img 

@flask_app.app.route('/general_model_grpc/', methods=['POST'])
def general_model_grpc():
    request_dict = flask.request.files.to_dict()
    model_name = list(request_dict.keys())[0]
    filestr = request_dict[model_name].read()
    print(model_name)
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    img = img_resize(img)
    img = np.expand_dims(np.array(img).astype(np.float32), axis=0)  # float32, (1, h, w, 3) representaiton 

    # 3. Prepare & Send Request 
    # ip_port = "0.0.0.0:32770"  # TODO change this to your ip:port 

    interceptors = (
        RetryOnRpcErrorClientInterceptor(
            max_attempts=4,
            sleeping_policy=ExponentialBackoff(init_backoff_ms=100, max_backoff_ms=1600, multiplier=2),
            status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
        ),
    )

    ip_port = "35.232.203.191:8500"
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel(ip_port, options=channel_opt)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc.intercept_channel(channel, *interceptors))
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name  # TODO change this to the model you're using 
    request.model_spec.signature_name = "predict_images" 
    request.inputs["input_img"].CopyFrom(  
        tf.make_tensor_proto(img, shape=list(img.shape))) 
    response = stub.Predict(request, 100.0)  # TODO change the request timeout, default is 10s

    # 4. Image Postprocess 
    output_img = tf.make_ndarray(response.outputs['output_img'])  # numpy array (1, H, W, 3)
    output_img = post_process(output_img)

    # 5. Save Image / Send image back to frontend 
    output_img = Image.fromarray(output_img[0])
    output_img.save('test_gRPC_img1.jpg')

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    output_img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return flask.send_file(file_object, mimetype='image/PNG')


@flask_app.app.route('/arbitrary_style_grpc/', methods=['POST'])
def arbitrary_style_grpc():
    # 1. Load Image
    filestr = flask.request.files.to_dict()['content_img'].read()
    npimg = np.fromstring(filestr, np.uint8)
    content_img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    filestr = flask.request.files.to_dict()['style_img'].read()
    npimg = np.fromstring(filestr, np.uint8)
    style_img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    # 2. Image Preprocess 
    content_img = img_resize(content_img)
    style_img = img_resize(style_img)
    print("content_img size:", content_img.shape)
    print("style_img size:", style_img.shape)
    content_img_np = np.array(content_img).astype(np.float32)
    content_img_np = np.expand_dims(content_img_np, axis=0)  # float32, (1, h, w, 3) representaiton 

    style_img_np = np.array(style_img).astype(np.float32)
    style_img_np = np.expand_dims(style_img_np, axis=0) # float32, (1, h, w, 3) representaiton 

    # 3. Prepare & Send Request 
    interceptors = (
        RetryOnRpcErrorClientInterceptor(
            max_attempts=4,
            sleeping_policy=ExponentialBackoff(init_backoff_ms=100, max_backoff_ms=1600, multiplier=2),
            status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
        ),
    )
    ip_port = "35.232.203.191:8500"
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel(ip_port, options=channel_opt)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc.intercept_channel(channel, *interceptors))
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "arbitary_style" # TODO change this to the model you're using 
    request.model_spec.signature_name = "predict_images" 
    request.inputs["content_img"].CopyFrom(  
            tf.make_tensor_proto(content_img_np, shape=list(content_img_np.shape)))  
    request.inputs["style_img"].CopyFrom(  
            tf.make_tensor_proto(style_img_np, shape=list(style_img_np.shape)))  
    response = stub.Predict(request, 200.0)  # TODO change the request timeout, default is 10s
    
    # 4. Image Postprocess 
    output_img = tf.make_ndarray(response.outputs['output_img']) # value range : [0-1], dtype float32, (1, H, W, 3)
    # output_img = output_img * 255 
    output_img = output_img.astype(np.uint8)  # value range : [0-255], dtype : uint8, (1, H, W, 3)
    output_img = Image.fromarray(output_img[0])
    output_img.save('test_gRPC_img2.jpg')

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    output_img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return flask.send_file(file_object, mimetype='image/PNG')


@flask_app.app.route('/', methods=['GET'])
def index():
    context = {
        "status": "success",
    }
    return flask.jsonify(**context)