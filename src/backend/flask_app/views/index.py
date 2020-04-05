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

def img_resize(img):
    '''
    Input:
        img (PIL.Image) : <PIL.Image> class objecy represent image 
    '''
    h = img.shape[0]
    w = img.shape[1]
    IMG_LONG_SIZE = 700.

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

@flask_app.app.route('/upload_img/', methods=['POST'])
def upload_img():
    filestr = flask.request.files.to_dict()['document'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    # Resize image
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    print("image size:", img.shape)
    cv2.imwrite("input.jpg", img)
    
    input_img = np.expand_dims(img, axis=0)            # (1, H, W, 3)

    # Prepare request
    data = json.dumps({"signature_name": "predict_images", 
                    "instances": input_img.tolist()})
    headers = {"content-type": "application/json"}
    print("send request to tf-server...")
    # Send Request
    # json_response = requests.post('http://localhost:${MACHINE_PORT_FOR_RESTfil}/v1/models/model:predict', \
    #                             data=data, headers=headers)
    json_response = requests.post('http://104.198.231.48:8501/v1/models/van-gogh:predict', \
                                data=data, headers=headers)
    print("received request from tf-server...")
    # Load response 
    output_img = json.loads(json_response.text)['predictions']
    output_img = np.asarray(output_img)                     # (1, H, W, 3)
    output_img = output_img[0]                              # (H, W, 3)
    # output_img = np.uint8(output_img)                       # (H, W, 3)

    # denormalize the image
    output_img = ((output_img + 1.) * 127.5).astype(np.uint8)
    # print("output_img", output_img.max(), output_img.min())
    # Save Image 
    cv_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output.jpg", cv_img)

    # convert numpy array to PIL Image
    output_img = Image.fromarray(output_img)

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    output_img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return flask.send_file(file_object, mimetype='image/PNG')

@flask_app.app.route('/general_model_grpc/', methods=['POST'])
def general_model_grpc():
    filestr = flask.request.files.to_dict()['document'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    img = img_resize(img)
    img = np.expand_dims(np.array(img).astype(np.float32), axis=0)  # float32, (1, h, w, 3) representaiton 

    # 3. Prepare & Send Request 
    ip_port = "0.0.0.0:32770"  # TODO change this to your ip:port 
    channel = grpc.insecure_channel(ip_port)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "monet"  # TODO change this to the model you're using 
    request.model_spec.signature_name = "predict_images" 
    request.inputs["input_img"].CopyFrom(  
        tf.make_tensor_proto(img, shape=list(img.shape))) 
    response = stub.Predict(request, 10.0)  # TODO change the request timeout, default is 10s

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

@flask_app.app.route('/', methods=['GET'])
def index():
    context = {
        "status": "success",
    }
    return flask.jsonify(**context)