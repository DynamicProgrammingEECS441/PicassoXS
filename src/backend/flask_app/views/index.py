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

@flask_app.app.route('/', methods=['GET'])
def index():
    context = {
        "status": "success",
    }
    return flask.jsonify(**context)