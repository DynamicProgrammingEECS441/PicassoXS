import grpc
from PIL import Image
import tensorflow as tf  # tf 2.x 
import numpy as np
# pip install tensorflow-serving-api 
from tensorflow_serving.apis import predict_pb2 # TODO you need to download tensorflow_serving on your server 
from tensorflow_serving.apis import prediction_service_pb2_grpc

def main():
    # 1. Load Image
    content_img = Image.open('./test_content_img1.jpg') # TODO change this to your imaeg read in  
    style_img = Image.open('./test_style_img1.jpg')

    # 2. Image Preprocess 
    # 2.1  Resize 
    # 2.2  Convert to numpy float 32 (1, H, W, 3) representation 
    # For code run on server
    #   you may need to find a way to resize image 
    #   numpy image representation is MANDATORY, this is due to the input formate of tf.make_tensor_proto
    def img_resize(img):
        '''
        Input:
            img (PIL.Image) : <PIL.Image> class objecy represent image 
        '''
        h, w = img.size
        IMG_LONG_SIZE = 256.

        if h > w : # h is the long side 
            h_new = int(IMG_LONG_SIZE)
            w_new = int(IMG_LONG_SIZE * (w * (1.0) ) / ( h  * (1.0) ) )
        else:      # w is the long side 
            w_new = int(IMG_LONG_SIZE)
            h_new = int(IMG_LONG_SIZE * (h * (1.0) ) / ( w  * (1.0) ) )
        img = img.resize((h_new, w_new), resample=Image.BILINEAR)
        return img 

    content_img = img_resize(content_img)
    style_img = img_resize(style_img)

    content_img_np = np.array(content_img).astype(np.float32)
    content_img_np = np.expand_dims(content_img_np, axis=0)  # float32, (1, h, w, 3) representaiton 

    style_img_np = np.array(style_img).astype(np.float32)
    style_img_np = np.expand_dims(style_img_np, axis=0) # float32, (1, h, w, 3) representaiton 

    # 3. Prepare & Send Request 
    ip_port = "0.0.0.0:8500"  # TODO change this to your ip:port 
    # if you run docker run -t -p 0000:8500 -p 0001:8501 xiaosong99/servable:latest-skeleton 
    # then the port should be "0000"
    # For more information, see `QuickStart_GeneralModel.md` 
    channel = grpc.insecure_channel(ip_port)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "arbitary_style" # TODO change this to the model you're using 
    request.model_spec.signature_name = "predict_images" 
    request.inputs["content_img"].CopyFrom(  
            tf.make_tensor_proto(content_img_np, shape=list(content_img_np.shape)))  
    request.inputs["style_img"].CopyFrom(  
            tf.make_tensor_proto(style_img_np, shape=list(style_img_np.shape)))  
    response = stub.Predict(request, 10.0)  # TODO change the request timeout, default is 10s
    
    # 4. Image Postprocess 
    output_img = tf.make_ndarray(response.outputs['output_img']) # value range : [0-255], dtype float32, (1, H, W, 3)
    output_img = output_img.astype(np.uint8)  # value range : [0-255], dtype : uint8, (1, H, W, 3)
    output_img_pil = Image.fromarray(output_img[0])
    output_img_pil.save('test_output_gRPC_img1.jpg')

if __name__ == '__main__':
    main()