import grpc
from PIL import Image
import tensorflow as tf  # tf 2.x 
import numpy as np
# pip install tensorflow-serving-api 
from tensorflow_serving.apis import predict_pb2 # TODO you need to download tensorflow_serving on your server 
from tensorflow_serving.apis import prediction_service_pb2_grpc

def main():
    # 1. Load image 
    # For code run on server, this should be image send from front end 
    img = Image.open('./test_input_img1.jpg')  # TODO change this to your imaeg read in 

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
    img = img_resize(img)
    img = np.expand_dims(np.array(img).astype(np.float32), axis=0)  # float32, (1, h, w, 3) representaiton 

    # 3. Prepare & Send Request 
    ip_port = "0.0.0.0:8500"  # TODO change this to your ip:port 
    # if you run docker run -t -p 0000:8500 -p 0001:8501 xiaosong99/servable:latest-skeleton  
    # then the port should be "0000"
    # For more information, see `QuickStart_GeneralModel.md`
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
    output_img = post_process(output_img)

    # 5. Save Image / Send image back to frontend 
    output_img_pil = Image.fromarray(output_img[0])
    output_img_pil.save('test_output_gRPC_img1.jpg')
    #plt.figure()
    #plt.imshow(output_img[0])
    #plt.axis('off')
    #plt.show()


if __name__ == '__main__':
    main()