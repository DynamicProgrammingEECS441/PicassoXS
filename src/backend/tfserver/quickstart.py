from PIL import Image
import json
import requests
import numpy as np

# Load Image
img = Image.open('./test_input_img3.jpg')
print('input image original size', img.size)

# Resize long side to 1024
h, w = img.size

if h > w : # h is the long side 
    h_new = int(128.)
    w_new = int(128. * (w * (1.0) ) / ( h  * (1.0) ) )
else:      # w is the long side 
    w_new = int(128.)
    h_new = int(128. * (h * (1.0) ) / ( w  * (1.0) ) )


img = img.resize((h_new, w_new), resample=Image.BILINEAR)
print('input image resized size', img.size)

input_img = np.array(img)
input_img = np.expand_dims(input_img, axis=0)            # (1, H, W, 3)
input_img = input_img / 127.5 - 1 
print('input image range {} - {}'.format(np.min(input_img), np.max(input_img)))

# Prepare request
data = json.dumps({"signature_name": "predict_images",
                   "instances": input_img.tolist()})
headers = {"content-type": "application/json"}

# Send Request
url = 'http://35.202.143.174:8501/v1/models/van-gogh:predict'
#url = 'http://localhost:8501/v1/models/van-gogh:predict'
json_response = requests.post(url, \
                              data=data, headers=headers)

# Load response
output_img = json.loads(json_response.text)['predictions']
output_img = np.asarray(output_img)                     # (1, H, W, 3)
output_img = output_img[0]                              # (H, W, 3)
output_img = (output_img + 1.) * 127.5
output_img = np.uint8(output_img)                       # (H, W, 3)

print('output image range {} - {}'.format(np.min(output_img), np.max(output_img)))

# Plot and check result
#plt.figure()
#plt.axis('off')
#plt.imshow(np.uint8(output_img))
#plt.show()
#plt.imshow()

# Save Image
output_img_pil = Image.fromarray(output_img)
output_img_pil.save('./test_output_img3_van-gogh.jpg')
