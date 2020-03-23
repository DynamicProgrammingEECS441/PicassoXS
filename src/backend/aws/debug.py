# python file
from PIL import Image
import json
import requests
import numpy as np
import boto3

# Load Image
input_img = np.array(Image.open('test.jpg'))
input_img = np.expand_dims(input_img, axis=0)            # (1, H, W, 3)
print('image size', input_img.shape)

# Prepare request
data = json.dumps({"signature_name": "predict_images",
                   "instances": input_img.tolist()})
headers = {"content-type": "application/json"}

import pdb; pdb.set_trace()

# Send Request
json_response = requests.post('http://172.31.47.255:0001/v1/models/model:predict', data=data, headers=headers)

# Load response
output_img = json.loads(json_response.text)['predictions']
output_img = np.asarray(output_img)                     # (1, H, W, 3)
output_img = output_img[0]                              # (H, W, 3)
output_img = np.uint8(output_img)                       # (H, W, 3)

# Plot and check result
#plt.figure()
#plt.axis('off')
#plt.imshow(np.uint8(output_img))
#plt.show()
#plt.imshow()

# Save Image
output_img_pil = Image.fromarray(output_img)
output_img_pil.save('test_response.jpg')
