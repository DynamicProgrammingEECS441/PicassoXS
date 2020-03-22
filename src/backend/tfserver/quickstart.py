from PIL import Image
import PIL 
import json
import requests
import numpy as np

# Load Image
img = Image.open('./test_input_img1.jpg')
print('input image original size', img.size)

# Resize long side to 1024
h, w = img.size

if h > w : # h is the long side 
    h = 1600 
    w = 1600 * (w / h)
    w = int(w)
else:
    w = 1600 
    h = 1600 * (h / w)
    h = int(h)

img = img.resize((w, h), resample=Image.BILINEAR)
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
json_response = requests.post('http://localhost:0001/v1/models/model:predict', \
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
output_img_pil.save('./test_output_img1.jpg')
