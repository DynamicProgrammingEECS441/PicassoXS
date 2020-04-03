import numpy as np
from PIL import Image

def imageToNumpy(image):
    image = np.asArray(image)
    userUpload = Image.open(image)
    userUpload.show()
    w,h = userUpload.size

    print(userUpload.size)

    width = w
    height = h

    if w > 1500 :
        width = 1499
        height = h * (1499/w)
    if h > 1500 :
        height = 1499
        width = w *(1499/h)
    userUpload = userUpload.resize((int(width),int(height)))
    userUpload.show()

    
