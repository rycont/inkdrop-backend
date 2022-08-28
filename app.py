import sys
from io import BytesIO
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow import keras
import numpy as np
from utils import rgb2hsl, hsl2rgb
import json

np.set_printoptions(
    threshold = sys.maxsize,
    suppress = True
)

SIZE = 384
STRIDE = 192

inkdrop = keras.models.load_model('./model')

def lightness_to_hue(_image):
    image = _image.copy()

    origin_shape = image.shape
    image = np.pad(image, ((0, SIZE - origin_shape[0]), (0, SIZE - origin_shape[1])), 'constant', constant_values=(0, 0))

    image_b = image.reshape((1, SIZE, SIZE, 1))
    image_h = inkdrop.predict(image_b)[0].squeeze()

    return image_h[
         : origin_shape[0],
         : origin_shape[1]
    ]

def inkdrop_to_image(_lightness):
    input_lightness = _lightness.copy()

    if input_lightness.shape == (SIZE, SIZE):
        return lightness_to_hue(input_lightness)

    elif input_lightness.shape[0] >= SIZE or input_lightness.shape[1] >= SIZE:
        print("Working on")
        width = input_lightness.shape[0]
        height = input_lightness.shape[1]

        start_x = 0
        start_y = 0

        hue_map = np.zeros((width, height))
        count_map = np.zeros((width, height))

        while start_x < width:
            while start_y < height:
                patch = input_lightness[
                    start_x : start_x + SIZE,
                    start_y : start_y + SIZE
                ]
                
                patch_hue = lightness_to_hue(patch)
                hue_map[
                    start_x : start_x + SIZE,
                    start_y : start_y + SIZE
                ] += patch_hue.squeeze()

                count_map[start_x:start_x+SIZE, start_y:start_y+SIZE] += 1

                start_y += STRIDE

            start_x += STRIDE
            start_y = 0

        hue_map = hue_map / count_map

        return hue_map

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/inkdrop/")
def inkdrop_api(
    picture: bytes = File()
):
    rgb = np.array(Image.open(BytesIO(picture)))[:,:,:3]
    hsl = rgb2hsl(rgb)
    
    lightness = hsl[:,:,2]
    hue = inkdrop_to_image(lightness).round(3)

    return json.dumps(hue.tolist())
