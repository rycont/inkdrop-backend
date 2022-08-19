from pydantic import BaseModel
from fastapi import FastAPI
from tensorflow import keras
from typing import List
import numpy as np

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

def inkdrop_to_image(_input_image):
    input_image = _input_image.copy()

    if input_image.shape == (SIZE, SIZE):
        return lightness_to_hue(input_image)

    elif input_image.shape[0] >= SIZE and input_image.shape[1] >= SIZE:
        print("Working on")
        width = input_image.shape[0]
        height = input_image.shape[1]

        start_x = 0
        start_y = 0

        hue_map = np.zeros((width, height))
        count_map = np.zeros((width, height))

        while start_x < width:
            while start_y < height:
                patch = input_image[
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

class ReqBody(BaseModel):
    lightness: List[float]

@app.post("/inkdrop")
def inkdrop_api(
    body: ReqBody
):
    res = inkdrop_to_image(np.array(body.lightness))
    print(res)
    return res
