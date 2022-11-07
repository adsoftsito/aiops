# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet#pretrained-models

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import base64
import io
import json

import numpy as np
from PIL import Image
import requests

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
#SERVER_URL = 'http://localhost:8500/v1/models/resnet:predict'
SERVER_URL = 'https://resnet-service-adsoftsito.cloud.okteto.net/v1/models/resnet:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
#IMAGE_URL = "https://www.pngkey.com/png/detail/1-16373_dogs-png-jpg-royalty-free-library-dog-and.png"
#IMAGE_URL = "https://www.pngkey.com/png/detail/1-18245_single-rose-image-single-red-rose-red-rose.png"
# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False


def main():
  # Download the image
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()

  if MODEL_ACCEPT_JPG:
    print ("JPG")
    # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  else:
    print ("No JPG")
    # Compose a JOSN Predict request (send the image tensor).
    jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
    #predict_request = json.dumps({'instances': jpeg_rgb})
    jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')

    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes 
    #base64.b64decode(dl_request.content).decode()
  

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 10
  index = 0
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][index]['classes']
#    prediction = response.json()
    print (prediction)
    #index = index + 1


#  print (prediction)
  print('Prediction class: {}, avg latency: {} ms'.format(
      np.argmax(prediction), (total_time * 1000) / num_requests))


if __name__ == '__main__':
  main()
