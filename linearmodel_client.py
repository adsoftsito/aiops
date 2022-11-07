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
"""A client that performs inferences on a Linear model using the REST API.

The client  queries the server over the REST API
 repeatedly and measures how long it takes to respond.

   python linearmodel_client.py
"""

from __future__ import print_function

import io
import json

import numpy as np
import requests

# The server URL specifies the endpoint of your server running the linear_model
# model with the name "linear_model" and using the predict interface.
SERVER_URL = 'https://linear-model-service-adsoftsito.cloud.okteto.net/v1/models/linear_model:predict'



def main():
  predict_request = '{"instances" : [ [0.0], [1.0], [2.0] ]}'

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
    prediction = response.json()
    print (prediction)

  print('Prediction class: {}, avg latency: {} ms'.format(
      np.argmax(prediction), (total_time * 1000) / num_requests))


if __name__ == '__main__':
  main()
