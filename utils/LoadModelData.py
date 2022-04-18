import os

import requests


def get_model_parameter(id):
    url = "http://localhost:9876/api/data/" + str(id)
    response = requests.get(url)
    parameters = eval(response.text)
    return parameters


def set_model_result(id, res):
    url = "http://localhost:9876/api/data/" + str(id)
    requests.post(url, data=res)


def set_model_path(id, res):
    url = "http://localhost:9876/api/data/train/" + str(id)
    requests.post(url, data=res)


def send_pid(pid):
    url = "http://localhost:9876/api/data/setPid/" + str(pid)
    requests.post(url)


if __name__ == '__main__':
    print(get_model_parameter(29))
