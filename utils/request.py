import os
import requests


def send_model_result(res):
    url = "http://localhost:9000/setModelResult"
    requests.post(url, data=res)


def send_pid(pid):
    url = "http://localhost:9000/setPid"
    requests.post(url, data={"pid": str(pid)})
