import os
import requests


def send_model_result(data):
    url = "http://localhost:9000/setModelResult"
    requests.post(url, data)


def send_pid(id, pid):
    url = "http://localhost:9000/setPid"
    requests.post(url, data={"id": int(id), "pid": str(pid)})


def send_persentage(id, percentage):
    url = "http://localhost:9000/setPercentage"
    requests.post(url, data={"id": int(id), "percentage": int(percentage)})
