import os
import requests


def send_pid(pid):
    url = "http://localhost:8080/";
    requests.post(url, data={"pid": pid})
