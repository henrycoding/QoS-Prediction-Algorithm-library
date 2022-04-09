import requests


def get_model_parameter(id):
    url = "http://localhost:9876/api/data/" + id
    response = requests.get(url)
    parameters = eval(response.text)
    return parameters


def set_model_result(id, res):
    url = "http://localhost:9876/api/data/" + str(id)
    response = requests.post(url, data=res)


if __name__ == '__main__':
    set_model_result(1)
