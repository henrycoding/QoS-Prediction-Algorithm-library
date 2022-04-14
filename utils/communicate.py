import json
import os
import socket
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.FNCF.model import start_train
from utils.LoadModelData import get_model_parameter, send_pid
from models.FNCF.predict import start_predict


# class SocketServer:
#
#     def __init__(self, port=55534) -> None:
#         super().__init__()
#         # 开启ip和端口
#         self.addr = None
#         self.conn = None
#         self.ip_port = ('127.0.0.1', port)
#         # 生成一个句柄
#         self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         # 绑定ip端口
#         self.sk.bind(self.ip_port)
#         self.sk.listen(5)
#
#     def wait_client(self):
#         print('server 55533 running...')
#         print(os.getpid())
#         self.conn, self.addr = self.sk.accept()
#         # 获取客户端请求数据
#         self.conn.settimeout(1)
#         client_data = self.conn.recv(1024)
#         # 解码 'UTF-8'
#         flag = str(client_data, 'UTF-8').split("-")
#         print(flag)
#         self.send_message(os.getpid())
#
#         parameters = get_model_parameter(flag[1])
#         print(parameters)
#         if flag[0] == "p":
#             start_predict(parameters)
#         elif flag[0] == 't':
#             start_train(parameters)
#         # self.wait_client()
#
#     def send_message(self, message):
#         msg = json.dumps(message, ensure_ascii=False)
#         self.conn.send(msg.encode('utf-8'))
#         self.conn.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        send_pid(os.getpid())
        flag = sys.argv[1].split("-")
        parameters = get_model_parameter(flag[1])
        print(parameters)
        if flag[0] == "p":
            start_predict(parameters)
        elif flag[0] == 't':
            start_train(parameters)

