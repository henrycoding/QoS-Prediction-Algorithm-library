import socket
import json

from utils.LoadModelData import get_model_parameter
from models.FNCF.predict import start_predict


class SocketServer:

    def __init__(self) -> None:
        super().__init__()
        # 开启ip和端口
        self.addr = None
        self.conn = None
        self.ip_port = ('127.0.0.1', 55533)
        # 生成一个句柄
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定ip端口
        self.sk.bind(self.ip_port)
        self.sk.listen(5)

    def wait_client(self):
        print('server 55533 running...')
        self.conn, self.addr = self.sk.accept()
        # 获取客户端请求数据
        self.conn.settimeout(1)
        client_data = self.conn.recv(1024)
        # 解码 'UTF-8'
        flag = str(client_data, 'UTF-8').split("-")
        print(flag)
        parameters = get_model_parameter(flag[1])
        print(parameters)
        if flag[0] == "p":
            start_predict(parameters)
        self.wait_client()

    def send_message(self, message):
        msg = json.dumps(message, ensure_ascii=False)
        self.conn.send(msg.encode('utf-8'))
        self.conn.close()
        self.wait_client()


if __name__ == '__main__':
    socketServer = SocketServer()
    socketServer.wait_client()
