from yacs.config import CfgNode

from models.base import ServerBase


class Server(ServerBase):
    """服务端只做模型参数的融合
    """

    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)
        self._params = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    def __repr__(self) -> str:
        return "Server()"
