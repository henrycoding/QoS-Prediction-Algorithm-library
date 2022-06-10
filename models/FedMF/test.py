from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from . import Clients, Server
from .model import FedMF

from utils.mylogger import TNLog

"""
RESULT FedMF: 
1000epoch
Density:0.05,type:rt,mae:0.5762619416345588,mse:2.083359291985126,rmse:1.4433846652868132
Density:0.1,type:rt,mae:0.5065232463307372,mse:1.6816029233716594,rmse:1.2967663333737731
Density:0.15,type:rt,mae:0.466420129810779,mse:1.4792783323010215,rmse:1.216255866296653
Density:0.2,type:rt,mae:0.43765304163567553,mse:1.3690546770840173,rmse:1.1700660994508034

lr 0.00005
Density:0.05,type:tp,mae:20.869833357513084,mse:3146.555328964253,rmse:56.09416483881593
Density:0.05,type:tp,mae:17.21535219997346,mse:2648.935200852116,rmse:51.46780742223352
Density:0.05,type:tp,mae:16.1133831087054,mse:2148.759199202129,rmse:46.354710647377885
Density:0.05,type:tp,mae:15.53844195763405,mse:2002.1836451574404,rmse:44.74576678477463

# 1 FedMF 优化后的结果


# 2 FedMF TP结果


0609
# 0 1 分布 0.01
Epoch:200 mae:0.6224332896232857,mse:2.3293892388569204,rmse:1.5262336776709262
Epoch:280 mae:0.5288375158000332,mse:1.759309544370584,rmse:1.3263896653587828
Epoch:60 mae:0.48850577621279484,mse:1.5306377615651567,rmse:1.2371894606587774
Epoch:180 mae:0.4702070464979563,mse:1.4428694039481238,rmse:1.2011949899779486


0 1 分布 0.00005
Epoch:135 mae:26.923630337021407,mse:4742.9312347693785,rmse:68.86894245426873
Epoch:95 mae:19.62376678297121,mse:2848.608981783482,rmse:53.37236159084102
mae:16.90701452800244,mse:2371.6316753889405,rmse:48.69940118100982
mae:15.340056435947972,mse:1924.6235004505058,rmse:43.87053111657649



# 0 0.1 分布 0.00005
Epoch:40 mae:21.611827532166032,mse:3306.3895273089397,rmse:57.50121326814713
mae:19.213533781972753,mse:2791.5082232960403,rmse:52.83472554386973
Epoch:40 mae:16.742236888148085,mse:2197.8994402856133,rmse:46.88176020890868
Epoch:60 mae:15.561386874798426,mse:2043.0222692806233,rmse:45.19980386329816


"""

# logger = TNLog('FedMF')
# logger.initial_logger()

# freeze_random()  # 冻结随机数 保证结果一致

for density in [0.2]:

    # 1
    type_ = "rt"
    latent_dim = 8
    lr = 0.01
    lambda_ = 0.1
    epochs = 2000

    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)
    clients = Clients(train_data, md_data.row_n, latent_dim)

    server = Server(md_data.col_n, latent_dim)

    mf = FedMF(server, clients)
    mf.fit(epochs, lambda_, lr, test_data)
    y, y_pred = mf.predict(test_data, False)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    mf.logger.critical(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
