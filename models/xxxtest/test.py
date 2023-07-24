from collections import namedtuple
from functools import partial

import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.XXXPlus.model import XXXPlus
from models.XXXPlus.resnet_utils import ResNetBasicBlock
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import SGD, Adam, optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decorator import timeit
from utils.evaluation import mae, mse, rmse
from utils.model_util import count_parameters, freeze_random

from .model import FedXXXLaunch, XXXPlusModel
"""

Fed:

2,2,2

[density:0.05,type:rt] Epoch:60 mae:0.39640194177627563,mse:1.8226033449172974,rmse:1.3500382900238037
[density:0.1,type:rt] Epoch:60 mae:0.36088812351226807,mse:1.638769268989563,rmse:1.280144214630127
[density:0.15,type:rt] Epoch:60 mae:0.34655630588531494,mse:1.5888218879699707,rmse:1.2604848146438599
[density:0.2,type:rt] Epoch:60 mae:0.3386975824832916,mse:1.5334124565124512,rmse:1.2383103370666504

[density:0.05,type:tp] Epoch:100 mae:16.623676300048828,mse:3133.034912109375,rmse:55.97351837158203
[density:0.1,type:tp] Epoch:100 mae:14.519242286682129,mse:2430.328369140625,rmse:49.29835891723633
[density:0.15,type:tp] Epoch:80 mae:13.745682716369629,mse:2111.67578125,rmse:45.952972412109375
[density:0.2,type:tp] Epoch:80 mae:13.631184577941895,mse:2176.6337890625,rmse:46.6544075012207

4 4 4 
[density:0.05,type:rt] Epoch:60 mae:0.39592623710632324,mse:1.803076982498169,rmse:1.3427870273590088
[density:0.1,type:rt] Epoch:60 mae:0.3592352271080017,mse:1.6179782152175903,rmse:1.2719976902008057
[density:0.15,type:rt] Epoch:80 mae:0.3353453576564789,mse:1.5067840814590454,rmse:1.2275112867355347
[density:0.2,type:rt] Epoch:80 mae:0.3254038989543915,mse:1.4776121377944946,rmse:1.2155706882476807

[density:0.05,type:tp] Epoch:80 mae:15.746493339538574,mse:2869.870849609375,rmse:53.57117462158203
[density:0.1,type:tp] Epoch:80 mae:13.736608505249023,mse:2279.198486328125,rmse:47.74095153808594
[density:0.15,type:tp] Epoch:80 mae:12.842589378356934,mse:1937.2783203125,rmse:44.014522552490234
[density:0.2,type:tp] Epoch:80 mae:12.18262767791748,mse:1775.5418701171875,rmse:42.13718032836914

8 8 8
[density:0.05,type:rt] Epoch:35 mae:0.3985772728919983,mse:1.797226905822754,rmse:1.340606927871704
[density:0.1,type:rt] Epoch:55 mae:0.35645025968551636,mse:1.5757331848144531,rmse:1.2552821636199951
[density:0.15,type:rt] Epoch:75 mae:0.3352337181568146,mse:1.4594119787216187,rmse:1.2080612182617188
[density:0.2,type:rt] Epoch:85 mae:0.31767740845680237,mse:1.4006195068359375,rmse:1.183477759361267

[density:0.05,type:tp] Epoch:90 mae:15.565387725830078,mse:2762.900146484375,rmse:52.563297271728516
[density:0.1,type:tp] Epoch:90 mae:12.96229362487793,mse:2074.105712890625,rmse:45.54235076904297
[density:0.15,type:tp] Epoch:90 mae:12.154128074645996,mse:1834.3294677734375,rmse:42.829071044921875
[density:0.2,type:tp] Epoch:90 mae:11.706123352050781,mse:1700.9140625,rmse:41.24213790893555


16 16 16
[density:0.05,type:rt] Epoch:65 mae:0.4096967577934265,mse:1.8146902322769165,rmse:1.3471044301986694
[density:0.1,type:rt] Epoch:75 mae:0.35850274562835693,mse:1.5438477993011475,rmse:1.2425167560577393
[density:0.15,type:rt] Epoch:75 mae:0.33630967140197754,mse:1.4385364055633545,rmse:1.1993900537490845
[density:0.2,type:rt] Epoch:65 mae:0.31893444061279297,mse:1.3725398778915405,rmse:1.171554446220398

[density:0.05,type:tp] Epoch:80 mae:15.342988967895508,mse:2699.822509765625,rmse:51.959815979003906
[density:0.1,type:tp] Epoch:80 mae:12.7587251663208,mse:2020.516845703125,rmse:44.95016098022461
[density:0.15,type:tp] Epoch:80 mae:11.635480880737305,mse:1686.2298583984375,rmse:41.06372833251953
[density:0.2,type:tp] Epoch:80 mae:11.280876159667969,mse:1615.932861328125,rmse:40.198665618896484

32 32 32
[density:0.05,type:tp] Epoch:160 mae:14.670263290405273,mse:2443.43310546875,rmse:49.431095123291016
[density:0.1,type:tp] Epoch:160 mae:13.012394905090332,mse:2018.0687255859375,rmse:44.92292022705078
[density:0.15,type:tp] Epoch:160 mae:11.701943397521973,mse:1701.3505859375,rmse:41.247432708740234
[density:0.2,type:tp] Epoch:160 mae:11.205389976501465,mse:1614.933349609375,rmse:40.18623352050781

[density:0.05,type:rt] Epoch:160 mae:0.40601715445518494,mse:1.773358702659607,rmse:1.3316751718521118
[density:0.1,type:rt] Epoch:160 mae:0.35660964250564575,mse:1.5247416496276855,rmse:1.2348042726516724
[density:0.15,type:rt] Epoch:160 mae:0.33663177490234375,mse:1.421412706375122,rmse:1.1922301054000854
[density:0.2,type:rt] Epoch:280 mae:0.3185778558254242,mse:1.3486533164978027,rmse:1.1613153219223022

64 64 64 
[density:0.05,type:tp] Epoch:200 mae:15.028202056884766,mse:2660.85107421875,rmse:51.583438873291016
[density:0.1,type:tp] Epoch:200 mae:12.76705265045166,mse:2014.603271484375,rmse:44.88433074951172
[density:0.15,type:tp] Epoch:200 mae:11.796093940734863,mse:1717.591796875,rmse:41.44384002685547
[density:0.2,type:tp] Epoch:200 mae:11.34628677368164,mse:1581.5189208984375,rmse:39.768314361572266

[density:0.05,type:rt] Epoch:200 mae:0.4031725823879242,mse:1.7362420558929443,rmse:1.3176653385162354
[density:0.1,type:rt] Epoch:200 mae:0.35359588265419006,mse:1.493826150894165,rmse:1.222221851348877
[density:0.15,type:rt] Epoch:315 mae:0.32597580552101135,mse:1.3775478601455688,rmse:1.173689842224121
[density:0.2,type:rt] Epoch:265 mae:0.3150278329849243,mse:1.3136166334152222,rmse:1.146131157875061

Non-Fed
[0.05_rt] Epoch:15 mae:0.37389495968818665,mse:1.7189104557037354,rmse:1.3110722303390503
[0.1_rt] Epoch:75 mae:0.32508647441864014,mse:1.5320173501968384,rmse:1.2377468347549438
[0.15_rt] Epoch:200 mae:0.30293840169906616,mse:1.4107856750488281,rmse:1.1877650022506714
[0.2_rt] Epoch:200 mae:0.28960633277893066,mse:1.3320720195770264,rmse:1.1541543006896973

[0.05_tp] Epoch:200 mae:13.811687469482422,mse:2294.42333984375,rmse:47.90013885498047
[0.1_tp] Epoch:200 mae:11.252021789550781,mse:1666.812255859375,rmse:40.82661056518555
[0.15_tp] Epoch:200 mae:10.421308517456055,mse:1445.4444580078125,rmse:38.01900100708008
[0.2_tp] Epoch:200 mae:10.136045455932617,mse:1378.9722900390625,rmse:37.134517669677734


0.1

[density:0.05,type:rt] Epoch:315 mae:0.37949123978614807,mse:1.7079064846038818,rmse:1.3068689107894897
[density:0.1,type:rt] Epoch:400 mae:0.3448878824710846,mse:1.5330452919006348,rmse:1.2381620407104492
[density:0.15,type:rt] Epoch:400 mae:0.3228624761104584,mse:1.4494836330413818,rmse:1.2039450407028198
[density:0.2,type:rt] Epoch:400 mae:0.3098350167274475,mse:1.3828232288360596,rmse:1.1759350299835205

[density:0.05,type:tp] Epoch:700 mae:14.945842742919922,mse:2664.945068359375,rmse:51.623104095458984
[density:0.1,type:tp] Epoch:700 mae:12.5775785446167,mse:1995.857421875,rmse:44.675018310546875
[density:0.15,type:tp] Epoch:700 mae:11.37606430053711,mse:1670.785888671875,rmse:40.875247955322266
[density:0.2,type:tp] Epoch:700 mae:11.064590454101562,mse:1568.4791259765625,rmse:39.60403060913086

0.3
14.95 51.35
12.51 44.34
11.69 41.74
11.24 39.73

0.3925 1.336
0.3567 1.257
0.3307 1.214
0.3186 1.185

0.5
[density:0.05,type:tp] Epoch:150 mae:14.698859214782715,mse:2498.902099609375,rmse:49.98902130126953

0.5
[density:0.05,type:tp] Epoch:150 mae:14.698859214782715,mse:2498.902099609375,rmse:49.98902130126953

[density:0.1,type:tp] Epoch:150 mae:12.415410995483398,mse:1921.6650390625,rmse:43.83679962158203

[density:0.15,type:tp] Epoch:150 mae:11.715423583984375,mse:1752.06103515625,rmse:41.857627868652344

[density:0.2,type:tp] Epoch:150 mae:10.96554946899414,mse:1530.513427734375,rmse:39.12177658081055

[density:0.05,type:rt] Epoch:175 mae:0.3919405937194824,mse:1.7514203786849976,rmse:1.3234124183654785

[density:0.1,type:rt] Epoch:175 mae:0.35684964060783386,mse:1.5232750177383423,rmse:1.2342102527618408

[density:0.15,type:rt] Epoch:175 mae:0.32604047656059265,mse:1.4150018692016602,rmse:1.1895384788513184

[density:0.2,type:rt] Epoch:150 mae:0.3133954405784607,mse:1.3681621551513672,rmse:1.169684648513794



1.0

[density:0.05,type:rt] Epoch:65 mae:0.4096967577934265,mse:1.8146902322769165,rmse:1.3471044301986694
[density:0.1,type:rt] Epoch:75 mae:0.35850274562835693,mse:1.5438477993011475,rmse:1.2425167560577393
[density:0.15,type:rt] Epoch:75 mae:0.33630967140197754,mse:1.4385364055633545,rmse:1.1993900537490845
[density:0.2,type:rt] Epoch:65 mae:0.31893444061279297,mse:1.3725398778915405,rmse:1.171554446220398

[density:0.05,type:tp] Epoch:80 mae:15.342988967895508,mse:2699.822509765625,rmse:51.959815979003906
[density:0.1,type:tp] Epoch:80 mae:12.7587251663208,mse:2020.516845703125,rmse:44.95016098022461
[density:0.15,type:tp] Epoch:80 mae:11.635480880737305,mse:1686.2298583984375,rmse:41.06372833251953
[density:0.2,type:tp] Epoch:80 mae:11.280876159667969,mse:1615.932861328125,rmse:40.198665618896484


Fed-Non-P

[density:0.05,type:rt] Epoch:180 mae:0.37927231192588806,mse:1.7944732904434204,rmse:1.339579463005066

[density:0.1,type:rt] Epoch:180 mae:0.35481879115104675,mse:1.6555014848709106,rmse:1.2866629362106323

[density:0.15,type:rt] Epoch:180 mae:0.34564530849456787,mse:1.6050324440002441,rmse:1.26689875125885

[density:0.2,type:rt] Epoch:175 mae:0.3376237154006958,mse:1.559515118598938,rmse:1.2488055229187012

[density:0.05,type:tp] Epoch:180 mae:15.02529239654541,mse:2450.156494140625,rmse:49.499053955078125

[density:0.1,type:tp] Epoch:340 mae:13.810931205749512,mse:2036.3375244140625,rmse:45.125797271728516

[density:0.15,type:tp] Epoch:340 mae:13.484705924987793,mse:1969.3216552734375,rmse:44.37704086303711

[density:0.2,type:tp] Epoch:265 mae:13.477914810180664,mse:1834.0618896484375,rmse:42.825950622558594


去掉 dropout mae:0.3924892544746399,mse:1.7166827917099,rmse:1.3102223873138428 Epoch:215 mae:15.217144966125488,mse:2527.579345703125,rmse:51.8250358581543
使用 BN mae:0.42298200726509094,mse:1.896214485168457,rmse:1.37703108787536     Epoch 145 mae 18.5340 RMSE 57.9141


"""

config = {
    "CUDA_VISIBLE_DEVICES": "0",
    "embedding_dims": [16, 16, 16],
    "density": 0.05,
    "type_": "rt",
    "epoch": 4000,
    "is_fed": False,
    "train_batch_size": 256,
    "lr": 0.001,
    "in_size": 16 * 6,
    "out_size": None,
    "blocks": [1024,512,256, 128, 64],
    "deepths": [1, 1, 1,1],
    "linear_layer": [1088, 64],
    "weight_decay": 0,
    "loss_fn": nn.L1Loss(),
    "is_personalized": True,
    "activation": nn.ReLU,
    "select": 0.3,
    "local_epoch": 5,
    "fed_bs": -1
    # "备注":"embedding初始化参数0,001"
}

epochs = config["epoch"]
density = config["density"]
type_ = config["type_"]

is_fed = config["is_fed"]
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]


def data_preprocess(triad,
                    u_info_obj: InfoDataset,
                    i_info_obj: InfoDataset,
                    is_dtriad=False):
    """解决uid和embedding时的id不一致的问题
    生成d_triad [[triad],[p_triad]]
    """
    r = []
    for row in tqdm(triad, desc="Gen d_triad", ncols=80):
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([[uid, iid, rate], [u, i, rate]]) if is_dtriad else r.append(
            [u, i, rate])
    return r


u_enable_columns = ["[User ID]", "[Country]", "[AS]"]
i_enable_columns = ["[Service ID]", "[Country]", "[AS]"]

fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

md = MatrixDataset(type_)
u_info = InfoDataset("user", u_enable_columns)
i_info = InfoDataset("service", i_enable_columns)
train, test = md.split_train_test(density)

# loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.L1Loss()
loss_fn = config["loss_fn"]

activation = config["activation"]
# activation = nn.ReLU

user_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": config["embedding_dims"],
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": config["embedding_dims"],
}

if config["is_fed"]:

    fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

    train_data = fed_data_preprocess(train, u_info, i_info)
    test_data = fed_data_preprocess(test, u_info, i_info)

    params = {
        "user_embedding_params": user_params,
        "item_embedding_params": item_params,
        "in_size": config["in_size"],
        "output_size": config["out_size"],
        "blocks_size": config["blocks"],
        "batch_size": config["fed_bs"],
        "deepths": config["deepths"],
        "activation": activation,
        "d_triad": train_data,
        "test_d_triad": test_data,
        "loss_fn": config["loss_fn"],
        "local_epoch": config["local_epoch"],
        "linear_layers": config["linear_layer"],
        "is_personalized": config["is_personalized"],
        "header_epoch": None,
        "personal_layer": "my_layer",
        "output_dim": 1,
        "optimizer": "adam",
        "use_gpu": True
    }

    model = FedXXXLaunch(**params)
    print(f"模型参数:", count_parameters(model))
    print(model)
    print(config)
    model.fit(epochs, config["lr"], 5, config["select"],
              f"density:{density},type:{type_}")

else:

    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["train_batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=2048)

    model = XXXPlusModel(user_params, item_params, config["in_size"],
                         config["out_size"], config["blocks"],
                         config["deepths"], loss_fn, activation,
                         config["linear_layer"])
    print(f"模型参数:", count_parameters(model))
    print(model)
    print(config)
    opt = Adam(model.parameters(),
               lr=config["lr"],
               weight_decay=config["weight_decay"])
    # opt = SGD(model.parameters(), lr=0.01)

    model.fit(train_dataloader,
              epochs,
              opt,
              eval_loader=test_dataloader,
              save_filename=f"{density}_{type_}")
