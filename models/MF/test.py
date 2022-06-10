from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import MFModel
"""
T
RESULT MF:
Density:0.05,type:rt,mae:0.576378887814029,mse:2.082243794355493,rmse:1.4429981962412473
Density:0.1,type:rt,mae:0.49478550883456834,mse:1.6236234720660405,rmse:1.274214845332623
Density:0.15,type:rt,mae:0.46929335905591363,mse:1.4799293112956513,rmse:1.216523452834203
Density:0.2,type:rt,mae:0.4395753397852491,mse:1.370772157241582,rmse:1.1707997938339338

Density:0.05,type:tp,mae:25.81130047175078,mse:4872.765566549135,rmse:69.80519727462372
Density:0.1,type:tp,mae:20.673126816311118,mse:3994.3217322699775,rmse:63.20064661275213
Density:0.15,type:tp,mae:17.115162743452927,mse:3001.432680489639,rmse:54.78533271314175
Density:0.2,type:tp,mae:15.728462584578748,mse:2351.604478290111,rmse:48.49334468037971



0609
# 0.01 01分布 定稿✅
MAE:0.62362,RMSE:1.53299
Density:0.1,type:rt,mae:0.5285582638721787,mse:1.7659958999885765,rmse:1.3289077846068087
Density:0.15,type:rt,mae:0.4878071698937158,mse:1.5336219791431043,rmse:1.238394920509247
MAE:0.46940,RMSE:1.20278

# 0 1 0.00005
MAE:26.47749,RMSE:67.46673
MAE:19.83326,RMSE:50.64019
MAE:16.84204,RMSE:47.48468
MAE:15.32499,RMSE:43.86034



# 0.0001
MAE:20.56002,RMSE:55.41779
MAE:17.36791,RMSE:48.43984
MAE:15.60340,RMSE:44.70647
MAE:14.72055,RMSE:42.44443


"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.2]:

    type_ = "tp"
    latent_dim = 8
    lr = 0.00005
    lambda_ = 0.1
    epochs = 400
    md_data = MatrixDataset(type_)
    train_data, test_data = md_data.split_train_test(density)

    mf = MFModel(md_data.row_n, md_data.col_n, latent_dim, lr, lambda_)
    mf.fit(train_data, test_data, epochs)
    y, y_pred = mf.predict(test_data)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
