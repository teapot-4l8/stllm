"""
只test训练好的模型
"""

import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model_ST_LLM_plus import ST_LLM
from ranger21 import Ranger
import matplotlib.pyplot as plt

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:180'

"""
python .\train_plus.py --data volume
"""

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="volume", help="data path")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lrate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=300, help="500")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--num_nodes", type=int, default=250, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=6, help="out_len")
parser.add_argument("--llm_layer", type=int, default=1, help="llm layer")
parser.add_argument("--U", type=int, default=1, help="unforzen layer")  # 在训练过程中允许其权重更新的层
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--wdecay", type=float, default=0.0001, help="weight decay rate"
    )
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "-",
    help="save path"
    )
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations"
    )
args = parser.parse_args()
adj_mx = load_graph_data(f"data/{args.data}/adj_mx.pkl") # nyc  (250, 250)   evdata  (275, 275)

class trainer:
    def __init__(
        self,
        scaler,
        adj_mx,
        input_dim,
        num_nodes,
        input_len,
        output_len,
        llm_layer, 
        U,
        lrate,
        wdecay,
        device
    ):
        self.model = ST_LLM(
            device, adj_mx, input_dim, num_nodes, input_len, output_len, llm_layer, U
        )
        self.model.to(device)
        # 初始化模型的优化器
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5  # 用于梯度裁剪（Gradient Clipping），防止梯度爆炸
        print("The number of parameters: {}".format(self.model.param_num()))  # The number of parameters: 47209740 模型的总参数数量
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))  # The number of trainable parameters: 3101964
        print(self.model)

    def train(self, input, real_val):  #  metrics = engine.train(trainx, trainy[:, 0, :, :])  real_val torch.Size([64, 250, 12])
        self.model.train()
        self.optimizer.zero_grad()  # Clears the gradients of all model parameters.
        output = self.model(input)  # output = self.model.forward(input)
        output = output.transpose(1, 3)  # torch.Size([64, 1, 250, 12])  
        real = torch.unsqueeze(real_val, dim=1)  # torch.Size([64, 1, 250, 12])
        predict = self.scaler.inverse_transform(output)  #torch.Size([64, 1, 250, 12]) This line takes your model's predictions (which are in a normalized or standardized range) and converts them back to the real-world scale
        loss = self.loss(predict, real, 0.0)
        loss.backward()  # This line tells PyTorch to “work backwards from the loss and figure out how each parameter contributed to it”—so it can be improved!
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()  # This line actually changes your model’s parameters, making it learn from the data!
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)


def plot(pred, real, i, zone, name):
    plt.figure(figsize=(12, 6))  # Always start with a new figure
    plt.plot(real, label='Actual Values', color='blue', linewidth=2)
    plt.plot(pred, label='Predicted Values', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.title(f'Zone {zone} Prediction vs Actual (Horizon {i+1})')
    plt.tight_layout()
    # plt.savefig(f'./csv_data/{name}_{i+1}_{zone}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free memory and prevent legend stacking


def save_to_csv(pred, real, i, zone, name):
    df = pd.DataFrame({
        "Actual": real.ravel(),
        "Predicted": pred.ravel(),
    })
    # df.to_csv(f"./csv_data/{name}_prelen{i+1}_zone_{zone}_811.csv", index=False)

def main():
    seed_it(6666)
    data = args.data
    args.data = "data//" + args.data
    args.num_nodes = 275

    device = torch.device(args.device)
    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"  # './logs/2025-05-31-22-29-31-bike_drop/'

    result = []
    test_result = []
    print(args)


    engine = trainer(
        scaler,
        adj_mx,
        args.input_dim,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.llm_layer,
        args.U,
        args.lrate,
        args.wdecay,
        device
    )

    model_path = r"D:\_________________________PythonProject\ST-LLM-Plus-main\logs\2025-06-05-19-23-06-volume\best_model.pth"
    engine.model.load_state_dict(torch.load(model_path))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)  # torch.Size([432, 6, 275, 3])
    realy = realy.transpose(1, 3)[:, 0, :, :]  # torch.Size([432, 275, 6])

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device) 
        testx = testx.transpose(1, 3)  # torch.Size([64, 3, 275, 12])
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)  # torch.Size([64, 1, 275, 6])
        outputs.append(preds.squeeze())  # torch.Size([64, 275, 6])

    yhat = torch.cat(outputs, dim=0)  # torch.Size([448, 275, 6]) TODO 
    yhat = yhat[: realy.size(0), ...]  # [651, 275, 6] TODO 651是什么
    print(yhat.shape)

    amae = []
    amape = []
    armse = []
    awmape = []

    test_m = []

    # 保存真实值对比 FIXME
    # df = pd.DataFrame(yhat[:, :, 0].cpu().numpy())
    # df.to_csv("./csv_data/all_real.csv")

    for i in range(args.output_len):
        # i = 几 只用前几个小时来预测将来output_len小时
        i = 5  # 这里是5，其实是将来6小时 i+1
        pred = scaler.inverse_transform(yhat[:, :, i])  # torch.Size([651, 275])
        real = realy[:, :, i]
        
        # metrics = util.metric(pred, real)  # torch.Size([651, 275])
        # log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        # print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        # TODO 计算所有zone 选一个最好的 MAPE  然后用这个zone测chatev
        zone = 42
        pred_single_zone = pred[:, zone:zone+1]
        real_single_zone = real[:, zone:zone+1] 
        plot(pred_single_zone.cpu().numpy(), real_single_zone.cpu().numpy(), i, zone, data)
        metrics = util.metric(pred_single_zone, real_single_zone)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        save_to_csv(pred_single_zone.cpu().numpy(), real_single_zone.cpu().numpy(), i, zone, data)

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

        break 

    log = "On average over 6 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    # test_csv.round(8).to_csv(f"{path}/zone42horizon6_811.csv") 

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
