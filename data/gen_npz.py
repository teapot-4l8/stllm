import argparse
import numpy as np
import os
import pandas as pd
import h5py


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape[0], df.shape[1]
    data_list = [df]

    if add_time_in_day:
        df.index = pd.to_datetime(df.index,format = '%Y-%m-%d %H:%M:%S')  # 明确指定时间字符串的格式为 年-月-日 小时:分钟:秒
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)  # (4368, 250, 2) add false
    print(data.shape)  # (4368, 250, 2)

    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive  确保在生成训练样本时，不会超出数据的时间范围
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)  # (4345, 12, 250, 2)
    y = np.stack(y, axis=0)  # (4345, 12, 250, 2)
    return x, y


def generate_train_val_test(args):
    # h5_path = os.path.join('h5data',args.h5_name+'.h5')
    # df = h5py.File(h5_path, 'r')
    df = h5py.File(r"D:\_________________________PythonProject\ST-LLM-Plus-main\data\h5data\nyc-bike.h5", 'r')
    rawdata = []
    for feature in ["pick", "drop"]:
        key = args.h5_name[-4:] + "_" + feature
        data = np.array(df[key])  # (4368, 250)  Number of Timesteps 4,368
        rawdata.append(data)
    rawdata = np.stack(rawdata, -1)  # (4368, 250, 2)

    x_offsets = np.sort(
        np.concatenate((np.arange(-(args.window-1), 1, 1),))
    )  # array([-11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   0])
   
    y_offsets = np.sort(np.arange(1, args.horizon+1, 1))  # array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

    x, y = generate_graph_seq2seq_io_data(
        rawdata,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim) 
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = 672 
    num_val = 672
    num_train = num_samples - num_test - num_val  

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    save_folder =  os.path.join(args.h5_name)
    os.mkdir(save_folder)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # train x:  (3001, 12, 250, 2) y: (3001, 12, 250, 2)
        print(cat, "x: ", _x.shape, "y:", _y.shape)       
        np.savez_compressed(
            os.path.join(save_folder, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_name",
        type=str,
        default="nyc_bike",
        help="Raw data",
    )
    parser.add_argument(
        "--window", type=int, default=12, help="the length of history seq"
    )
    parser.add_argument(
        "--horizon", type=int, default=12, help="the length of predict seq"
    )
    
    args = parser.parse_args()
    main(args)