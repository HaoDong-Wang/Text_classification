import time
import torch
import numpy as np
from utils import get_Logger
from importlib import import_module
import argparse

dataset = './datasets/CHIP-CTC'  # 数据集, 注：末尾不带 /
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='CNN', required=False, help='choose a model: Bert, ERNIE')
parser.add_argument('--save_path', type=str, default=dataset + '/data/out.npz', required=False, help='the save path of predictions on test set')

args = parser.parse_args()
print(args)

if __name__ == '__main__':

    model_name = args.model
    x = import_module('models.' + model_name)
    if model_name == 'TextCNN':
        # embedding = 'embedding_Tencent.npz'
        embedding = 'embedding_SougouNews.npz'
        config = x.Config(dataset, embedding)
    else:
        config = x.Config(dataset)
    logger = get_Logger(config, dataset)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name == 'TextCNN':
        from utils_TextCNN import build_dataset, build_iterator, get_time_dif
        vocab, train_data, dev_data, test_data = build_dataset(config, ues_word = False)   # help='True for word, False for char'
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        config.n_vocab = len(vocab)
    else:
        from utils import build_dataset, build_iterator, get_time_dif
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    logger.info(config)
    logger.info(args)
    model = x.Model(config).to(config.device)

    if model_name == 'TextCNN':
        from train_eval_TextCNN import train
        train(config, model, train_iter, dev_iter, test_iter, logger)
    else:
        from train_eval import train
        train(config, model, train_iter, dev_iter, test_iter, args, logger)
