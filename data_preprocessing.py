import torch
import pandas as pd
import random
from torch.utils.data import DataLoader,Dataset
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
from tqdm import tqdm
# print(torch.cuda.is_available())

class Config():
    PATH_TO_CHECKPT = "checkpoint/checkpoint.pth"
    EPOCH = 10
    LEARNING_RATE = 0.0001
    RAW_DATA_PATH = "data/newdata/filterd_with_coinventors_500000.csv"
    raw_data_path = "data/blocked/jaro_blocked.csv"
    renew_train_split = True
    train_data_path = "data/divided/train_set.csv"
    validate_data_path = "data/divided/validate_set.csv"
    test_data_path = "data/divided/test_set.csv"
    featured_data_export_path = "data/featured/featured.csv"
    model_weight_path = "./weights/model_weights.pth"
    test_percentage = 0.1
    validate_percentage = 0.1
    train_batch_size = 32
    train_number_epochs = 10
    num_workers = 0
    MODEL_2_TRAIN_PATH = "data/newdata/model2train.csv"
    MODEL_2_TEST_PATH = "data/newdata/model2test.csv"
def rinse_data(filepath: str):
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        if isinstance(row["encoded_title"], float) or isinstance(row["encoded_abstract"], float):
            df.drop(index, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(filepath, index=False)

from sklearn.model_selection import train_test_split
import os


