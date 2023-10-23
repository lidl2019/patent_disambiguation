import torch
import pandas as pd
import random
from torch.utils.data import DataLoader,Dataset
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
from tqdm import tqdm

import helpers


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
    BASE_TRAIN_PATH = "data/newdata/base_train.csv"
    BASE_TEST_PATH = "data/newdata/base_test.csv"
def rinse_data(filepath: str):
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        if isinstance(row["encoded_title"], float) or isinstance(row["encoded_abstract"], float):
            df.drop(index, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(filepath, index=False)

from sklearn.model_selection import train_test_split
import os


def set_data_for_baseline(datapath, dataloader):

    title_similarity = []
    abstract_similarity = []
    date_similarity = []
    coinventor_in_common = []
    location_similarities = []
    labels = []
    for data in tqdm(dataloader):
        patent0, patent1, label = data
        date0, date1 = patent0[1], patent1[1]
        coinventor0, coinventor1 = patent0[2], patent1[2]
        lat0, lat1 = patent0[3], patent1[3]
        long0, long1 = patent0[4], patent1[4]
        title0, title1 = patent0[5], patent1[5]
        abstract0, abstract1 = patent0[6], patent1[6]

        for t0, t1 in zip(title0, title1):
            title_similarity.append(helpers.cosine_similarity(t0, t1))
        for a0, a1 in zip(abstract0, abstract1):
            abstract_similarity.append(helpers.cosine_similarity(a0, a1))

        for d0, d1 in zip(date0, date1):
            date_similarity.append(helpers.timestamp_similarity(d0, d1))

        for c0, c1 in zip(coinventor0, coinventor1):
            coinventor_in_common.append(helpers.coinventors_in_common(c0, c1))

        for la0, lon0, la1, lon1 in zip(lat0, long0, lat1, long1):
            # print(la0, lon0, la1, lon1)
            location_similarities.append(helpers.location_similarity(la0, lon0, la1, lon1))

        labels.extend(label.cpu().numpy())
    print(f"lengths: {len(title_similarity)},"
          f"{len(date_similarity)},"
          f"{len(coinventor_in_common)},"
          f"{len(location_similarities)},"
          f"{len(labels)}")
    df = pd.DataFrame({
        "title_similarity": title_similarity,
        "abstract_similarity": abstract_similarity,
        "date_similarity": date_similarity,
        "coinventor_in_common": coinventor_in_common,
        "location_similarities": location_similarities,
        "labels": labels
    })
    df.to_csv(datapath, index=False)
    print("saved!")