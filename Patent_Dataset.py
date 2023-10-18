import torch
import pandas as pd
import random
from torch.utils.data import Dataset
import helpers

class PatentsDataset(Dataset):
    def __init__(self, training_data_path=None, transform=None):
        self.df = pd.read_csv(training_data_path)
        self.transform = transform
        self.data = {}
        for index, row in self.df.iterrows():
            if row["inventor_id"] not in self.data.keys():
                self.data[row["inventor_id"]] = []
            #             self.data[row["inventor_id"]].append(self.row_id_to_features(index))
            # key:  inventor_id, value: patent information (encoded title and abstract, date, co-inventors)
            self.data[row["inventor_id"]].append([self.row_id_to_features(index),
                                                  row["patent_date"],
                                                  row["co_inventors"],
                                                  row["latitude"], row["longitude"]
                                                  ])

    #         self.fullset = {}

    def __len__(self):
        return len(self.df)

    def string_to_list(s: str):
        try:
            if s == None:
                print(s)
            s = s.strip("[]")  # remove brackets
            list_of_ints = [float(i[:-1]) for i in s.split()]
            return list_of_ints
        except Exception as e:
            print(s)
            print(type(s))
            print(e)
            raise e

    def row_id_to_features(self, row_id):
        row = self.df.iloc[row_id]
        return helpers.row_to_features(row)

    def rand_inventor_id(self):
        inventor_id = random.choice(list(self.data.keys()))
        return inventor_id

    def rand_patent_from_inventor(self, inventor_id):
        return random.choice(self.data[inventor_id])

    def visualize_selector(self, label_count=5, label_min_datapoint=10):
        patent_list = []
        label_list = []
        _cnt = 0
        picked_list = []
        while _cnt < label_count:
            rand_id = self.rand_inventor_id()
            if rand_id in picked_list:
                continue
            else:
                picked_list.append(rand_id)
            if len(self.data[rand_id]) >= label_min_datapoint:
                for patent in self.data[rand_id]:
                    patent_list.append(patent)
                    label_list.append(rand_id)
                _cnt += 1
        return patent_list, label_list

    def __getitem__(self, idx):
        anchor_inventor_id = self.rand_inventor_id()
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while len(self.data[anchor_inventor_id]) < 2:
                anchor_inventor_id = self.rand_inventor_id()
                # if len(self.data[anchor_inventor_id]) >= 2:
                #     print(anchor_inventor_id, len(self.data[anchor_inventor_id]))
            anchor, other = random.sample(self.data[anchor_inventor_id], 2)
        else:
            anchor = self.rand_patent_from_inventor(anchor_inventor_id)
            other_inventor_id = anchor_inventor_id
            while anchor_inventor_id == other_inventor_id:
                other_inventor_id = self.rand_inventor_id()
            other = self.rand_patent_from_inventor(anchor_inventor_id)
        if self.transform is not None:
            anchor = self.transform(anchor)
            other = self.transform(other)
        return anchor, other, torch.tensor([should_get_same_class], dtype=torch.float32)