import collections
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch
from Patent_Dataset import PatentsDataset
import math
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def co_inventors_similarity(nlist1, nlist2):
    total_similarity = 0
    for str1 in nlist1:
        for str2 in nlist2:
            total_similarity += name_similarity(str1, str2)
    average_similarity = total_similarity / (len(nlist1) * len(nlist2))
    return 1 - average_similarity


def get_baseline_x_y(df):
    x = df[["title_similarity","abstract_similarity", "date_similarity", "coinventor_in_common", "location_similarities"]]
    y_ = df["labels"].tolist()
    y = []

    for r in y_:
        cur = int(float(r.strip("[]")))
        y.append(cur)

    return x, y
def coinventors_in_common(list1, list2):

    ans = 0

    for n in list1:
        if n != "NA" and n in list2:
            ans += 1

    return ans

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    a = math.sin(d_lat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def cosine_similarity(str1, str2):
    words1 = str1.split()
    words2 = str2.split()

    # Create word count vectors
    counter1 = collections.Counter(words1)
    counter2 = collections.Counter(words2)

    # Get the set of all words
    all_words = set(counter1.keys()).union(set(counter2.keys()))

    # Create vectors for both strings
    vector1 = [counter1[word] for word in all_words]
    vector2 = [counter2[word] for word in all_words]

    # Calculate the dot product and magnitude of both vectors
    dot_product = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
    magnitude1 = math.sqrt(sum([vector1[i] * vector1[i] for i in range(len(vector1))]))
    magnitude2 = math.sqrt(sum([vector2[i] * vector2[i] for i in range(len(vector2))]))

    # Calculate cosine similarity
    if magnitude1 * magnitude2 == 0:
        return 0  # To handle the case where one or both vectors are all zeros
    else:
        return dot_product / (magnitude1 * magnitude2)


def get_x_y(df):
    x = df[["euclidean_distance", "date_similarity", "coinventor_in_common", "location_similarities"]]
    y_ = df["labels"].tolist()
    y = []

    for r in y_:
        cur = int(float(r.strip("[]")))
        y.append(cur)

    return x, y

def location_similarity(lat1, lon1, lat2, lon2):
    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # The greatest possible haversine distance on Earth is half the circumference, roughly 20037.5 km.
    max_distance = 20037.5

    # Normalize the distance to get the similarity score.
    score = distance / max_distance

    return score

def timestamp_similarity(t1, t2):
    day1 = datetime.strptime(t1,'%Y-%m-%d')
    day2 = datetime.strptime(t2, '%Y-%m-%d')
    difference = abs((day1 - day2).total_seconds())

    decay_factor = 0.00000001
    similarity = np.exp(-decay_factor * difference)
    return 1 - similarity
def row_to_features(row):
    # name_tensor = torch.tensor([(ord(row['disambig_inventor_name_first'][0].lower()) - ord('a')), (ord(row['disambig_inventor_name_last'][0].lower()) - ord('a'))])
    title_tensor = torch.tensor(PatentsDataset.string_to_list(row["encoded_title"]))
    abstract_tensor = torch.tensor(PatentsDataset.string_to_list(row["encoded_abstract"]))
    # male_flag_tensor = torch.tensor([float(row["male_flag"])])
    pos = torch.tensor([(row["latitude"] + 90)/180 , (row["longitude"]+180) / 360])

    features = torch.cat((title_tensor, abstract_tensor), dim=0)
    features = torch.cat((features, pos), dim=0).to(torch.float32)
#     print(title_tensor)
    return features


def name_to_list(s: str):
    s = s.strip("[]")
    list_of_names = [i for i in s.split("'") if len(i) > 0 and "," not in i]
    return list_of_names

def name_similarity(n1, n2):
    return 1 - fuzz.ratio(n1, n2) / 100.0


def get_coinventors(df, map):
    coinventor_list = []
    for index, row in df.iterrows():
        cur_co_inventors = []
        cur_name = (row["disambig_inventor_name_first"] + " " if len(row["disambig_inventor_name_first"]) > 0
                    else "") + (row["disambig_inventor_name_last"] if len(row["disambig_inventor_name_last"]) > 0
                                else "")

        if len(map[row["patent_id"]]) == 1:
            cur_co_inventors.append("NA")
        else:
            for n in map[row["patent_id"]]:
                if n != cur_name:
                    cur_co_inventors.append(n)
        coinventor_list.append(str(cur_co_inventors))

    df["co_inventors"] = coinventor_list

    return df


def get_inventors_from_patent_id(df):
    unique_patent_id = set(df["patent_id"].tolist())
    patent_inventor_map = {}
    for id in unique_patent_id:
        patent_inventor_map[id] = []

    for index, row in df.iterrows():

        patent_inventor_map[row["patent_id"]].append((row["disambig_inventor_name_first"] + " "
                                                      if len(row["disambig_inventor_name_first"]) > 0 else "") +
                                                     (row["disambig_inventor_name_last"]
                                                      if len(row["disambig_inventor_name_last"]) > 0 else "")
                                                     )

    return patent_inventor_map


def generate_coinventor_col(df):
    map = get_inventors_from_patent_id(df)

    return get_coinventors(df, map)