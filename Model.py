import pandas as pd
import torch.cuda
from tqdm import tqdm
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import helpers
from Configuration import Config
from Patent_Network import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Encoder_Model():
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)

    def encode(self, text):
        return [round(i, 4) for i in self.model.encode(text)]

    def encode_to_str(self, text):
        return str(self.encode(text))

class Baseline():

    def __init__(self, train_path, test_path):
        self.model = LogisticRegression()
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
    def fit(self, remove_duplicates = False):
        if remove_duplicates:
            self.train_df.drop_duplicates(inplace=True)

        x_train, y_train = helpers.get_baseline_x_y(self.train_df)
        self.model.fit(x_train, y_train)

    def predict(self, remove_duplicates = False):
        if remove_duplicates:
            self.test_df.drop_duplicates(inplace=True)
        x_test, y_test = helpers.get_baseline_x_y(self.test_df)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Baseline Accuracy: {accuracy * 100:.2f}%")

class Model():
    def __init__(self, traindataloader, testdataloader, validationdataloader=None):
        self.optimal_threshold = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = traindataloader
        self.test_loader = testdataloader
        self.valid_loader = validationdataloader
        self.criterion = ContrastiveLoss(margin=5)
        self.valid = True if validationdataloader else False

        self.dim = -1
        for data in self.train_loader:
            dim1, dim2, targets = data

            self.dim = len(dim1[0][0])
            print(f"input dimension = {self.dim}")
            break
        self.model = PatentNetwork(self.dim)
    def save_check_point(self, model, optimizer, filename=Config.PATH_TO_CHECKPT):
        print(f"saving weights to {filename}")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, model, optimizer, filename=Config.PATH_TO_CHECKPT):
        print(f"loading weights from {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def prepare_visualization_dataset(self, datapath, train = True):

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.load_checkpoint(self.model, self.optimizer, Config.PATH_TO_CHECKPT)

        dataloader = self.train_loader if train else self.test_loader
        self.model.eval()
        self.model.to(self.device)
        euclidean_distance = []
        date_similarity = []
        coinventor_in_common = []
        location_similarities = []
        labels = []
        name_1 = []
        name_2 = []
        inventor_id1 = []
        inventor_id2 = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                patent0, patent1, label = data
                date0, date1 = patent0[1], patent1[1]
                coinventor0, coinventor1 = patent0[2], patent1[2]
                lat0, lat1 = patent0[3], patent1[3]
                long0, long1 = patent0[4], patent1[4]
                id1, id2 = patent0[5], patent1[5]
                name1, name2 = patent0[6], patent1[6]

                # print(latitude0[0], longitude0[0], latitude1[0], longitude1[0])

                # print(len(location0), len(location0[0]))
                coinventor0 = [helpers.name_to_list(i) for i in coinventor0]
                coinventor1 = [helpers.name_to_list(i) for i in coinventor1]

                if torch.cuda.is_available():
                    patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                else:
                    patent0, patent1, label = patent0[0], patent1[0], label

                output1 = self.model(patent0)
                output2 = self.model(patent1)


                euclidean_dist = nn.functional.pairwise_distance(output1, output2)
                euclidean_distance.extend(euclidean_dist.cpu().numpy())

                labels.extend(label.cpu().numpy())

                for d0, d1 in zip(date0, date1):
                    date_similarity.append(helpers.timestamp_similarity(d0, d1))

                for c0, c1 in zip(coinventor0, coinventor1):
                    coinventor_in_common.append(helpers.coinventors_in_common(c0,c1))

                for la0, lon0, la1, lon1 in zip(lat0, long0, lat1, long1):
                    # print(la0, lon0, la1, lon1)
                    location_similarities.append(helpers.location_similarity(la0, lon0, la1, lon1))

                for n in name1:
                    name_1.append(n)

                for n in name2:
                    name_2.append(n)

                for i in id1:
                    inventor_id1.append(i)

                for i in id2:
                    inventor_id2.append(i)
        df = pd.DataFrame({
            "euclidean_distance": euclidean_distance,
            "date_similarity": date_similarity,
            "coinventor_in_common": coinventor_in_common,
            "location_similarities": location_similarities,
            "labels": labels,
            "name1": name_1,
            "name2": name_2,
            "inventor_id1": inventor_id1,
            "inventor_id2": inventor_id2
        })
        df.to_csv(datapath, index=False)
        print("saved!")




    def set_data_for_pairwise(self, datapath, train = True):

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.load_checkpoint(self.model, self.optimizer, Config.PATH_TO_CHECKPT)

        dataloader = self.train_loader if train else self.test_loader
        self.model.eval()
        self.model.to(self.device)

        euclidean_distance = []
        date_similarity = []
        coinventor_in_common = []
        location_similarities = []
        labels = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                patent0, patent1, label = data
                date0, date1 = patent0[1], patent1[1]
                coinventor0, coinventor1 = patent0[2], patent1[2]
                lat0, lat1 = patent0[3], patent1[3]
                long0, long1 = patent0[4], patent1[4]

                # print(latitude0[0], longitude0[0], latitude1[0], longitude1[0])

                # print(len(location0), len(location0[0]))
                coinventor0 = [helpers.name_to_list(i) for i in coinventor0]
                coinventor1 = [helpers.name_to_list(i) for i in coinventor1]

                if torch.cuda.is_available():
                    patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                else:
                    patent0, patent1, label = patent0[0], patent1[0], label

                output1 = self.model(patent0)
                output2 = self.model(patent1)
                euclidean_dist = nn.functional.pairwise_distance(output1, output2)
                euclidean_distance.extend(euclidean_dist.cpu().numpy())
                labels.extend(label.cpu().numpy())

                for d0, d1 in zip(date0, date1):
                    date_similarity.append(helpers.timestamp_similarity(d0, d1))

                for c0, c1 in zip(coinventor0, coinventor1):
                    coinventor_in_common.append(helpers.coinventors_in_common(c0,c1))

                for la0, lon0, la1, lon1 in zip(lat0, long0, lat1, long1):
                    # print(la0, lon0, la1, lon1)
                    location_similarities.append(helpers.location_similarity(la0, lon0, la1, lon1))
                    # print(helpers.location_similarity(la0,lon0,la1,lon1))
        print(f"lengths: {len(euclidean_distance)},"
              f"{len(date_similarity)},"
              f"{len(coinventor_in_common)},"
              f"{len(location_similarities)},"
              f"{len(labels)}")
        df = pd.DataFrame({
            "euclidean_distance":euclidean_distance,
            "date_similarity":date_similarity,
            "coinventor_in_common":coinventor_in_common,
            "location_similarities":location_similarities,
            "labels":labels
        })
        df.to_csv(datapath, index = False)
        print("saved!")




    def train(self, epoch=Config.EPOCH, filename=Config.PATH_TO_CHECKPT, from_pretrain=False):
        counter = []
        loss_history = []

        valid_history = []
        iteration_number = 0.0
        device = self.device
        self.model.to(device)
        best_accuracy = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        count = 0
        if from_pretrain:
            self.load_checkpoint(self.model, self.optimizer, filename)

        for e in range(epoch):

            with tqdm(
                    iterable=self.train_loader,
                    bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',
            ) as t:
                self.model.train()
                for data in self.train_loader:
                    patent0, patent1, label = data
                    if torch.cuda.is_available():
                        patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                    else:
                        patent0, patent1, label = patent0[0], patent1[0], label

                    self.optimizer.zero_grad()
                    # print(patent0.shape)
                    # print(patent1.shape)
                    output1 = self.model(patent0)
                    output2 = self.model(patent1)

                    loss_contrastive = self.criterion(output1, output2, label)
                    loss_contrastive.backward()
                    self.optimizer.step()
                    iteration_number += 1
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                    t.set_postfix_str(f"train_loss={sum(loss_history) / len(loss_history):.6f}")
                    t.update()

                if self.valid:
                    self.model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in self.valid_loader:
                            patent0, patent1, label = data
                            if torch.cuda.is_available():
                                patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                            else:
                                patent0, patent1, label = patent0[0], patent1[0], label

                            output1 = self.model(patent0)
                            output2 = self.model(patent1)
                            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                            predicted = (euclidean_distance > 2.5).float()  # Prediction logic reversed
                            correct += (
                                    predicted.squeeze() == label.squeeze()).sum().item()  # Ensure label and predicted have the same dimensions
                            total += label.size(0)
                        accuracy = 100*correct/total
                        valid_history.append(accuracy)
                        print('Test Accuracy with threashold {:.4f}: {:.2f}%'.format(2.5, accuracy))
                        if accuracy >= best_accuracy:
                            best_accuracy = accuracy
                            self.save_check_point(model=self.model, optimizer=self.optimizer, filename=filename)



        # self.save_check_point(model=self.model, optimizer=self.optimizer, filename=filename)
        plt.plot(counter, loss_history)
        plt.savefig("training_result.png", dpi=300)

    def evaluate(self, threshold, from_pretrain = True, filename = Config.PATH_TO_CHECKPT):
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        if from_pretrain:
            self.load_checkpoint(self.model, self.optimizer, filename=filename)

        self.model.eval()
        self.model.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                patent0, patent1, label = data
                if torch.cuda.is_available():
                    patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                else:
                    patent0, patent1, label = patent0[0], patent1[0], label

                output1 = self.model(patent0)
                output2 = self.model(patent1)
                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predicted = (euclidean_distance > threshold).float()  # Prediction logic reversed
                correct += (
                            predicted.squeeze() == label.squeeze()).sum().item()  # Ensure label and predicted have the same dimensions
                total += label.size(0)

        print('Test Accuracy with threashold {:.4f}: {:.2f}%'.format(threshold, 100 * correct / total))
        return correct / total

    def test_ROC_Curve(self, from_pretrain = True, filename = Config.PATH_TO_CHECKPT):
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        if from_pretrain:
            self.load_checkpoint(self.model, self.optimizer, filename=filename)
        self.model.eval()

        self.model.to(self.device)
        true_labels = []
        predicted_distance = []
        with torch.no_grad():
            for data in self.test_loader:
                patent0, patent1, label = data

                if torch.cuda.is_available():
                    patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                else:
                    patent0, patent1, label = patent0[0], patent1[0], label

                output1 = self.model(patent0)
                output2 = self.model(patent1)
                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predicted_distance.extend(euclidean_distance.cpu().numpy())
                true_labels.extend(label.cpu().numpy())

        true_labels = np.array(true_labels)
        predicted_distance = np.array(predicted_distance)

        fpr, tpr, thresholds = roc_curve(true_labels, predicted_distance, pos_label=1)

        # Calculate the AUC (Area under the ROC Curve)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("test_ROC_curve.png", dpi=300)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Optimal threshold: ", optimal_threshold)
        self.optimal_threshold = optimal_threshold

        self.evaluate(optimal_threshold)
        return optimal_threshold

    def get_vector_cluster(self, threshold = 2.5):
        t = self.optimal_threshold if self.optimal_threshold != -1 else threshold

        pass


    def test_threshold_curve(self, thresholds, from_pretrain = True, filename = Config.PATH_TO_CHECKPT):
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        if from_pretrain:
            self.load_checkpoint(self.model, self.optimizer, filename=filename)

        self.model.to(self.device)
        accuracies = []
        self.model.eval()
        for threshold in thresholds:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.test_loader:
                    patent0, patent1, label = data
                    p_dates0, p_dates1 = patent0[1], patent1[1]
                    if torch.cuda.is_available():
                        patent0, patent1, label = patent0[0].cuda(), patent1[0].cuda(), label.cuda()
                    else:
                        patent0, patent1, label = patent0[0], patent1[0], label

                    output1 = self.model(patent0)
                    output2 = self.model(patent1)
                    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                    predicted = (euclidean_distance > threshold).float()
                    correct += (predicted.squeeze() == label.squeeze()).sum().item()
                    total += label.size(0)
            accuracy = correct / total
            accuracies.append(accuracy)
        plt.plot(thresholds, accuracies)
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for different thresholds')
        plt.savefig("test_threshold_curve.png", dpi = 300)

        thresholds_np = np.array(thresholds)
        accuracies_np = np.array(accuracies)
        max_accuracy_idx = np.argmax(accuracies_np)
        optimal_threshold = thresholds_np[max_accuracy_idx]
        print("Optimal threshold: ", optimal_threshold)
        self.evaluate(optimal_threshold)
        return optimal_threshold

