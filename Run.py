import torch.cuda

from data_preprocessing import *
from helpers import *
from Model import *
from Patent_Dataset import *

def preprocess_data():
    if Config.renew_train_split or ( not os.path.exists(Config.train_data_path) or not os.path.exists(Config.test_data_path) or not os.path.exists(Config.validate_data_path) ):
    #     raw_df = pd.read_csv(Config.raw_data_path)
        raw_df = pd.read_csv(Config.RAW_DATA_PATH)
        train, test = train_test_split(raw_df, test_size=Config.test_percentage, random_state=42)
        train, validate = train_test_split(train, test_size=Config.validate_percentage, random_state=42)
        train.to_csv(Config.train_data_path, index=False)
        validate.to_csv(Config.validate_data_path, index=False)
        test.to_csv(Config.test_data_path, index=False)
        print(len(raw_df))
        print(len(train))
        print(len(validate))
        print(len(test))

    if Config.renew_train_split:
        rinse_data(Config.train_data_path)
        rinse_data(Config.validate_data_path)
        rinse_data(Config.test_data_path)
    # df = pd.read_csv(Config.RAW_DATA_PATH)





if __name__ == '__main__':

    print(f"using GPU: {torch.cuda.is_available()}")
    # preprocess_data()
    train_dataset = PatentsDataset(Config.train_data_path)
    test_dataset = PatentsDataset(Config.test_data_path)

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=Config.num_workers,
                                  batch_size=Config.train_batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=Config.num_workers,
                                  batch_size=Config.train_batch_size, drop_last=True)

    print("dataloader initialized")

    # set_data_for_baseline(Config.BASE_TRAIN_PATH, train_dataloader)
    # set_data_for_baseline(Config.BASE_TEST_PATH, test_dataloader)
    #
    # # start the baseline
    #
    # Baseline_model = Baseline(Config.BASE_TRAIN_PATH, Config.BASE_TEST_PATH)
    # Baseline_model.fit(remove_duplicates=True)
    # Baseline_model.predict(remove_duplicates=True)

    # Model
    Patent_Model = Model(train_dataloader, test_dataloader)

    Patent_Model.train(epoch=30, from_pretrain=True)
    Patent_Model.test_ROC_Curve(from_pretrain=True)

    # Patent_Model.set_data_for_pairwise(Config.MODEL_2_TRAIN_PATH, train=True)
    # Patent_Model.set_data_for_pairwise(Config.MODEL_2_TEST_PATH, train=False)
