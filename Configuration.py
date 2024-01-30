class Config():
    PATH_TO_CHECKPT = "checkpoint/additional_features.pth"
    EPOCH = 10
    LEARNING_RATE = 0.001
    RAW_DATA_PATH = "data/newdata/filterd_with_cpc_500000.csv"
    #RAW_DATA_PATH = "data/newdata/filterd_with_coinventors_100000.csv"
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
    DATASET_LENGTH = 200000
    ADD_CPC = True