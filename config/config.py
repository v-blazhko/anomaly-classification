import os

ORIG_INPUT_DATASET = "datasets/orig"
BASE_PATH = "datasets/idc"

TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

NUM_EPOCHS = 1
INIT_LR = 1e-1
BATCH_SIZE = 32
