# import the necessary packages
from os import path 

BASH_PATH = 'G:/tiny_imagenet_200'
MX_OUTPUT = 'G:/tiny_imagenet_200'

WORD_IDS = path.sep.join([BASH_PATH, 'words.txt'])
WORD_NUM = path.sep.join([BASH_PATH, 'word_name.txt']) # <<<
TRAIN_DIR = path.sep.join([BASH_PATH, 'train'])
VAL_DIR = path.sep.join([BASH_PATH, 'val/images'])
VAL_LIST = path.sep.join([BASH_PATH, 'val/val_annotations.txt'])

TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, 'lists/train.lst'])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, 'lists/val.lst'])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, 'lists/test.lst'])

TRAIN_MX_REC = path.sep.join([MX_OUTPUT, 'rec/train.rec'])
VAL_MX_REC = path.sep.join([MX_OUTPUT, 'rec/val.rec'])
TEST_MX_REC = path.sep.join([MX_OUTPUT, 'rec/test.rec'])

DATASET_MEAN = path.sep.join([MX_OUTPUT, 'output/imagenet_mean.json'])

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES 
BATCH_SIZE = 4
EPOCHS = 10
NUM_DEVICES = 1

