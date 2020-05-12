# import the necessary packages
from config import imagenet_alexnet_config as config
from sklearn.model_selection import train_test_split
from imagenethelper import ImageNetHelper
import numpy as np
import progressbar
import json
import cv2

print('[INFO] loading image paths...')
inh = ImageNetHelper(config)
trainPaths, trainLabels = inh.buildTrainingSet()
valPaths, valLabels = inh.buildValidationSet()
classLabelNamesMappings = inh.labelNameMappings

print('[INFO] saving class labels...')
content = ''
for wordID, labelName in classLabelNamesMappings.items():
    content += '{}\t{}\n'.format(wordID, labelName)
with open(config.WORD_NUM, 'w') as f:
    f.write(content)

print('[INFO] constructing splits...')
splits = train_test_split(trainPaths, trainLabels,
                          test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
trainPaths, testPaths, trainLabels, testLabels = splits

datasets = [
    ('train', trainPaths, trainLabels, config.TRAIN_MX_LIST),
    ('val', valPaths, valLabels, config.VAL_MX_LIST),
    ('test', testPaths, testLabels, config.TEST_MX_LIST)
]

R, G, B = [], [], []
for dType, paths, labels, outputPath in datasets:
    print('[INFO] building {}...'.format(outputPath))
    widgets = ['Building List: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    f = open(outputPath, 'w')
    for i, (path, label) in enumerate(zip(paths, labels)):
        row = '\t'.join([str(i), str(label), path])
        f.write('{}\n'.format(row))
        if dType == 'train':
            image = cv2.imread(path)
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        pbar.update(i)
    pbar.finish()
    f.close()

print('[INFO] serializing means...')
D = {'R':np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
