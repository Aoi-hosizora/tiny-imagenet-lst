# import the necessary packages
import numpy as np
import os

class ImageNetHelper:

    def __init__(self, config):
        self.config = config
        self.labelNumbersMappings, self.labelNameMappings = self.buildClassLabels()

    def buildClassLabels(self):
        # n00001740	entity
        # n00001930	physical entity
        rows = open(self.config.WORD_IDS).read().strip().split('\n')
        labelNumbersMappings = {}
        labelNameMappings = {}
        dirs = os.listdir(self.config.TRAIN_DIR)
        for idx, wordID in enumerate(dirs):
            labelNumbersMappings[wordID] = idx
        for row in rows:
            wordID, name = row.split('\t')
            if wordID in labelNumbersMappings:
                idx = labelNumbersMappings[wordID]
                labelNameMappings[idx] = name
        return labelNumbersMappings, labelNameMappings
    
    def buildTrainingSet(self):
        paths = []
        labels = []
        dirs = os.listdir(self.config.TRAIN_DIR)
        for wordID in dirs:
            wordDir = os.path.sep.join([self.config.TRAIN_DIR, wordID, 'images'])
            images = os.listdir(wordDir)
            for image in images:
                path = os.path.sep.join([wordDir, image])
                label = self.labelNumbersMappings[wordID]
                paths.append(path)
                labels.append(label)
        return np.array(paths), np.array(labels)

    def buildValidationSet(self):
        # val_0.JPEG	n03444034	0	32	44	62
        # val_1.JPEG	n04067472	52	55	57	59
        paths = []
        labels = []
        valFilenames = open(self.config.VAL_LIST).read().strip().split('\n')
        for row in valFilenames:
            name, wordID, *_ = row.strip().split('\t')
            path = os.path.sep.join([self.config.VAL_DIR, name])
            label = self.labelNumbersMappings[wordID]
            paths.append(path)
            labels.append(label)
        return np.array(paths), np.array(labels)
