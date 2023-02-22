import torch
import os
from torch.utils.data import Dataset


# Define the second dataset that uses the first dataset
class ZeroShotDataset(Dataset):
    def __init__(self, _dataset, classes=[], root_dir=''):
        self._dataset = _dataset
        self.classes = classes
        self.root_dir = root_dir

        self.zsl_dict = load_embeddings(
            "glove.6B.100d.txt", train_classes, zsl_classes)  # {'class': vector}

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


def load_embeddings(path, wanted_embs, zero_shot_classes):
    """
    path: Path of word embeddings
    wanted_embs: String list of training classes of zero shot model
    zero_shot_classes: String list of all classes in zero shot model
    """

    total_classes = wanted_embs+zero_shot_classes
    embed_file = open(path, encoding='gb18030', errors='ignore')
    text = embed_file.read()
    word_sep = text.split('\n')

    # Last element is blank
    word_sep.pop()
    word_dict = {}

    for word_part in word_sep:
        temp_array = word_part.split()
        if temp_array[0] in total_classes:
            embeddingList = list(map(float, temp_array[1:]))
            word_dict[temp_array[0]] = embeddingList  # 每个用100维的向量去表示

    return word_dict
