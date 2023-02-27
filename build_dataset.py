import torch
import os
from torch.utils.data import Dataset


# Define the second dataset that uses the first dataset
class ZeroShotDataset(Dataset):
    def __init__(self, _dataset, classes=[], root_dir='.'):
        self._dataset = _dataset
        self.classes = classes
        self.root_dir = root_dir

        self.zsl_dict = load_embeddings(
            os.path.join(root_dir, "glove.6B.100d.txt"), classes)  # {'class': vector}
        
        self.embedding_metric = get_embed_matrix(self.classes, self.zsl_dict)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


def load_embeddings(path, classes):
    """
    path: Path of word embeddings
    wanted_embs: String list of training classes of zero shot model
    zero_shot_classes: String list of all classes in zero shot model
    """

    embed_file = open(path, encoding='gb18030', errors='ignore')
    text = embed_file.read()
    word_sep = text.split('\n')

    # Last element is blank
    word_sep.pop()
    word_dict = {}

    for word_part in word_sep:
        temp_array = word_part.split()
        if temp_array[0] in classes:
            embeddingList = list(map(float, temp_array[1:]))
            word_dict[temp_array[0]] = embeddingList  # 每个用100维的向量去表示

    return word_dict


def get_embed_matrix(classes, dic):
    matrix = torch.zeros(len(classes), 100)   # 100为词向量长度
    for i, name in enumerate(classes):
        matrix[i, :] = torch.tensor(dic[name], dtype=torch.float32)

    return matrix