from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, flag, x_train, y_train):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        if self.flag == 'train':
            return self.x_train, self.y_train
        else:
            return index, self.x_train, self.y_train
