import math
import os
import random
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from .metric_utils import draw_roc_curve, draw_confusion_matrix


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss

        if not os.path.exists(path):
            os.mkdir(path)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        self.val_loss_min = val_loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, epoch, train_loader, optimizer, criterion, device):

    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil((batch_idx + 1) / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


def test(model, test_loader, criterion, device, batch_size, show_roc_curve=False, show_confusion_matrix=False):

    model.eval()
    test_loss = 0
    correct = 0
    # 预测类别的数量
    num_classes = 10

    total_target = torch.ones(len(test_loader.dataset))
    total_pred = torch.ones(len(test_loader.dataset))
    total_soft_max_pred = torch.ones((len(test_loader.dataset), num_classes))
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            total_target[index * batch_size: (index+1) * batch_size] = target
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            total_soft_max_pred[index * batch_size: (index+1) * batch_size] = output
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            total_pred[index * batch_size: (index+1) * batch_size] = pred
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    if show_roc_curve:
        draw_roc_curve(total_target, total_soft_max_pred, range(10))

    if show_confusion_matrix:
        draw_confusion_matrix(total_target, total_pred, range(10))

    print('\nTest: average loss: {:.6f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    total_target = total_target.numpy()
    total_pred = total_pred.numpy()

    precision = precision_score(total_target, total_pred, average='macro')
    recall = recall_score(total_target, total_pred, average='macro')
    f1 = f1_score(total_target, total_pred, average='macro')
    # print('precision is', precision_score(total_target, total_pred, average='macro'))
    # print('recall score is', recall_score(total_target, total_pred, average='macro'))
    # print('f1 score is', f1_score(total_target, total_pred, average='macro'))

    return test_loss, correct / len(test_loader.dataset), precision, recall, f1

