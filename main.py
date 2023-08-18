import numpy
import torch.cuda
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils.train_utils import train, test
from model import MLP
from config import Args
from utils.train_utils import EarlyStopping, set_seed


args = Args().get_parsr()
set_seed(args.seed)

digits = datasets.load_digits()
X = digits.data.astype(numpy.float32)
y = digits.target.astype(numpy.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_size)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=args.train_size)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size_train, drop_last=True, shuffle=True)
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, drop_last=True, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP(64, 10).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
early_stopping = EarlyStopping(args.patience)

for epoch in range(args.epochs):
    train(model, epoch, train_loader, optimizer, criterion, device)
    val_loss, val_acc, _, _, _ = test(model, val_loader, criterion, device, batch_size=args.batch_size_test)
    early_stopping(val_loss, model, args.output_dir)

    if early_stopping.early_stop:
        print('Early stopping, save checkpoints in', args.output_dir)
        break

best_model = MLP(64, 10).to(device)
print('Load best model checkpoints')
best_model.load_state_dict(torch.load(args.output_dir + '/' + 'model_checkpoint.pth'))
_, accuracy, precision, recall, f1 = test(model, test_loader, criterion, device, batch_size=args.batch_size_test,
                                          show_roc_curve=True, show_confusion_matrix=True)

print('precision is', precision)
print('recall score is', recall)
print('f1 score is', f1)






