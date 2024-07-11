import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

data_path = 'data/feat.xlsx'
data = pd.read_excel(data_path)
x = data.iloc[:, 2:].to_numpy(dtype=np.float32)  # 假设前两列不是特征
y = data[data.columns[1]].to_numpy(dtype=np.int64)  # 假设第一列是标签

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_fold_fpr = []
all_fold_tpr = []
all_fold_auc = []
best_loss_values = []
best_roc_curves = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_size = X_train.shape[1]
    hidden_size1 = 128
    hidden_size2 = 64
    hidden_size3 = 32
    num_classes = len(np.unique(y))
    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss_values = []

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            ##
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
    best_loss_values.append(loss_values)

    model.eval()
    with torch.no_grad():
        all_probs = []
        all_labels = []
        for inputs, labels in test_loader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设是二分类问题，取正类概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            predicted_labels = predicted.cpu().numpy()
            results_df = pd.DataFrame({
                'True Labels': all_labels,
                'Predicted Labels': predicted_labels
            })

            # results_df.to_csv(f'results/bp/fold_{fold + 1}_results.csv', index=False)

        # 计算FPR、TPR和AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        best_roc_curves.append((fpr, tpr))
        print(f'Fold {fold+1} AUC: {roc_auc:.4f}')

plt.figure(figsize=(10, 8))

for i, (fpr, tpr) in enumerate(best_roc_curves):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (area = {roc_auc:.2f})')

plt.plot([-0.05, 1], [-0.05, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.title('ROC Curves for All Folds', fontsize=24)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc='lower right')

plt.grid(True)
plt.legend()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.show()


font = {'family': 'SimHei',
        'size': 10}
plt.rc('font', **font)


plt.figure(figsize=(10, 5))
# 刻度大小
plt.tick_params(labelsize=11)

for i, loss_values in enumerate(best_loss_values):
    plt.plot(loss_values, label=f'Fold {i+1}')
# 设置横坐标的范围
plt.xlim([0.0, 1500])
plt.ylim([0.0, 1.0])

plt.legend(loc='upper right')


plt.title('Best Loss Curve for ', fontsize=24)
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Loss', fontsize=20, labelpad=30)

plt.legend()

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from matplotlib.ticker import FuncFormatter

def thousand_formatter(x, pos):
    return f'{int(x):,}'.replace(',', ' ')
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousand_formatter))

y_ticks = plt.gca().get_yticks()
y_ticks = [tick for tick in y_ticks if tick != 0.0]
plt.gca().set_yticks(y_ticks)
plt.gca().set_yticklabels([f'{tick:.1f}' for tick in y_ticks])

plt.show()
