import pandas
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix

data_path = r'data/feat.xlsx'

data = pd.read_excel(data_path)
x = data.iloc[:,2:]
x = x.to_numpy(dtype=np.float32)
y= data[data.columns[1]]

features = np.array(x)
targets = np.array(y).reshape(-1,1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
mccs = []
aucs = []
sensitivities = []
specificities = []

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = targets[train_index], targets[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)


    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)

    sensitivities.append(sn)
    specificities.append(sp)

    mcc = matthews_corrcoef(y_test, y_pred)
    mccs.append(mcc)

    # 计算AUC
    if len(np.unique(y_test)) > 1:
        y_prob = xgb.predict_proba(X_test)[:, 1]
    else:
        auc = 'undefined'
    aucs.append(auc)

# 输出平均结果
print(f"平均模型准确率: {np.mean(accuracies):.4f}")
print(f"平均灵敏度 (Sn): {np.mean(sensitivities):.4f}")
print(f"平均特异度 (Sp): {np.mean(specificities):.4f}")
print(f"平均Matthews相关系数 (MCC): {np.mean(mccs):.4f}")
print(f"平均曲线下面积 (AUC): {np.mean([auc for auc in aucs if auc != 'undefined']):.4f}")

# 使用全部数据训练模型并预测
xgb.fit(scaler.fit_transform(features), targets)

