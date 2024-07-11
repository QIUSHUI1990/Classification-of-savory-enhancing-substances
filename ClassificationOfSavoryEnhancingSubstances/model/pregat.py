import random, argparse
from rdkit import Chem
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from figure.heatmap import draw_heatmap

atom_encoding = {
        0:0,'C': 1, 'N': 2, 'O': 3, 'S': 4, 'H': 5 
    }
bond_encoding = {
        'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3, 'TRIPLE': 4 
    }
def tensortogat(x,batch):
    classified_data = [[]]
    atomsm = batch.numpy()
    x = x.detach().numpy()
    labelnum = 0
    for element, label in zip(x, atomsm):
        if label == labelnum:
            classified_data[-1].append(element)
        else:
            classified_data[-1] = np.array(classified_data[-1])
            labelnum = labelnum+1
            classified_data.append([])
            classified_data[-1].append(element)
    classified_data[-1] = np.array(classified_data[-1])
    scalar_feature_gat_padded = np.array(
        [np.pad(sf, ((0, 24 - sf.shape[0]), (0, 0)), 'constant') for sf in classified_data])
    return  scalar_feature_gat_padded
class GAT(nn.Module):
    # 1, 4, 128, 256
    def __init__(self, num_node_features, num_classes, hidden_size1,hidden_size2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_size1,heads=4)
        self.conv2 = GATConv(hidden_size1 * 4, hidden_size2)
        self.conv3 = GATConv(hidden_size2, num_classes)


    def forward(self, data,hyper,valsmile,test):
        x, edge_index ,edge_attr, batch = data.x, data.edge_index,data.edge_attr,data.batch
        x = self.conv1(x, edge_index,edge_attr)
        if test:
            gatxxx = tensortogat(x,batch)
            mols = [Chem.MolFromSmiles(smi) for smi in valsmile]
            draw_heatmap(hyper, gatxxx, mols)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = pyg_nn.global_add_pool(x, batch)
        return F.log_softmax(x, dim=1), x
def custom_collate(data_list):
    return Batch.from_data_list(data_list)
def getdatalist(smiles_list):
    maskdatalist = []

    for smiles in smiles_list:
        if len(smiles) != 1:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                node_encodings = [atom_encoding[atom.GetSymbol()] for atom in mol.GetAtoms()]
                edges = [[],[]]
                for bond in mol.GetBonds():
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    edges[0].append(begin_idx)
                    edges[1].append(end_idx)

                edge_encodings = []
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    edge_encoding = bond_encoding.get(bond_type.name, None)
                    edge_encodings.append(edge_encoding)
        x = torch.tensor(node_encodings, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_attr = torch.tensor(edge_encodings, dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        maskgraph_data = maskatom(graph_data)
        maskdatalist.append(maskgraph_data)
    return maskdatalist
def maskatom(graph_data):
    num_nodes = graph_data.x.size(0)
    random_index = torch.randint(0, num_nodes, (1,)).item()
    y=torch.tensor(((graph_data.x[random_index])-1), dtype=torch.long)
    graph_data.x[random_index] = float(0)
    maskgraph_data = graph_data
    maskgraph_data.x = graph_data.x.view(-1,1)
    maskgraph_data.y=y.view(-1)
    return maskgraph_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--units_conv', default=24, type=int)
    parser.add_argument('--num_atoms', default=24, type=int)
    parser.add_argument('--fold_total', default=10, type=int)
    parser.add_argument('--fold', default=1, type=int)

    hyper = parser.parse_args()
    df = pd.read_csv('data/smiles.csv')
    df = df.values
    smiles_list = [i[0] for i in df]
    maskdatalist = getdatalist(smiles_list)
    train_loader = DataLoader(maskdatalist, batch_size=266, shuffle=True, collate_fn=custom_collate)

    model = GAT(1,4,128,128)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    epoches = 1000
    losses = []
    for epoch in range(epoches):
        epoch_loss = 0.0
        index = 0
        correct1 = 0
        total1 = 0
        for data in train_loader:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
            train = data.y
            model.train()
            optimizer.zero_grad()
            output,x_feat = model(data,hyper,smiles_list,False)
            _, predicted1 = torch.max(output, 1)

            flat_output = output.view(-1,4)
            flat_labels = train.view(-1)
            loss = criterion(flat_output, flat_labels)
            loss.backward()
            optimizer.step()
            index = index + 1
            epoch_loss = epoch_loss+loss.item()
            total1 += train.size(0)
            correct1 += (predicted1 == train).sum().item()
        losses.append(epoch_loss / (index))

        print(f'Epoch: {epoch}, Loss: {epoch_loss/index}')

    print(f'Accuracy on train set_train: {100 * correct1 / total1:.2f}%')
    x_feat = x_feat.detach().numpy()

    #np.savetxt('GAT_feat.csv', x_feat, delimiter=',')
    losses = np.array(losses[:300])
    class_accuracies = [0, 0, 0, 0]
    for pred, true in zip(predicted1, train):
        class_accuracies[pred] += (pred == true)
    class_counts = np.bincount(train)

    i = 1
    for accuracy, count in zip(class_accuracies, class_counts):
        print(f'Accuracy on train class{i}: {100 * accuracy / count:.2f}%')
        i = i + 1

    model.eval()
    correct = 0
    total = 0
    index1 = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in train_loader:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
            test = data.y
            output,_ = model(data,hyper,smiles_list,True)
            _, predicted = torch.max(output, 1)
            flat_output1 = output.view(-1, 4)
            flat_labels1 = test.view(-1)
            loss1 = criterion(flat_output1, flat_labels1)
            index1 = index1 + 1
            test_loss = test_loss + loss1.item()
            total += test.size(0)
            correct += (predicted == test).sum().item()
        print(f'Loss_test: {test_loss / (index1):.4f}')
        print(f'Accuracy on test set_test: {100 * correct / total:.2f}%')
    class_accuracies1 = [0, 0, 0, 0]
    for pred1, true1 in zip(predicted, test):
        class_accuracies1[pred1] += (pred1 == true1)

    class_counts1 = np.bincount(test)
    j = 1
    for accuracy1, count1 in zip(class_accuracies1, class_counts1):
        print(f'Accuracy on test class{j}:{100 * accuracy1 / count1:.2f}%')
        j = j + 1
    # print(f'Fold {fold + 1}:', class_report)
    # foldloss.append(loss.item())
