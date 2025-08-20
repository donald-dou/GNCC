import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import cplex
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, PNAConv, TransformerConv
from torch.optim.lr_scheduler import StepLR

# ========= 超参数 =========
MPS_PATH      = "f2gap40400.mps"
HIDDEN        = 128    # 隐藏层维度
EPOCHS        = 200    # 最大训练轮数
LR            = 1e-2   # 初始学习率
WEIGHT_DECAY  = 1e-4   # L2 正则化
PATIENCE      = 20     # 早停耐心轮数
NUM_CLASSES   = 2      # 二分类
DROP_P        = 0.5    # Dropout 概率
NUM_RUNS      = 50     # 运行次数（种子1-50）

# ===== 数据加载 & 图构建 =====
def load_mps_and_build_data(mps_path: str) -> Data:
    c = cplex.Cplex()
    c.read(mps_path)
    c.solve()
    sol = c.solution
    x_vals = sol.get_values()
    n_vars = len(x_vals)
    x_opt = torch.tensor([round(v) for v in x_vals], dtype=torch.long)

    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i, x=torch.ones(1), label=int(x_opt[i]))
    n_conss = c.linear_constraints.get_num()
    for j in range(n_conss):
        G.add_node(n_vars + j, x=torch.zeros(1), label=-1)
    for j in range(n_conss):
        row = c.linear_constraints.get_rows(j)
        for vid, coef in zip(row.ind, row.val):
            if abs(coef) > 1e-9:
                G.add_edge(vid, n_vars + j)

    cycles = nx.cycle_basis(G)
    cycle_sets = [ {(min(cycle[t], cycle[(t+1)%len(cycle)]), 
                    max(cycle[t], cycle[(t+1)%len(cycle)]))
                    for t in range(len(cycle))}
                   for cycle in cycles ]
    T = nx.minimum_spanning_tree(G)
    cut_sets = []
    for u, v in list(T.edges()):
        T.remove_edge(u, v)
        comps = list(nx.connected_components(T))
        T.add_edge(u, v)
        lab = {node: idx for idx, comp in enumerate(comps) for node in comp}
        cut_sets.append({(min(x, y), max(x, y)) for x, y in G.edges() if lab[x] != lab[y]})

    all_sets = cycle_sets + cut_sets
    n_nodes = G.number_of_nodes()
    x = torch.stack([G.nodes[i]['x'] for i in range(n_nodes)], dim=0).float()
    y = torch.tensor([G.nodes[i]['label'] for i in range(n_nodes)], dtype=torch.long)
    edge_index, edge_attr = [], []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        feat = [1.0 if key in s else 0.0 for s in all_sets]
        edge_index += [[u, v], [v, u]]
        edge_attr  += [feat, feat]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    idx = list(range(n_vars))
    random.shuffle(idx)
    n_train = int(0.8 * n_vars)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    mask_train = torch.zeros(n_nodes, dtype=torch.bool)
    mask_test  = torch.zeros(n_nodes, dtype=torch.bool)
    mask_train[train_idx] = True
    mask_test[test_idx]   = True
    data.train_mask = mask_train
    data.test_mask  = mask_test
    return data

# ===== 模型定义 =====
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = nn.Linear(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROP_P, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                             nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                             nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.lin   = nn.Linear(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROP_P, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class TransformerNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.lin   = nn.Linear(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROP_P, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class PNA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        deg = torch.arange(0, 11, dtype=torch.long)
        self.conv1 = PNAConv(in_channels, hidden_channels,
                             aggregators=['mean','max','sum'], scalers=['identity'], deg=deg)
        self.conv2 = PNAConv(hidden_channels, hidden_channels,
                             aggregators=['mean','max','sum'], scalers=['identity'], deg=deg)
        self.lin   = nn.Linear(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index, None))
        x = F.dropout(x, p=DROP_P, training=self.training)
        x = F.relu(self.conv2(x, edge_index, None))
        return self.lin(x)

# ===== 训练与评估 =====
def train_model(data, model, device):
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0; patience = 0
    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad()
        out = model(data)
        loss= criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step(); scheduler.step()
        model.eval()
        pred = out.argmax(dim=1)
        acc_test = (pred[data.test_mask]==data.y[data.test_mask]).float().mean().item()
        if acc_test > best_acc:
            best_acc = acc_test; patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    return best_acc

# ===== 主流程 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(1, NUM_RUNS+1))

    model_results = { 'GCN': [], 'GIN': [], 'Transformer': [], 'PNA': [] }

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        data = load_mps_and_build_data(MPS_PATH)
        models = {
            'GCN':         GCN(data.num_node_features, HIDDEN, NUM_CLASSES),
            'GIN':         GIN(data.num_node_features, HIDDEN, NUM_CLASSES),
            'Transformer': TransformerNet(data.num_node_features, HIDDEN, NUM_CLASSES),
            'PNA':         PNA(data.num_node_features, HIDDEN, NUM_CLASSES)
        }
        for name, m in models.items():
            acc = train_model(data, m, device)
            model_results[name].append(acc)
        print(f"Done seed {seed}")

    for name, accs in model_results.items():
        arr = np.array(accs)
        mean_acc = arr.mean()
        std_acc  = arr.std()
        print(f"\nModel: {name}")
        print(f" Mean Accuracy over {NUM_RUNS} seeds: {mean_acc:.4f}, Std: {std_acc:.4f}")

if __name__ == "__main__":
    main()

