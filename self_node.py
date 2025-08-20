#!/usr/bin/env python3
# train_compare.py -- 在合成 BLP 实例上对比 GNCC 与多种 GNN 基线（不修改原始 GNCC 代码）

import numpy as np
import torch
import random
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    TransformerConv,
    PNAConv,
    NNConv,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict

# ====== 合成 BLP 实例生成 ======
def generate_blp_instance(n_vars: int, n_cons: int,
                          density: float = 0.2, seed: int = None) -> Data:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    x_star = np.random.randint(0, 2, size=n_vars)
    A = (np.random.rand(n_cons, n_vars) < density).astype(int)
    for i in range(n_cons):
        if A[i].sum() == 0:
            j = np.random.randint(n_vars)
            A[i, j] = 1
    b = []
    for i in range(n_cons):
        sel = np.where(A[i] == 1)[0]
        s = x_star[sel].sum()
        rhs = s if np.random.rand() < 0.5 else s + np.random.randint(0, len(sel)+1)
        b.append(rhs)
    c = np.random.randn(n_vars)
    edges = []
    for i in range(n_cons):
        for j in np.where(A[i] == 1)[0]:
            edges += [[i, n_cons + j], [n_cons + j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    cons_feat = torch.tensor(b, dtype=torch.float).view(n_cons, 1)
    var_feat  = torch.tensor(c, dtype=torch.float).view(n_vars, 1)
    x = torch.cat([cons_feat, var_feat], dim=0)
    y = torch.cat([
        torch.zeros(n_cons, dtype=torch.long),
        torch.tensor(x_star, dtype=torch.long)
    ])
    var_mask = torch.zeros(n_cons + n_vars, dtype=torch.bool)
    var_mask[n_cons:] = True
    return Data(x=x, edge_index=edge_index, y=y, var_mask=var_mask)

# ====== 原始环-割共现特征增强（不改动） ======
def augment_cycle_cut(data: Data) -> Data:
    G = nx.Graph()
    N = data.x.size(0)
    G.add_nodes_from(range(N))
    for u, v in data.edge_index.t().tolist():
        G.add_edge(u, v)
    cycles = nx.cycle_basis(G)
    cycle_sets = [
        {(min(c[t], c[(t+1)%len(c)]), max(c[t], c[(t+1)%len(c)])) for t in range(len(c))}
        for c in cycles
    ]
    T = nx.minimum_spanning_tree(G)
    cut_sets = []
    for u, v in list(T.edges()):
        T.remove_edge(u, v)
        comps = list(nx.connected_components(T))
        T.add_edge(u, v)
        lab = {n: idx for idx, comp in enumerate(comps) for n in comp}
        cut_sets.append({(min(x, y), max(x, y)) for x, y in G.edges() if lab[x] != lab[y]})
    attrs = []
    for u, v in data.edge_index.t().tolist():
        key = (min(u, v), max(u, v))
        feat = [1.0 if key in s else 0.0 for s in (cycle_sets + cut_sets)]
        attrs.append(feat)
    data.edge_attr = torch.tensor(attrs, dtype=torch.float)
    return data

# ====== 原始 GNCC 模型定义（不改动） ======
class GNCC(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, edge_dim, num_classes):
        super().__init__()
        self.edge_nn1 = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, in_ch * hid_ch), torch.nn.ReLU(),
            torch.nn.Linear(in_ch * hid_ch, in_ch * hid_ch)
        )
        self.conv1 = NNConv(in_ch, hid_ch, self.edge_nn1, aggr='mean')
        self.edge_nn2 = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hid_ch * hid_ch), torch.nn.ReLU(),
            torch.nn.Linear(hid_ch * hid_ch, hid_ch * hid_ch)
        )
        self.conv2 = NNConv(hid_ch, hid_ch, self.edge_nn2, aggr='mean')
        self.lin   = torch.nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.lin(x)

# ====== 原始 GNCC 训练 & 评估函数（不改动） ======
def train(model, data, opt, crit, device):
    model.train()
    data = data.to(device)
    opt.zero_grad()
    out = model(data)
    loss = crit(out[data.var_mask], data.y[data.var_mask])
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    data = data.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    mask = data.var_mask
    return (pred[mask] == data.y[mask]).float().mean().item()

# ====== 基线 GNN 定义 ======
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.lin   = torch.nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_ch, hid_ch), torch.nn.ReLU(),
            torch.nn.Linear(hid_ch, hid_ch)
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hid_ch, hid_ch), torch.nn.ReLU(),
            torch.nn.Linear(hid_ch, hid_ch)
        )
        self.conv2 = GINConv(nn2)
        self.lin   = torch.nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class TransformerNet(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        self.conv1 = TransformerConv(in_ch, hid_ch, heads=4, concat=False)
        self.conv2 = TransformerConv(hid_ch, hid_ch, heads=4, concat=False)
        self.lin   = torch.nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class PNA(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        deg = torch.arange(0, 11, dtype=torch.long)
        self.conv1 = PNAConv(in_ch, hid_ch,
                             aggregators=['mean','max','sum'],
                             scalers=['identity'], deg=deg)
        self.conv2 = PNAConv(hid_ch, hid_ch,
                             aggregators=['mean','max','sum'],
                             scalers=['identity'], deg=deg)
        self.lin   = torch.nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

# ====== 主流程 ======
if __name__ == "__main__":
    # 参数
    NUM_RUNS   = 20
    N_VARS     = 400
    N_CONS     = 400
    DENSITY    = 0.02
    BASE_SEED  = 0
    HIDDEN_DIM = 64
    LR         = 1e-3
    EPOCHS     = 100
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预生成并增强数据（仅一次）
    data = generate_blp_instance(N_VARS, N_CONS, DENSITY, seed=BASE_SEED)
    data = augment_cycle_cut(data)
    data = data.to(DEVICE)

    # 模型列表（GNCC 保持原样）
    model_constructors: Dict[str, torch.nn.Module] = {
        'GCN':         GCN,
        'GIN':         GIN,
        'Transformer': TransformerNet,
        'PNA':         PNA,
        'GNCC':        GNCC
    }

    summary = {}
    for name, ctor in model_constructors.items():
        accs = []
        print(f"\n=== Evaluating {name} ===")
        for run in range(NUM_RUNS):
            seed = BASE_SEED + run
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # 构建模型
            if name == 'GNCC':
                edge_dim = data.edge_attr.size(1)
                model = ctor(data.x.size(1), HIDDEN_DIM, edge_dim, 2).to(DEVICE)
            else:
                model = ctor(data.x.size(1), HIDDEN_DIM, 2).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
            criterion = torch.nn.CrossEntropyLoss()

            # 训练 & 评估
            best_acc = 0.0
            patience = 0
            for epoch in range(1, EPOCHS + 1):
                loss = train(model, data, optimizer, criterion, DEVICE)
                acc  = evaluate(model, data, DEVICE)
                scheduler.step(acc)
                if acc > best_acc:
                    best_acc, patience = acc, 0
                else:
                    patience += 1
                    if patience >= 20:
                        break

            accs.append(best_acc)
            print(f"{name} Run {run+1}/{NUM_RUNS}: Best Acc = {best_acc:.4f}")

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs, ddof=1)
        summary[name] = (mean_acc, std_acc)
        print(f">>> {name} Average: {mean_acc:.4f} ± {std_acc:.4f}")

    # 汇总输出
    print("\n=== Final Summary ===")
    for name, (m, s) in summary.items():
        print(f"{name:12s}: Mean Acc = {m:.4f}, Std = {s:.4f}")

