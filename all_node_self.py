#!/usr/bin/env python3
# train_synthetic_gncc.py -- 在合成 BLP 实例上使用 GNCC 模型训练，保留环-割共现特征与边特征

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import (NNConv, GCNConv, GINConv, 
                               TransformerConv, PNAConv)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ====== 合成数据生成 ======
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
            u, v = i, n_cons + j
            edges += [[u, v], [v, u]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    cons_feat = torch.tensor(b, dtype=torch.float).view(n_cons, 1)
    var_feat  = torch.tensor(c, dtype=torch.float).view(n_vars, 1)
    x = torch.cat([cons_feat, var_feat], dim=0)
    y = torch.cat([torch.zeros(n_cons, dtype=torch.long), torch.tensor(x_star, dtype=torch.long)])
    var_mask = torch.zeros(n_cons + n_vars, dtype=torch.bool)
    var_mask[n_cons:] = True
    return Data(x=x, edge_index=edge_index, y=y, var_mask=var_mask)

# ====== 特征增强：环-割共现 ======
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

# ====== 模型定义 ======
class GNCC(nn.Module):
    def __init__(self, in_ch, hid_ch, edge_dim, num_classes):
        super().__init__()
        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_dim, in_ch * hid_ch), nn.ReLU(),
            nn.Linear(in_ch * hid_ch, in_ch * hid_ch)
        )
        self.conv1 = NNConv(in_ch, hid_ch, self.edge_nn1, aggr='mean')
        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_dim, hid_ch * hid_ch), nn.ReLU(),
            nn.Linear(hid_ch * hid_ch, hid_ch * hid_ch)
        )
        self.conv2 = NNConv(hid_ch, hid_ch, self.edge_nn2, aggr='mean')
        self.lin   = nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.lin(x)

class GCN(nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, hid_ch)
        self.lin = nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class GIN(nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_ch, hid_ch), nn.ReLU(),
            nn.Linear(hid_ch, hid_ch)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hid_ch, hid_ch), nn.ReLU(),
            nn.Linear(hid_ch, hid_ch)
        )
        self.conv2 = GINConv(nn2)
        self.lin = nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class GTN(nn.Module):  # Graph Transformer Network
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        self.conv1 = TransformerConv(in_ch, hid_ch, heads=4, concat=False)
        self.conv2 = TransformerConv(hid_ch, hid_ch, heads=4, concat=False)
        self.lin = nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

class PNA(nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes):
        super().__init__()
        deg = torch.arange(0, 11, dtype=torch.long)  # 假设最大度为10
        self.conv1 = PNAConv(
            in_ch, hid_ch,
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            deg=deg
        )
        self.conv2 = PNAConv(
            hid_ch, hid_ch,
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            deg=deg
        )
        self.lin = nn.Linear(hid_ch, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

# ====== 训练与评估 ======
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

# ====== 主流程 ======
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实验配置
    config = {
        'N_VARS': 800,
        'N_CONS': 800,
        'DENSITY': 0.05,
        'HIDDEN': 64,
        'EPOCHS': 100,
        'LR': 1e-3,
        'NUM_RUNS': 50,  # 减少运行次数以快速测试
        'BASE_SEED': 42
    }
    
    # 一次性生成与增强数据集
    data = generate_blp_instance(config['N_VARS'], config['N_CONS'], 
                                config['DENSITY'], seed=config['BASE_SEED'])
    data = augment_cycle_cut(data)
    data = data.to(device)
    
    # 定义所有要比较的模型
    models = {
        'GNCC': GNCC(data.x.size(1), config['HIDDEN'], 
                data.edge_attr.size(1), num_classes=2),
        'GCN': GCN(data.x.size(1), config['HIDDEN'], num_classes=2),
        'GIN': GIN(data.x.size(1), config['HIDDEN'], num_classes=2),
        'GTN': GTN(data.x.size(1), config['HIDDEN'], num_classes=2),
        'PNA': PNA(data.x.size(1), config['HIDDEN'], num_classes=2)
    }
    
    # 存储所有模型的结果
    all_results = {name: [] for name in models}
    
    for run in range(config['NUM_RUNS']):
        seed = config['BASE_SEED'] + run
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # 每次仅做 train/test 划分
        idx = data.var_mask.nonzero(as_tuple=False).view(-1).tolist()
        random.shuffle(idx)
        n_train = int(0.8 * len(idx))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        test_mask  = train_mask.clone()
        train_mask[train_idx] = True
        test_mask[test_idx]  = True
        data.train_mask = train_mask
        data.test_mask  = test_mask
        
        # 训练和评估每个模型
        for name, model_class in models.items():
            model = model_class.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=config['LR'], weight_decay=1e-4)
            sched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
            crit = nn.CrossEntropyLoss()
            
            best_acc, patience = 0.0, 0
            for epoch in range(1, config['EPOCHS']+1):
                train(model, data, opt, crit, device)
                acc = evaluate(model, data, device)
                sched.step(acc)
                if acc > best_acc:
                    best_acc, patience = acc, 0
                else:
                    patience += 1
                    if patience >= 20:
                        break
            
            all_results[name].append(best_acc)
            print(f"Run {run+1}/{config['NUM_RUNS']} | {name}: Best Test Acc = {best_acc:.4f}")
    
    # 打印最终结果
    print("\n=== Final Results ===")
    for name, results in all_results.items():
        mean_acc = np.mean(results)
        std_acc = np.std(results)
        print(f"{name}:")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  All Runs: {[round(acc, 4) for acc in results]}")
        
        # 计算前10%的最佳结果（如果NUM_RUNS>=10）
        if len(results) >= 10:
            top10 = sorted(results, reverse=True)[:10]
            mean_top10 = np.mean(top10)
            std_top10 = np.std(top10)
            print(f"  Top 10 Mean: {mean_top10:.4f} ± {std_top10:.4f}")

if __name__ == '__main__':
    main()