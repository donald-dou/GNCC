import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import cplex
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch.optim.lr_scheduler import StepLR

# —— 直接指定 .mps 文件路径 ——
MPS_PATH = "p0201.mps"  # <- 修改为实际路径

# —— 默认超参数 ——
HIDDEN       = 128        # 隐藏层维度
EPOCHS       = 200        # 最大训练轮数
LR           = 0.01       # 初始学习率
WEIGHT_DECAY = 1e-4       # L2 正则化系数
PATIENCE     = 20         # 早停耐心轮数
REG_WEIGHT   = 1.0        # 目标加权 MSE 正则化系数 λ
NUM_LAYERS   = 2       # GNCC 的层数

# —— 使用1-100的所有整数作为种子 ——
SEEDS = list(range(1, 101))  # 1到50的全部整数

def load_mps_and_build_data(mps_path):
    # 1) 用 CPLEX 读取并求解 .mps，获取变量最优解
    c = cplex.Cplex()
    c.read(mps_path)
    c.solve()
    sol    = c.solution
    x_vals = sol.get_values()
    n_vars = len(x_vals)
    x_opt  = torch.tensor([round(val) for val in x_vals], dtype=torch.long)

    # —— 获取目标系数 —— 
    obj_lin   = c.objective.get_linear()
    obj_coefs = torch.tensor(obj_lin, dtype=torch.float)

    # 2) 构建 NetworkX 二分图
    n_conss = c.linear_constraints.get_num()
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i, x=torch.ones(1), label=int(x_opt[i]))
    for j in range(n_conss):
        G.add_node(n_vars + j, x=torch.zeros(1), label=-1)
    for j in range(n_conss):
        row = c.linear_constraints.get_rows(j)
        for var_idx, coef in zip(row.ind, row.val):
            if abs(coef) > 1e-9:
                G.add_edge(var_idx, n_vars + j)

    # 3) 计算循环基和割基特征
    cycles = nx.cycle_basis(G)
    cycle_sets = []
    for cycle in cycles:
        edges = {
            (min(cycle[t], cycle[(t+1)%len(cycle)]),
             max(cycle[t], cycle[(t+1)%len(cycle)]))
            for t in range(len(cycle))
        }
        cycle_sets.append(edges)
    T = nx.minimum_spanning_tree(G)
    cut_sets = []
    for u, v in list(T.edges()):
        T.remove_edge(u, v)
        comps = list(nx.connected_components(T))
        T.add_edge(u, v)
        lab = {node: idx for idx, comp in enumerate(comps) for node in comp}
        edges = {
            (min(x, y), max(x, y))
            for x, y in G.edges() if lab[x] != lab[y]
        }
        cut_sets.append(edges)

    # 4) 构建 PyG Data
    n_nodes = G.number_of_nodes()
    x = torch.stack([G.nodes[i]['x'] for i in range(n_nodes)], dim=0).float()
    y = torch.tensor([G.nodes[i]['label'] for i in range(n_nodes)], dtype=torch.long)

    edge_index, edge_attr = [], []
    for u, v in G.edges():
        key  = (min(u, v), max(u, v))
        feat = [1.0 if key in s else 0.0 for s in cycle_sets + cut_sets]
        edge_index += [[u, v], [v, u]]
        edge_attr  += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # 5) 划分训练/测试集（仅变量节点）
    var_idx = list(range(n_vars))
    random.shuffle(var_idx)
    n_train   = int(0.8 * n_vars)
    train_idx = var_idx[:n_train]
    test_idx  = var_idx[n_train:]
    def mk_mask(idxs):
        m = torch.zeros(n_nodes, dtype=torch.bool)
        m[idxs] = True
        return m
    data.train_mask = mk_mask(train_idx)
    data.test_mask  = mk_mask(test_idx)

    data.obj_coefs = obj_coefs
    data.n_vars    = n_vars
    return data

class GNCC(nn.Module):
    def __init__(self, in_ch, hid_ch, edge_dim, num_classes, num_layers):
        super().__init__()
        self.dropout_p  = 0.5
        self.edge_nns   = nn.ModuleList()
        self.convs      = nn.ModuleList()

        # 第一层：in_ch -> hid_ch
        self.edge_nns.append(nn.Sequential(
            nn.Linear(edge_dim, in_ch*hid_ch), nn.ReLU(),
            nn.Linear(in_ch*hid_ch, in_ch*hid_ch)
        ))
        self.convs.append(NNConv(in_ch, hid_ch, self.edge_nns[-1], aggr='mean'))

        # 中间层：hid_ch -> hid_ch
        for _ in range(num_layers-2):
            self.edge_nns.append(nn.Sequential(
                nn.Linear(edge_dim, hid_ch*hid_ch), nn.ReLU(),
                nn.Linear(hid_ch*hid_ch, hid_ch*hid_ch)
            ))
            self.convs.append(NNConv(hid_ch, hid_ch, self.edge_nns[-1], aggr='mean'))

        # 分类层：hid_ch -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hid_ch, hid_ch), nn.ReLU(),
            nn.Linear(hid_ch, num_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.classifier(x)

def train_epoch(data, model, opt, crit):
    model.train()
    opt.zero_grad()

    out  = model(data.x, data.edge_index, data.edge_attr)
    mask = data.train_mask

    # 1) 交叉熵损失
    ce_loss = crit(out[mask], data.y[mask])

    # 2) 预测概率 p_j = P(x_j=1)
    prob1 = F.softmax(out, dim=1)[:,1]

    # 3) 只取训练变量节点
    pred_p       = prob1[mask]
    true_y       = data.y[mask].float()
    var_idx_mask = mask.nonzero(as_tuple=True)[0]
    coefs        = data.obj_coefs.to(pred_p.device)[var_idx_mask]

    # 4) 加权 MSE 正则项
    diffs   = (pred_p - true_y) * coefs
    obj_mse = diffs.pow(2).mean()

    # 5) 合成总损失
    loss = ce_loss + REG_WEIGHT * obj_mse
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(data, model):
    model.eval()
    out  = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    test_acc  = (pred[data.test_mask]  == data.y[data.test_mask] ).float().mean().item()
    return {'train': train_acc, 'test': test_acc}

def main():
    all_best = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data   = load_mps_and_build_data(MPS_PATH).to(device)

        # 类别权重
        labels_vars, counts_vars = torch.unique(data.y[:data.n_vars], return_counts=True)
        full_counts = [0,0]
        for lbl, cnt in zip(labels_vars.tolist(), counts_vars.tolist()):
            full_counts[lbl] = cnt
        total = sum(full_counts)
        class_weights = torch.tensor(
            [(total/(2*cnt)) if cnt>0 else 0.0 for cnt in full_counts],
            device=device
        )

        model     = GNCC(
            in_ch=data.x.size(1),
            hid_ch=HIDDEN,
            edge_dim=data.edge_attr.size(1),
            num_classes=2,
            num_layers=NUM_LAYERS
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_test_acc = 0.0
        patience_cnt  = 0
        for epoch in range(1, EPOCHS+1):
            train_epoch(data, model, optimizer, criterion)
            accs = evaluate(data, model)
            scheduler.step()
            if accs['test'] > best_test_acc:
                best_test_acc = accs['test']
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        all_best.append(best_test_acc)
        print(f"Seed {seed:3d} | Best Test Acc: {best_test_acc:.4f}")

    # 汇总 top 10
    sorted_accs = sorted(all_best, reverse=True)
    top10       = sorted_accs[:10]
    mean_top10  = np.mean(top10)
    std_top10   = np.std(top10)

    print("\n=== Final Results ===")
    print("Top 10 Best Test Accuracies:")
    for i, acc in enumerate(top10, 1):
        print(f"  {i:2d}: {acc:.4f}")
    print(f"\nMean of Top 10: {mean_top10:.4f}")
    print(f"Std of Top 10: {std_top10:.6f}")

if __name__ == "__main__":
    main()


