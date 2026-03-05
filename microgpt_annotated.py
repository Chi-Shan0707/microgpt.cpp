"""
microgpt_explainer.py

这是一个"可直接训练 + 推理"的完整 microGPT 脚本，
并在行间加入更细的中文注释，覆盖：
1) 代码意图（工程层面）
2) LLM 算法原理（训练目标、优化、采样）
3) Transformer 思想（表示、注意力、残差、归一化）

使用：
    python microgpt_explainer.py
"""

import math
import os
import random

# -----------------------------
# 全局随机种子
# -----------------------------
# 固定随机性，保证"同一份代码 + 同一份数据"尽量产出同样的训练轨迹。
# 这对调参特别关键：当 loss 变化时，你能确认是参数导致，而不是随机噪声导致。
random.seed(42)

# -----------------------------
# 数据准备
# -----------------------------
# 如果本地没有 input.txt，就自动下载一个最小语料（名字列表）。
# 语言模型学的是条件概率分布：给定前缀，预测下一个符号。
# 因此数据本身就是模型"世界观"的来源。
if not os.path.exists("input.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

# 每一行视作一条样本，清理空行后再打乱。
# 打乱顺序可以降低训练时的局部顺序偏差，让优化更平稳。
docs = [line.strip() for line in open("input.txt", encoding="utf-8") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# -----------------------------
# 字符级 Tokenizer
# -----------------------------
# 这里使用最直观的字符级词表：把出现过的字符去重排序。
# 模型无法直接处理字符串，只能处理整数 token id。
uchars = sorted(set("".join(docs)))

# 额外引入 BOS（序列边界 token），用于提示"从哪里开始/何时结束"。
# 在推理阶段，若再次采到 BOS，就把它当作停止信号。
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# 预建映射表，避免频繁做线性查找（O(V)）。
stoi = {ch: i for i, ch in enumerate(uchars)}


# -----------------------------
# 自动求导核心：Value
# -----------------------------
# 这是最小 autograd：每个标量都携带"值 + 梯度 + 依赖关系"。
# 反向传播时按拓扑逆序回传梯度，完整体现链式法则。
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        # data: 前向数值
        self.data = data
        # grad: 反向时 dLoss/dSelf
        self.grad = 0
        # _children: 本节点依赖的输入节点
        self._children = children
        # _local_grads: 本节点对每个 child 的局部导数
        self._local_grads = local_grads

    # ---------- 基础算子：加法 ----------
    # 线性层累加、残差连接都离不开加法。
    # 这里把局部导数直接写死为 (1, 1)。
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    # ---------- 基础算子：乘法 ----------
    # 点积、缩放、线性映射的核心都是乘法。
    # 乘法的局部导数分别是"对方的当前值"。
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # 这些运算是训练里最常用的"基础积木"：
    # exp/log 支撑 softmax 与 NLL，relu 提供非线性表达能力。
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    # 从 loss 出发反传：
    # 1) 先 DFS 建拓扑顺序
    # 2) 再逆序累计梯度
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# -----------------------------
# 模型规模与参数初始化
# -----------------------------
# 这是"能跑通"的最小 GPT 配置：
# 层数、宽度、上下文窗口、头数都很小，便于理解和调试。
n_layer = 4  # 原为1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head


# 创建参数矩阵：每个元素都是可求导 Value。
# 使用高斯随机初始化，避免神经元完全对称。
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


# 把全部参数放进 state_dict，结构和标准 GPT 骨架一致：
# embedding + attention/mlp block + lm_head。
state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

# 打平参数列表，优化器就可以统一遍历更新。
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# -----------------------------
# 基础网络函数
# -----------------------------
def linear(x, w):
    # 线性层 y = W x：
    # 每行权重与输入做点积，得到一个输出神经元。
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    # softmax 前先减最大值，是经典数值稳定技巧：
    # 不改变概率分布，但能显著降低上溢风险。
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    # RMSNorm 只按均方根缩放，不做均值中心化。
    # 在 Pre-Norm 结构下，这通常能让训练更稳。
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# -----------------------------
# GPT 前向（单步：给定当前 token 和位置，预测下一个 token 分布）
# -----------------------------
def gpt(token_id, pos_id, keys, values):
    # token embedding 负责"这个符号是什么"，
    # position embedding 负责"它在序列里第几位"。
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    # 先做归一化，让后续投影更稳定。
    x = rmsnorm(x)

    for li in range(n_layer):
        # ===== 1) Multi-Head Self-Attention =====
        # 残差支路：保留原始信息，避免每层都"推倒重来"。
        x_residual = x
        x = rmsnorm(x)

        # q/k/v 三个投影：
        # q 用来"提问"，k 用来"匹配"，v 是"被读取的内容"。
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        # 追加到 K/V cache：当前 token 以后会成为"历史上下文"的一部分。
        # 这也是自回归推理可做增量计算的关键。
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            # 第 h 个头只看自己那一段通道，
            # 多头并行相当于从多个"关系视角"读上下文。
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]

            # 相关性分数 = q · k / sqrt(d)
            # 除以 sqrt(d) 是为了控制方差，避免 softmax 太尖锐。
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # softmax 后得到注意力权重：非负、和为 1。
            attn_weights = softmax(attn_logits)

            # 用注意力权重对历史 value 做加权求和，
            # 得到当前头在"上下文阅读"后的输出。
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        # 多头拼接后做一次输出投影，再接残差。
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        # ===== 2) MLP =====
        # MLP 子层负责"特征变换"：
        # attention 擅长路由信息，MLP 擅长重编码信息。
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    # 最后映射到词表维度，得到每个 token 的 logits。
    # logits 经过 softmax 就是 next-token 概率分布。
    logits = linear(x, state_dict["lm_head"])
    return logits


# -----------------------------
# Adam 优化器
# -----------------------------
# Adam 会维护每个参数的一阶矩 m、二阶矩 v，
# 对 noisy gradient 通常比纯 SGD 更稳。
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)


# -----------------------------
# 训练循环
# -----------------------------
# 训练目标：让模型在每个位置都更偏向真实的"下一个 token"。
# 这里就是最标准的自回归语言建模（token-level NLL）。
num_steps = 1000
for step in range(num_steps):
    # 取一条样本并编码，首尾加 BOS 作为边界。
    # 训练对齐关系是：prefix -> next token。
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]

    # 限制上下文窗口，避免序列过长导致计算开销过大。
    n = min(block_size, len(tokens) - 1)

    # 每层维护自己的 K/V cache，按位置逐步累积上下文。
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    # teacher forcing：喂入真实前缀，监督真实下一个 token。
    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 取序列平均损失，便于不同长度样本之间横向比较。
    loss = (1 / n) * sum(losses)

    # 自动反传，得到每个参数的梯度。
    loss.backward()

    # Adam 更新 + 线性学习率衰减 + 梯度清零。
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2

        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0

    print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")


# -----------------------------
# 推理采样
# -----------------------------
# 推理从 BOS 开始，逐步生成，直到遇到 BOS 或达到长度上限。
# 这里使用概率采样而不是贪心，生成会更有多样性。
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)

        # temperature 缩放：温度低更保守，温度高更发散。
        probs = softmax([l / temperature for l in logits])

        # 按概率采样下一个 token，模拟"有随机性的续写"。
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break

        sample.append(uchars[token_id])

    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
