# MicroGPT C++ 复刻 TODO LIST

> 遵照 `microgpt.py` 的逻辑顺序，逐模块拆解。
> 风格原则：**现代 C++（C++17）、简洁、好用，不堆砌冷门语法**。

---

## 整体数据流（LLM 视角）

```
原始文本
  → Tokenizer（字符级词表）
    → Embedding（token + position）
      → [Attention + MLP] × n_layer（Transformer Block）
        → lm_head（映射到词表）
          → Softmax → NLL Loss
            → Backward（autograd）
              → Adam 更新参数
                → 推理采样
```

---

## 模块一：自动微分引擎 `Value`

> 对应 py：`class Value`

**需要实现：**
- `struct Value`，持有 `double data`、`double grad`
- 支持 `+ - * / pow exp log relu` 运算
- `backward()` 方法：DFS 拓扑排序 + 逆序链式反传

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 节点存储 | `std::shared_ptr<Value>`，避免手动管理生命周期 |
| 反向传播回调 | 在每个节点存一个 `std::function<void()> _backward`，构造时就写好梯度累积逻辑 |
| 拓扑排序 | `std::vector<Value*>` + `std::unordered_set<Value*>` 做 DFS visited |
| 依赖关系 | `std::vector<std::shared_ptr<Value>> _children` |

**示例接口：**
```cpp
struct Value {
    double data, grad = 0;
    std::vector<std::shared_ptr<Value>> _children;
    std::function<void()> _backward = [] {};

    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value relu() const;
    Value log() const;
    Value exp() const;
    void backward();
};
using Val = std::shared_ptr<Value>;
Val make_val(double data);
```

---

## 模块二：数据准备 + Tokenizer

> 对应 py：读取 `input.txt`、构建词表、`stoi` 映射

**需要实现：**
- 读取文件所有行（已有 `readAllLines`）
- 去重排序建词表（`uchars`），加入 `BOS` token
- 构建 `char → int` 映射（`stoi`）和 `int → char` 映射（`itos`）

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 读取文件行 | `std::ifstream` + `std::getline` + `std::vector<std::string>` |
| 去重排序 | `std::set<char>` 收集 → 转为 `std::vector<char>` |
| 字符映射 | `std::unordered_map<char, int> stoi` 和 `std::vector<char> itos` |
| 随机打乱 | `std::shuffle(docs.begin(), docs.end(), 一个 mt19937 引擎)` |

**示例接口：**
```cpp
struct Tokenizer {
    std::vector<char> itos;
    std::unordered_map<char, int> stoi;
    int BOS, vocab_size;
    
    Tokenizer(const std::vector<std::string>& docs);
    std::vector<int> encode(const std::string& s) const;
};
```

---

## 模块三：参数矩阵 `state_dict`

> 对应 py：`matrix()` 函数 + `state_dict` 字典

**需要实现：**
- 用高斯随机数初始化权重矩阵
- 用字符串 key 管理所有参数（方便按名字访问，对应 py 的 `state_dict`）
- 提取所有参数的平坦列表（用于优化器遍历）

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 权重矩阵类型 | `using Matrix = std::vector<std::vector<Val>>;` |
| 参数字典 | `std::unordered_map<std::string, Matrix> state_dict` |
| 高斯随机数 | `std::normal_distribution<double>` + `std::mt19937` |
| 平坦参数列表 | `std::vector<Val> params`，遍历 state_dict 逐一 push_back |

**示例接口：**
```cpp
Matrix make_matrix(int nout, int nin, double std = 0.08);
// state_dict["wte"] = make_matrix(vocab_size, n_embd);
```

---

## 模块四：基础网络层函数

> 对应 py：`linear()` / `softmax()` / `rmsnorm()`

**需要实现：**
- `linear(x, W)`：矩阵-向量乘法，返回 `vector<Val>`
- `softmax(logits)`：减最大值 + exp + 归一，返回概率向量
- `rmsnorm(x)`：均方根归一化

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 向量类型 | `using Vec = std::vector<Val>;`，贯穿全文 |
| 函数签名 | 普通函数，参数和返回值都是 `Vec` 或 `const Vec&` |
| softmax 数值稳定 | `std::max_element` 找最大值再偏移 |

**示例接口：**
```cpp
using Vec = std::vector<Val>;
Vec linear(const Vec& x, const Matrix& w);
Vec softmax(const Vec& logits);
Vec rmsnorm(const Vec& x);
```

---

## 模块五：GPT 前向传播

> 对应 py：`def gpt(token_id, pos_id, keys, values)`

这是整个项目最复杂的部分，包含：
1. Token + Position Embedding 相加
2. 对每层执行：
   - RMSNorm → Q/K/V 投影
   - 多头注意力（分头 → 点积打分 → softmax → 加权求和 → 拼接）
   - 输出投影 → 残差连接
   - RMSNorm → MLP（fc1 → ReLU → fc2） → 残差连接
3. lm_head 投影输出 logits

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| KV Cache | `std::vector<std::vector<Vec>> keys, values`（外层是层数，内层是历史时间步） |
| 分头切片 | 用下标范围 `Vec(q.begin() + hs, q.begin() + hs + head_dim)` |
| 残差加法 | range-based for + index，或写一个简短的 `vec_add(a, b)` 工具函数 |
| 整体封装 | 建议封装成 `struct GPT { Vec forward(int token_id, int pos_id, ...); }` |

---

## 模块六：Adam 优化器

> 对应 py：Adam 的 `m / v` 更新 + 线性学习率衰减

**需要实现：**
- 存储每个参数的一阶矩 `m`、二阶矩 `v`
- 按步骤更新参数，含偏置修正 + lr 衰减
- 每步结束后梯度清零

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 状态存储 | `std::vector<double> m, v`，与 `params` 等长 |
| 整体封装 | `struct Adam { void step(int t); }` 持有 `params` 引用 |

**示例接口：**
```cpp
struct Adam {
    std::vector<Val>& params;
    std::vector<double> m, v;
    double lr, beta1, beta2, eps;
    
    Adam(std::vector<Val>& params, double lr = 0.01);
    void step(int t, double lr_decay = 1.0);
    void zero_grad();
};
```

---

## 模块七：训练循环

> 对应 py：`for step in range(num_steps):`

**需要实现：**
- 按步骤取一条样本、编码为 token 序列
- 初始化 KV Cache
- 逐位置前向 → 取法 `softmax(logits)[target] → -log` → 累积 loss
- `loss.backward()` → Adam 更新

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 序列截断 | `std::min(block_size, (int)tokens.size() - 1)` |
| loss 累加 | `Val loss = make_val(0); loss = loss + loss_t;` |
| 进度输出 | `std::cout << "\rstep " << step << " loss " << loss->data` 配合 `std::flush` |

---

## 模块八：推理采样

> 对应 py：inference 段

**需要实现：**
- 从 BOS 开始，每步调用 `gpt()` 得到 logits
- 用 temperature 缩放 logits，再采样下一个 token
- 遇到 BOS 或达到 `block_size` 则停止
- 解码 token id 序列 → 字符串输出

**推荐 C++ 语法：**
| 功能 | 用法 |
|---|---|
| 概率采样 | `std::discrete_distribution<int>` + `std::mt19937` |
| temperature 缩放 | 直接对 logits 的 `data` 值除以 temperature（注意是 `Val`，需 `/= T`） |
| 结果输出 | `std::string result; result += tokenizer.itos[token_id];` |

---

## 推荐文件结构

```
MicroGPT/
├── input.txt             # 训练数据
├── generate_equations.cpp  # 已有
├── microgpt.cpp          # 主程序（main + 训练 + 推理）
├── value.hpp             # 模块一：Value autograd
├── tokenizer.hpp         # 模块二：Tokenizer
├── model.hpp             # 模块三~五：state_dict + 网络函数 + GPT
├── optimizer.hpp         # 模块六：Adam
└── TODOLIST.md           # 本文件
```

---

## 推进顺序建议

```
[x] 已有：readAllLines、wchar_t 字符集读取
[ ] 1. value.hpp：先跑通 f(x) = x*x，验证 backward() 结果
[ ] 2. tokenizer.hpp：读入 input.txt，打印词表和编码示例
[ ] 3. model.hpp：只写 linear/softmax/rmsnorm，单元测试
[ ] 4. model.hpp：实现 GPT 前向，先不训练，只看 logits 维度对不对
[ ] 5. optimizer.hpp：Adam
[ ] 6. microgpt.cpp：拼接训练循环，观察 loss 下降
[ ] 7. microgpt.cpp：加入推理采样，对比 py 版输出
```
