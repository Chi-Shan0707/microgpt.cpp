#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <list>
#include <cmath>
#include <map>
#include <random>
#include <limits>
#include <cassert>
#include <numeric>
using namespace std;

// ── 全局超参数 ────────────────────
struct Config {
    int vocab_size = 27;
    int n_embd     = 16;
    int n_head     = 4;
    int n_layer    = 2;
    int n_hidden   = 64;
    int block_size = 32;
    double lr      = 0.001;
} cfg;

struct Value; // 前置声明

// ── 内存池管理 ────────────────────

// 1. 参数池：存放 WTE, WPE, Weights。整个程序运行期间不销毁。
list<Value> param_pool;
// 2. 计算图池：存放中间计算结果。每一步训练（Step）结束后清空。
list<Value> graph_pool;

class Value
{
public:
    double data;
    double grad;
    vector<Value*> children;
    vector<double> local_grads;

    Value() : data(0.0), grad(0.0) {}
    explicit Value(double d) : data(d), grad(0.0) {}

    // 辅助函数：创建一个新节点到计算图池
    static Value* make_new(double d,
        const vector<Value*>& _children = {},
        const vector<double>& _grads = {}) {
        graph_pool.emplace_back();
        Value* v = &graph_pool.back();
        v->data = d;
        v->children = _children;
        v->local_grads = _grads;
        return v;
    }

    // ── 运算符重载 (全部操作指针) ──────────────────
    // 加法
    static Value* add(Value* a, Value* b) {
        return make_new(a->data + b->data, {a, b}, {1.0, 1.0});
    }

    // 乘法
    static Value* mul(Value* a, Value* b) {
        return make_new(a->data * b->data, {a, b}, {b->data, a->data});
    }

    // 减法
    static Value* sub(Value* a, Value* b) {
        return make_new(a->data - b->data, {a, b}, {1.0, -1.0});
    }

    // Log
    static Value* log(Value* a) {
        return make_new(std::log(a->data), {a}, {1.0 / a->data});
    }

    // Exp (用于 Softmax)
    static Value* exp(Value* a)
    {
        double e = std::exp(a->data);
        return make_new(e, {a}, {e});
    }

    // Relu
    static Value* relu(Value* a)
    {
        double d = (a->data > 0) ? a->data : 0.0;
        double g = (a->data > 0) ? 1.0 : 0.0;
        return make_new(d, {a}, {g});
    }

    // 反向传播
    void backward() {
        vector<Value*> topo;
        set<Value*> visited;
        build_topo(this, topo, visited);

        this->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Value* v = *it;
            for (size_t i = 0; i < v->children.size(); ++i) {
                // 【核心修复】这里不再涉及对象拷贝，操作的永远是稳定的指针
                v->children[i]->grad += v->grad * v->local_grads[i];
            }
        }
    }

private:
    void build_topo(Value* v, vector<Value*>& topo, set<Value*>& visited) {
        if (visited.find(v) != visited.end()) return;
        visited.insert(v);
        for (auto child : v->children) build_topo(child, topo, visited);
        topo.push_back(v);
    }
};

// ── 容器类 (现在存储指针 Value*) ────────────────
class Vector {
public:
    vector<Value*> data;

    // 预分配大小
    void resize(size_t n, Value* val = nullptr) {
        data.resize(n, val);
    }
    size_t size() const { return data.size(); }

    Value* operator[](size_t i) const {
        return data[i];
    }
    //传回指针，且这里是浅拷贝（告诉别人这个门牌号），不过我们承诺这个是const，我们不更改其中数据
    Value*& operator[](size_t i) {
        return data[i];
    }
    //传回引用，可读写；且这里是零拷贝，我们直接可以针对这个【记录门牌号】的变量动手
};

class Matrix {
public:
    vector<vector<Value*>> data;
    size_t row, col;

    // 矩阵乘向量
    Vector operator*(const Vector& vec) const {
        Vector result;
        result.resize(row);
        for (size_t i = 0; i < row; ++i) {
            // 初始化累加器（注意：需要创建一个常数0节点）
            Value* sum = Value::make_new(0.0);
            for (size_t j = 0; j < col; ++j) {
                Value* prod = Value::mul(data[i][j], vec[j]);
                sum = Value::add(sum, prod);
            }
            result[i] = sum;
        }
        return result;
    }
};

Vector linear(const Matrix& weights, const Vector& input)
{
    return weights * input;
}

// softmax：对每个 logit 节点做 exp，累加，再除以总和
// 返回的每个 Value* 都连接在计算图中，梯度可以反传
Vector softmax(const Vector& logits)
{
    size_t n = logits.size();
    // 数值稳定性：找最大值（只用 data，不参与梯度）
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i)
        max_val = max(max_val, logits[i]->data);

    // 计算 exp(logit - max)，shift 节点的梯度直接传给 logit[i]
    vector<Value*> exps(n);
    for (size_t i = 0; i < n; ++i) {
        Value* shifted = Value::make_new(logits[i]->data - max_val, {logits[i]}, {1.0});
        exps[i] = Value::exp(shifted);
    }

    // 求和
    Value* sum_val = Value::make_new(0.0);
    for (size_t i = 0; i < n; ++i)
        sum_val = Value::add(sum_val, exps[i]);

    // 归一化：p[i] = exp[i] / sum
    Vector result;
    result.resize(n);
    for (size_t i = 0; i < n; ++i) {
        double ei = exps[i]->data;
        double s  = sum_val->data;
        // dc/d(exp[i]) = 1/s, dc/d(sum) = -exp[i]/s^2
        result[i] = Value::make_new(ei / s, {exps[i], sum_val}, {1.0 / s, -ei / (s * s)});
    }
    return result;
}

// 逐元素 relu，返回连接计算图的 Value*
Vector relu(const Vector& input)
{
    Vector result;
    result.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        result[i] = Value::relu(input[i]);
    return result;
}

// 向量点积：a·b = Σ a[i]->data * b[i]->data（返回 double，不入图）
double dot(const Vector& a, const Vector& b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i]->data * b[i]->data;
    return sum;
}

// 标量乘向量：s * v（标量不入图）
Vector scale(double s, const Vector& v)
{
    Vector result;
    result.resize(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = Value::mul(v[i], Value::make_new(s));
    return result;
}

// 向量逐元素加法：a + b
Vector add(const Vector& a, const Vector& b)
{
    Vector result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = Value::add(a[i], b[i]);
    return result;
}

Vector resnorm(const Vector& input)
{
    double sum_sq = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
        sum_sq += input[i]->data * input[i]->data;
    double norm = sqrt(sum_sq);

    Vector result;
    result.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        result[i] = Value::make_new(input[i]->data / norm, {input[i]}, {1.0 / norm});
    return result;
}


struct AttentionBlock
{
    Matrix wq, wk, wv; // 权重矩阵
    size_t n_embd, n_head, head_dim;
    Vector forward(const Vector& x, vector<Vector>& keys, vector<Vector>& values)
    {
        // ── 第一步：投影 Q / K / V ──────────────────────────────
        Vector q = wq * x;  // 当前 token："要查什么"
        Vector k = wk * x;  // 当前 token："能提供什么"
        Vector v = wv * x;  // 当前 token："实际内容"

        // 把本次的 k, v 追加到历史缓存（KV cache）
        keys.push_back(k);
        values.push_back(v);

        // ── 第二步：计算每个历史位置的 attention score ─────────
        // score[i] = dot(q, keys[i]) / sqrt(head_dim)
        size_t seq_len = keys.size();
        Vector scores;
        scores.resize(seq_len);
        double scale_factor = 1.0 / sqrt((double)head_dim);
        for (size_t i = 0; i < seq_len; ++i)
            scores[i] = Value::make_new(dot(q, keys[i]) * scale_factor);

        // ── 第三步：softmax → 权重（和为 1）───────────────────
        Vector weights = softmax(scores);

        // ── 第四步：加权求和 values ────────────────────────────
        // output = Σ weights[i] * values[i]
        Vector output;
        output.resize(n_embd, Value::make_new(0.0));
        for (size_t i = 0; i < seq_len; ++i)
            output = add(output, scale(weights[i]->data, values[i]));

        return output;
    }
};

struct MLPBlock
//这个写起来比AttentionBlock简单，因为它只有两次线性变换
{
    Matrix w1, w2; // 权重矩阵
    size_t n_embd, n_hidden;
    Vector forward(const Vector& x)
    {
        Vector hidden = relu(linear(w1, x));
        /**
         * x = relu(linear(w1, x))
         *
         * 这样会报错，因为我们和编译器说了x不被更改
         * 这也体现了一个好习惯：函数参数用 const& 表达"我只读这个输入"，内部计算结果用局部变量承接。
         */
        return linear(w2, hidden);           // hidden (n_hidden) → output (n_embd)
    }
};

class Tokenize
{
public:
    vector<char> vocab;        // 词表：下标即 token_id，值为对应字符
    map<char, int> char_to_id; // 字符 → token_id 的映射
    int BOS;                   // Begin-Of-Sequence 特殊 token 的 id

    vector<int> encode(const string& text)
    {
        vector<int> tokens;
        for (char ch : text) {
            auto it = char_to_id.find(ch);
            if (it != char_to_id.end()) {
                tokens.push_back(it->second); // 转换为 token_id
            } else {
                tokens.push_back(BOS); // 未知字符用 BOS 代替
            }
        }
        return tokens;
    }

    string decode(const vector<int>& tokens)
    {
        string text;
        for (int token_id : tokens) {
            if (token_id >= 0 && token_id < (int)vocab.size()) {
                text += vocab[token_id]; // token_id → 字符
            } else {
                text += '?'; // 无效 id 用 '?' 代替
            }
        }
        return text;
    }
};

struct GPT
{
    // ── 嵌入层参数 ────────────────────────────────────────────────────
    Matrix wte;      // Token  Embedding：词表大小 × n_embd
                     //   作用：把 token_id（整数）查表变成 n_embd 维向量
                     //   例：token_id=5 → wte 第5行 → 一个 n_embd 维向量

    Matrix wpe;      // Position Embedding：最大序列长度 × n_embd
                     //   作用：把位置编号（0,1,2,...）变成 n_embd 维向量
                     //   例：pos_id=3 → wpe 第3行 → 一个 n_embd 维向量

    Matrix lm_head;  // 语言模型输出头：vocab_size × n_embd
                     //   作用：把最终隐藏状态映射回词表大小的 logits
                     //   例：n_embd 维向量 → vocab_size 维 logits → softmax → 概率分布

    // ── Transformer 层（每层配对一个 Attention + 一个 MLP）────────────
    vector<AttentionBlock> attn_blocks; // N 层 Attention，每层调用 AttentionBlock::forward
    vector<MLPBlock>       mlp_blocks;  // N 层 MLP，与 attn_blocks 一一对应

    // ── 前向传播 ──────────────────────────────────────────────────────
    // 工作流程：
    //   步骤1  Embedding：x = wte[token_id] + wpe[pos_id]
    //   步骤2  逐层串行执行（共 N 层）：
    //            x = AttentionBlock.forward(x, keys, values)
    //            x = MLPBlock.forward(x)
    //   步骤3  输出头：logits = lm_head * x   (vocab_size 维)
    Vector forward(int token_id, int pos_id, vector<Vector>& keys, vector<Vector>& values)
    {
        // 步骤1：Embedding 层——直接取矩阵对应行（查表，不是矩阵乘法）
        // wte.data[token_id] 就是词表第 token_id 行，存储 Value* 指针
        Vector tok_emb; tok_emb.data = wte.data[token_id]; // Token Embedding
        Vector pos_emb; pos_emb.data = wpe.data[pos_id];   // Position Embedding
        Vector x = add(tok_emb, pos_emb);                  // x = tok_emb + pos_emb

        // 步骤2：逐层 Attention → MLP（串行！不是并行！）
        for (size_t i = 0; i < attn_blocks.size(); ++i)
        {
            x = attn_blocks[i].forward(x, keys, values); // Attention：聚合历史信息
            x = mlp_blocks[i].forward(x);                // MLP：对聚合结果做非线性变换
            // 注：完整实现还需在此处加 RMSNorm + 残差连接
        }

        // 步骤3：输出头，得到 vocab_size 维 logits
        return lm_head * x;
    }

    // 收集模型所有可训练参数的指针，供优化器使用
    vector<Value*> params()
    {
        vector<Value*> ps;
        // 辅助 lambda：把一个 Matrix 的所有 Value* 追加进 ps
        auto add_matrix = [&](Matrix& m) {
            for (auto& row : m.data)
                for (auto& val : row)
                    ps.push_back(val);
        };
        add_matrix(wte);
        add_matrix(wpe);
        add_matrix(lm_head);
        for (auto& blk : attn_blocks) {
            add_matrix(blk.wq);
            add_matrix(blk.wk);
            add_matrix(blk.wv);
        }
        for (auto& blk : mlp_blocks) {
            add_matrix(blk.w1);
            add_matrix(blk.w2);
        }
        return ps;
    }
};

void SGD_step(vector<Value*>& params, double lr)
{
    for (auto& param : params) {
        param->data -= lr * param->grad; // 更新参数：θ = θ - lr * dL/dθ
        param->grad = 0.0; // 清零梯度，为下一轮计算做准备
    }
}

void train(GPT& model, const vector<string>& data, Tokenize& tokenizer,
           int num_steps, int log_interval = 100)
{
    for (int step = 0; step < num_steps; ++step)
    {
        // 【关键】每一步清空计算图，但保留参数（param_pool 不清空）
        graph_pool.clear();

        // 1. 采样 + 构建 token 序列：BOS + encode(doc) + BOS
        string doc = data[step % data.size()];
        vector<int> tokens = {tokenizer.BOS};          // 前置 BOS
        vector<int> encoded = tokenizer.encode(doc);
        tokens.insert(tokens.end(), encoded.begin(), encoded.end());
        tokens.push_back(tokenizer.BOS);               // 后置 BOS

        // 2. 前向传播：对每个位置预测下一个 token
        vector<Vector> keys, values; // KV cache（从空开始，每个位置追加）
        Value* total_loss = Value::make_new(0.0);
        int count = 0;

        for (int pos = 0; pos < (int)tokens.size() - 1; ++pos)
        {
            Vector logits = model.forward(tokens[pos], pos, keys, values);
            Vector probs  = softmax(logits);

            // Cross Entropy Loss: -log(probs[target])
            int target       = tokens[pos + 1];
            Value* prob      = probs[target];
            Value* log_prob  = Value::log(prob);
            total_loss = Value::sub(total_loss, log_prob); // loss += -log(p)
            count++;
        }

        // 3. 平均损失 = total_loss / count
        Value* mean_loss = Value::mul(total_loss, Value::make_new(1.0 / count));

        // 4. 反向传播
        mean_loss->backward();

        // 5. SGD 更新参数（在梯度算完之后立刻更新）
        auto ps = model.params();
        SGD_step(ps, cfg.lr);

        if (step % log_interval == 0)
        {
            cout << "Step " << step << ", Loss: " << mean_loss->data << endl;
        }
    }
}

// 按概率分布随机采样一个 token_id
// temperature 应在调用 softmax 前除到 logits 上；
// 这里 probs 已经是 softmax 结果，直接加权随机采样
int sample(const Vector& probs, double /*temperature*/)
{
    static mt19937 rng(42); // 固定种子，方便复现
    vector<double> p;
    for (size_t i = 0; i < probs.size(); ++i)
        p.push_back(probs[i]->data);
    discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}

// tokenizer 以参数传入，不放进 GPT 里——职责分离：GPT 只管向量计算
string generate(GPT& model, Tokenize& tokenizer, int max_len = 16, double temp = 0.5)
{
    vector<Vector> keys, values; // KV cache（从空开始）

    int token_id = tokenizer.BOS; // 从 BOS 开始生成
    vector<int> generated;

    for (int pos = 0; pos < max_len; ++pos)
    {
        Vector logits  = model.forward(token_id, pos, keys, values);
        Vector probs   = softmax(logits);
        token_id       = sample(probs, temp);

        if (token_id == tokenizer.BOS)
            break; // 遇到 BOS（当作 EOS）停止生成

        generated.push_back(token_id);
    }
    return tokenizer.decode(generated);
}

// 辅助函数：创建 rows×cols 的随机初始化矩阵（小随机数）
// 参数存入 param_pool，整个训练期间不销毁
Matrix rand_matrix(int rows, int cols)
{
    static mt19937 rng(123);
    normal_distribution<double> dist(0.0, 0.02); // 均值0、标准差0.02 的小随机数
    Matrix m;
    m.row = rows;
    m.col = cols;
    m.data.resize(rows, vector<Value*>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            param_pool.emplace_back(dist(rng));
            m.data[i][j] = &param_pool.back();
        }
    return m;
}

int main()
{
    // ═══ 1. 读取训练数据 ═══════════════════════════════════════
    vector<string> data;
    ifstream fin("input.txt");
    if (fin.is_open()) {
        string line;
        while (getline(fin, line))
            if (!line.empty()) data.push_back(line);
        fin.close();
    }
    if (data.empty()) {
        // 没有 input.txt 就用硬编码示例（简单英文名字）
        data = {"emma", "olivia", "ava", "luna", "sophia", "mia", "harper",
                "isabella", "amelia", "evelyn", "abigail", "ella", "charlotte"};
    }
    cout << "Training samples: " << data.size() << endl;

    // ═══ 2. 构建 Tokenizer（从训练数据中收集所有字符）═════════════
    set<char> charset;
    for (auto& s : data)
        for (char c : s)
            charset.insert(c);

    Tokenize tokenizer;
    for (char c : charset) {
        tokenizer.char_to_id[c] = (int)tokenizer.vocab.size();
        tokenizer.vocab.push_back(c);
    }
    tokenizer.BOS = (int)tokenizer.vocab.size(); // BOS id = 词表末尾
    tokenizer.vocab.push_back('#');               // BOS 占位

    // ═══ 3. 更新配置 ═══════════════════════════════════════════
    cfg.vocab_size = (int)tokenizer.vocab.size();
    cout << "Vocab size: " << cfg.vocab_size << " (" << cfg.vocab_size - 1 << " chars + BOS)" << endl;

    // ═══ 4. 初始化 GPT 模型（随机权重）═══════════════════════════
    GPT model;
    model.wte     = rand_matrix(cfg.vocab_size, cfg.n_embd);  // 词表大小 × n_embd
    model.wpe     = rand_matrix(cfg.block_size, cfg.n_embd);  // 最大序列长 × n_embd
    model.lm_head = rand_matrix(cfg.vocab_size, cfg.n_embd);  // 词表大小 × n_embd

    int head_dim = cfg.n_embd / cfg.n_head;
    for (int i = 0; i < cfg.n_layer; ++i) {
        // 每层一个 AttentionBlock
        AttentionBlock attn;
        attn.n_embd   = cfg.n_embd;
        attn.n_head   = cfg.n_head;
        attn.head_dim = head_dim;
        attn.wq = rand_matrix(cfg.n_embd, cfg.n_embd);
        attn.wk = rand_matrix(cfg.n_embd, cfg.n_embd);
        attn.wv = rand_matrix(cfg.n_embd, cfg.n_embd);
        model.attn_blocks.push_back(attn);

        // 每层一个 MLPBlock
        MLPBlock mlp;
        mlp.n_embd   = cfg.n_embd;
        mlp.n_hidden = cfg.n_hidden;
        mlp.w1 = rand_matrix(cfg.n_hidden, cfg.n_embd);  // n_embd → n_hidden
        mlp.w2 = rand_matrix(cfg.n_embd, cfg.n_hidden);  // n_hidden → n_embd
        model.mlp_blocks.push_back(mlp);
    }
    cout << "Model initialized: " << cfg.n_layer << " layers, "
         << cfg.n_embd << " dim, " << cfg.n_head << " heads" << endl;

    // ═══ 5. 训练 ═══════════════════════════════════════════════
    int num_steps = 100;
    cout << "\nStarting training for " << num_steps << " steps..." << endl;
    train(model, data, tokenizer, num_steps, /*log_interval=*/10);

    // ═══ 6. 生成示例 ═══════════════════════════════════════════
    cout << "\n=== Generated samples ===" << endl;
    for (int i = 0; i < 5; ++i) {
        // temp=0.8: 比较保守的生成; temp=1.0: 完全按原始概率生成
        string result = generate(model, tokenizer, cfg.block_size, 0.8);
        cout << "  [" << i << "] " << result << endl;
    }

    return 0;
}
