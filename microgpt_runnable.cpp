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
#include <algorithm>
#include <iomanip>

using namespace std;

// ── 全局超参数 ────────────────────
struct Config {
    int vocab_size = 27;
    int n_embd     = 16;
    int n_head     = 4;
    int n_layer    = 4;
    int n_hidden   = 64;
    int block_size = 32;
    int training_steps = 1000;
    double lr      = 0.005;
    int num_samples = 20;
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
    static Value* make_new(double d, const vector<Value*>& _children = {}, const vector<double>& _grads = {}) {
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
    static Value* exp(Value* a) {
        double e = std::exp(a->data);
        return make_new(e, {a}, {e});
    }

    // Relu
    static Value* relu(Value* a) {
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
    size_t size() const { 
        return data.size(); 
    }



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

// ── 基础算子 (适配 Value*) ────────────────
Vector add(const Vector& a, const Vector& b) {
    Vector res; res.resize(a.size());
    for(size_t i=0; i<a.size(); ++i) res[i] = Value::add(a[i], b[i]);
    return res;
}

Vector scale(double s, const Vector& v) {
    Vector res; res.resize(v.size());
    Value* s_node = Value::make_new(s); // 常数节点
    for(size_t i=0; i<v.size(); ++i) res[i] = Value::mul(s_node, v[i]);
    return res;
}

Vector softmax(const Vector& logits) {
    Vector res; res.resize(logits.size());
    double max_val = -1e9;
    for(auto v : logits.data) if(v->data > max_val) max_val = v->data;
    
    // 这一步虽然不反传 max_val 的梯度，但为了数值稳定性，通常只做 data 计算
    Value* max_node = Value::make_new(max_val); // 常数
    
    vector<Value*> exps;
    Value* sum_exp = Value::make_new(0.0);
    
    for(auto v : logits.data) {
        Value* shifted = Value::sub(v, max_node);
        Value* e = Value::exp(shifted);
        exps.push_back(e);
        sum_exp = Value::add(sum_exp, e);
    }
    
    // 此时 sum_exp 是整个图的一部分，包含完整的梯度链
    // 既然没有除法算子，我们用 x * (sum^-1) 或者扩充除法
    // 这里我们简单扩充一个 Value::div 辅助
    for(size_t i=0; i<logits.size(); ++i) {
        // div 实现： a * (b^-1)
        Value* inv_sum = Value::make_new(1.0 / sum_exp->data, {sum_exp}, {-1.0 / (sum_exp->data * sum_exp->data)});
        res[i] = Value::mul(exps[i], inv_sum);
    }
    return res;
}

Vector relu(const Vector& input) {
    Vector res; res.resize(input.size());
    for(size_t i=0; i<input.size(); ++i) res[i] = Value::relu(input[i]);
    return res;
}

// ── 神经网络模块 ────────────────
struct AttentionBlock {
    Matrix wq, wk, wv, wo; // 加个 wo (output projection) 比较完整，或者省略
    size_t n_embd, n_head, head_dim;

    Vector forward(const Vector& x, vector<Vector>& keys, vector<Vector>& values) {
        Vector q = wq * x;
        Vector k = wk * x;
        Vector v = wv * x;
        
        keys.push_back(k);
        values.push_back(v);
        
        // Multi-head logic simplified to 1 head for clarity if dims match, 
        // or strictly follow splitting. 为了配合你的代码逻辑，这里假设 n_head 维度在内部处理
        // 你的代码原逻辑其实类似于 Single Head Attention (所有维度一起算)，这在微型模型里没问题。
        
        // 计算 Score
        size_t seq_len = keys.size();
        Vector scores; scores.resize(seq_len);
        double scale_factor = 1.0 / sqrt((double)head_dim);
        Value* scale_node = Value::make_new(scale_factor);

        for(size_t t=0; t<seq_len; ++t) {
            // Dot product q · k_t
            Value* dot = Value::make_new(0.0);
            for(size_t i=0; i<q.size(); ++i) {
                dot = Value::add(dot, Value::mul(q[i], keys[t][i]));
            }
            scores[t] = Value::mul(dot, scale_node);
        }
        
        Vector weights = softmax(scores);
        
        // Weighted Sum
        Vector output; output.resize(n_embd);
        for(size_t i=0; i<n_embd; ++i) output[i] = Value::make_new(0.0);
        
        for(size_t t=0; t<seq_len; ++t) {
            Value* w = weights[t];
            for(size_t i=0; i<n_embd; ++i) {
                output[i] = Value::add(output[i], Value::mul(w, values[t][i]));
            }
        }
        return output; 
    }
};

struct MLPBlock 
{
    Matrix w1, w2;

    Vector forward(const Vector& x) {
        return w2 * relu(w1 * x);
    }
};

// ── 模型类 ────────────────
struct GPT {
    Matrix wte, wpe, lm_head;
    vector<AttentionBlock> attn_blocks;
    vector<MLPBlock> mlp_blocks;

    Vector forward(int token_id, int pos_id,
                   vector<vector<Vector>>& layer_keys,
                   vector<vector<Vector>>& layer_values)
    {
        Vector tok; tok.data = wte.data[token_id]; 
        Vector pos; pos.data = wpe.data[pos_id];
        Vector x = add(tok, pos);

        for(size_t i=0; i<attn_blocks.size(); ++i) {
            // Attention + Residual
            Vector attn_out = attn_blocks[i].forward(x, layer_keys[i], layer_values[i]);
            x = add(x, attn_out); 
            
            // MLP + Residual
            Vector mlp_out = mlp_blocks[i].forward(x);
            x = add(x, mlp_out);
        }
        return lm_head * x;
    }

    // 收集所有参数指针 (指向 param_pool)
    vector<Value*> params() 
    {
        vector<Value*> ps;
        auto add_mat = [&](Matrix& m) {
            for(auto& r : m.data) for(auto& v : r) ps.push_back(v);
        };
        add_mat(wte); add_mat(wpe); add_mat(lm_head);
        for(auto& b : attn_blocks) 
        { 
            add_mat(b.wq); 
            add_mat(b.wk); 
            add_mat(b.wv); 
        }
        for(auto& b : mlp_blocks) 
        { 
            add_mat(b.w1); 
            add_mat(b.w2); 
        }
        return ps;
    }
};

// ── 初始化工具 (放入 param_pool) ────────────────
Matrix rand_matrix(int rows, int cols) {
    static mt19937 rng(42);
    normal_distribution<double> dist(0.0, 0.2); // 稍微加大初始化方差
    Matrix m; m.row = rows; m.col = cols;
    m.data.resize(rows, vector<Value*>(cols));
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            // 【关键】参数存入永久池 param_pool
            param_pool.emplace_back(dist(rng));
            m.data[i][j] = &param_pool.back();
        }
    }
    return m;
}

// ── Tokenizer ────────────────
class Tokenize 
{
public:
    vector<char> vocab;
    map<char, int> char_to_id;
    int BOS;
    vector<int> encode(const string& text) {
        vector<int> tokens;
        for (char ch : text) {
            if (char_to_id.count(ch)) tokens.push_back(char_to_id[ch]);
            else tokens.push_back(BOS);
        }
        return tokens;
    }
    string decode(const vector<int>& tokens) {
        string s;
        for(int id : tokens) if(id >=0 && id < (int)vocab.size() && id != BOS) s += vocab[id];
        return s;
    }
};
// 按照给定的概率分布进行随机采样
int sample(const Vector& probs) 
{
    // 建议在实际使用中把 42 换成 std::random_device{}() 以获得真正的随机生成
    static mt19937 rng(std::random_device{}()); 
    
    vector<double> p;
    for (size_t i = 0; i < probs.size(); ++i) {
        // 提取前向传播的 data 值
        p.push_back(probs[i]->data);
    }
    
    // discrete_distribution 会自动根据传入的权重数组进行轮盘赌采样
    discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}
// tokenizer 以参数传入，不放进 GPT 里——职责分离：GPT 只管向量计算
string generate(GPT& model, Tokenize& tokenizer, int max_len = 16, double temp = 0.5) 
{
    // 【关键】生成开始前，清空计算图，防止与上一轮训练的数据混淆
    graph_pool.clear();

    int n_layers = (int)model.attn_blocks.size();
    vector<vector<Vector>> layer_keys(n_layers), layer_values(n_layers); // 按层独立 KV cache
    int token_id = tokenizer.BOS; // 从 BOS 开始生成
    vector<int> generated;

    for (int pos = 0; pos < max_len; ++pos) 
    {
        // 1. 前向传播：模型会把中间结果放入 graph_pool，并更新 keys, values
        Vector logits = model.forward(token_id, pos, layer_keys, layer_values);
        
        // 2. 温度缩放 (Temperature Scaling)
        // 注意：必须在 softmax 之前，将 logits 除以 temperature
        Vector scaled_logits; 
        scaled_logits.resize(logits.size());
        for(size_t i = 0; i < logits.size(); ++i) {
            // temp 越小，分布越尖锐（越贪心）；temp 越大，分布越平缓（越多幻觉）
            scaled_logits[i] = Value::make_new(logits[i]->data / temp); 
        }

        // 3. 转化为概率
        Vector probs = softmax(scaled_logits);
        
        // 4. 根据概率采样下一个 token
        token_id = sample(probs);

        if (token_id == tokenizer.BOS) {
            break; // 遇到 BOS（当作 EOS）停止生成
        }

        generated.push_back(token_id);
        
        // 【警告】这里绝对不能调用 graph_pool.clear()！
        // 否则 KV cache 指向的内存会被释放，下一步生成将发生段错误。
    }
    
    // 【收尾】生成结束后，清空计算图，释放这 max_len 步产生的临时节点
    graph_pool.clear();
    
    return tokenizer.decode(generated);
}
// ── 主程序 ────────────────
int main() {
    // 1. Data
    freopen("input_names.txt", "r", stdin);
    vector<string> data;
    string line;
    while (getline(cin, line))
    {
        if (!line.empty()) data.push_back(line);
    }
    // 2. Tokenizer
    Tokenize tokenizer;
    set<char> chars;
    for(auto& s: data) for(char c: s) chars.insert(c);
    for(char c: chars) {
        tokenizer.char_to_id[c] = tokenizer.vocab.size();
        tokenizer.vocab.push_back(c);
    }
    tokenizer.BOS = tokenizer.vocab.size();
    tokenizer.vocab.push_back('#');
    cfg.vocab_size = tokenizer.vocab.size();

    // 3. Init Model
    GPT model;
    model.wte = rand_matrix(cfg.vocab_size, cfg.n_embd);
    model.wpe = rand_matrix(cfg.block_size, cfg.n_embd);
    model.lm_head = rand_matrix(cfg.vocab_size, cfg.n_embd); // Fix dimensions
    
    // Init blocks
    int head_dim = cfg.n_embd / cfg.n_head;
    for(int i=0; i<cfg.n_layer; ++i) {
        AttentionBlock attn;
        attn.n_embd = cfg.n_embd; attn.n_head = cfg.n_head; attn.head_dim = head_dim;
        attn.wq = rand_matrix(cfg.n_embd, cfg.n_embd);
        attn.wk = rand_matrix(cfg.n_embd, cfg.n_embd);
        attn.wv = rand_matrix(cfg.n_embd, cfg.n_embd);
        model.attn_blocks.push_back(attn);
        
        MLPBlock mlp;
        mlp.w1 = rand_matrix(cfg.n_hidden, cfg.n_embd);
        mlp.w2 = rand_matrix(cfg.n_embd, cfg.n_hidden);
        model.mlp_blocks.push_back(mlp);
    }

    // 4. Train Loop
    cout << "Start training..." << endl;
    for(int step=0; step< cfg.training_steps; ++step) {
        // 【关键】每一步清空计算图，但保留参数
        graph_pool.clear();
        
        string doc = data[step % data.size()];
        vector<int> tokens = {tokenizer.BOS};
        auto enc = tokenizer.encode(doc);
        tokens.insert(tokens.end(), enc.begin(), enc.end());
        tokens.push_back(tokenizer.BOS);
        
        int n_layers = (int)model.attn_blocks.size();
        vector<vector<Vector>> layer_keys(n_layers), layer_values(n_layers);
        Value* total_loss = Value::make_new(0.0);
        int count = 0;

        for(size_t pos=0; pos<tokens.size()-1; ++pos) {
            Vector logits = model.forward(tokens[pos], pos, layer_keys, layer_values);
            Vector probs = softmax(logits);
            
            // Cross Entropy Loss: -log(probs[target])
            int target = tokens[pos+1];
            Value* prob = probs[target];
            Value* log_prob = Value::log(prob);
            total_loss = Value::sub(total_loss, log_prob); // loss += -log(p)
            count++;
        }
        
        // Mean Loss
        Value* mean_loss = Value::mul(total_loss, Value::make_new(1.0/count));
        
        // Backward
        mean_loss->backward();
        
        // SGD
        for(auto p : model.params()) {
            p->data -= 0.05 * p->grad; // Learning Rate
            p->grad = 0.0; // Zero grad
        }
        
        if(step % 10 == 0) {
            cout << "Step " << step << " Loss: " << mean_loss->data << endl;
        }
    }
    // ═══ 6. 生成示例 ═══════════════════════════════════════════
    cout << "\n=== Generated samples ===" << endl;
    for (int i = 0; i < cfg.num_samples; ++i) {
        // temp=0.5: 比较保守的生成; temp=1.0: 完全按原始概率生成
        string result = generate(model, tokenizer, cfg.block_size, 0.8);
        cout << "  [" << i << "] " << result << endl;
    }
    return 0;
}