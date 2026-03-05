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

//  ──────────── global hyperparameters ────────────
struct Config {
    int num_vocab = 27;
    int dim_embd     = 16;
    int num_head     = 4;
    int num_layer    = 4;
    int dim_hidden   = 64;
    int dim_block = 32;
    int num_training_steps = 1000;
    double lr      = 0.005;
    int num_samples = 20;
} cfg;

struct Value;
list<Value> param_pool, graph_pool;
class Value
{
public:
    double data, grad;
    vector<Value*> children;
    vector<double> local_grads;
    Value() : data(0.0), grad(0.0) {}
    explicit Value(double d) : data(d), grad(0.0) {}
    static Value* make_new(double d, const vector<Value*>& _children = {}, const vector<double>& _grads = {}) {
        graph_pool.emplace_back();
        Value* v = &graph_pool.back();
        v->data = d;
        v->children = _children;
        v->local_grads = _grads;
        return v;
    }
    static Value* add(Value* a, Value* b) {  return make_new(a->data + b->data, {a, b}, {1.0, 1.0});}
    static Value* mul(Value* a, Value* b) {  return make_new(a->data * b->data, {a, b}, {b->data, a->data});}
    static Value* sub(Value* a, Value* b) {  return make_new(a->data - b->data, {a, b}, {1.0, -1.0});}
    static Value* log(Value* a) { return make_new(std::log(a->data), {a}, {1.0 / a->data});}
    static Value* exp(Value* a) {  double e = std::exp(a->data); return make_new(e, {a}, {e});}
    static Value* relu(Value* a) {
        double d = (a->data > 0) ? a->data : 0.0;
        double g = (a->data > 0) ? 1.0 : 0.0;
        return make_new(d, {a}, {g});
    }
    void backward() {
        vector<Value*> topo;
        set<Value*> visited;
        build_topo(this, topo, visited);
        this->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Value* v = *it;
            for (size_t i = 0; i < v->children.size(); ++i) v->children[i]->grad += v->grad * v->local_grads[i];
        }
    }
    void build_topo(Value* v, vector<Value*>& topo, set<Value*>& visited) {
        if (visited.find(v) != visited.end()) return;
        visited.insert(v);
        for (auto child : v->children) build_topo(child, topo, visited);
        topo.push_back(v);
    }
};
class Vector{
public:
    vector<Value*> data;
};
class Matrix{
public:
    vector<vector<Value*>> data;
    size_t row, col;
    Vector operator*(const Vector& vec) const {
        Vector result;
        (result.data).resize(row,nullptr);
        for(size_t i=0; i<row; ++i) {
            Value* sum = Value::make_new(0.0);
            for(size_t j=0; j<col; ++j) {
                Value* prod = Value::mul(data[i][j], vec.data[j]);
                sum = Value::add(sum, prod);
            }
            result.data[i] = sum;
        }
        return result;
    }
};
Vector add(const Vector& a, const Vector& b) {
    Vector res; res.data.resize(a.data.size(),nullptr);
    for(size_t i=0; i<a.data.size(); ++i) res.data[i] = Value::add(a.data[i], b.data[i]);
    return res;
}
Vector scale(double s, const Vector& v) {
    Vector res; res.data.resize(v.data.size(),nullptr);
    Value* s_node = Value::make_new(s); // 常数节点
    for(size_t i=0; i<v.data.size(); ++i) res.data[i] = Value::mul(s_node, v.data[i]);
    return res;
}
Vector softmax(const Vector& logits) {
    Vector res; res.data.resize(logits.data.size(),nullptr);
    double max_val = -1e9;
    for (auto v : logits.data) if (v->data > max_val) max_val = v->data;
    Value* max_node = Value::make_new(max_val); // 常数
    vector<Value*> exps;
    Value* sum_exp = Value::make_new(0.0);
    for (auto v : logits.data) {
        Value* shifted = Value::sub(v, max_node);
        Value* e = Value::exp(shifted);
        exps.push_back(e);
        sum_exp = Value::add(sum_exp, e);
    }
    for (auto i=0; i<logits.data.size(); ++i) {
        Value* inv_sum = Value::make_new(1.0 / sum_exp->data, {sum_exp}, {-1.0 / (sum_exp->data * sum_exp->data)});
        res.data[i] = Value::mul(exps[i], inv_sum);
    }
    return res;
}
Vector relu(const Vector& input){
    Vector res; res.data.resize(input.data.size(),nullptr);
    for(auto i = 0; i<input.data.size(); ++i) res.data[i] = Value::relu(input.data[i]); 
    return res;
}
struct AttentionBlock {
    Matrix Wq , Wk, Wv, Wo;
    size_t dim_embd, num_head, head_dim;
    Vector forward(const Vector& x, vector<Vector>& keys, vector<Vector>& values) {
        Vector q = Wq * x, k = Wk * x, v = Wv * x;
        keys.push_back(k);
        values.push_back(v);
        auto seq_len = keys.size();
        Vector scores; scores.data.resize(seq_len,nullptr);
        double scale_factor = 1.0 / sqrt((double)head_dim);
        Value* scale_node = Value::make_new(scale_factor);
        for(auto t=0; t<seq_len; ++t) {
            Value* dot = Value::make_new(0.0);
            for(auto i=0; i<q.data.size(); ++i) dot = Value::add(dot, Value::mul(q.data[i], keys[t].data[i]));
            scores.data[t] = Value::mul(dot, scale_node);
        }
        Vector weights = softmax(scores);
        Vector output; output.data.resize(dim_embd,nullptr);
        for(size_t i=0; i<dim_embd; ++i) output.data[i] = Value::make_new(0.0);
        for(size_t t=0; t<seq_len; ++t) {
            Value* w = weights.data[t];
            for(size_t i=0; i<dim_embd; ++i)     output.data[i] = Value::add(output.data[i], Value::mul(w, values[t].data[i]));
        }
        return output;
    }
};
struct MLPBlock{
    Matrix W1,W2;
    Vector forward(const Vector& x) {
        return W2 * relu(W1 * x);
    }
};
struct GPT{
    Matrix wte , wpe, lm_head;
    vector<AttentionBlock> attn_blocks;//wordtokenembedding worldpositionembedding languagemodelhead
    vector<MLPBlock> mlp_blocks;
    Vector forward(int token_id, int pos_id, vector<vector<Vector>>& layer_keys, vector<vector<Vector>>& layer_values) {
        Vector tok; tok.data = wte.data[token_id];
        Vector pos; pos.data = wpe.data[pos_id];
        Vector x = add(tok, pos);
        for(auto i=0; i<attn_blocks.size(); ++i) {
            Vector attn_out = attn_blocks[i].forward(x, layer_keys[i], layer_values[i]);
            x = add(x, attn_out);
            Vector mlp_out = mlp_blocks[i].forward(x);
            x = add(x, mlp_out);
        }
        Vector logits = lm_head * x;
        return logits;
    }
    vector<Value*> params()
    {
        vector<Value*> ps;
        auto add = [&](auto& m) {
            for (auto& r : m.data)
                for (auto& v : r)ps.push_back(v);
        };
        add(wte); add(wpe); add(lm_head);
        for (auto& b : attn_blocks) { add(b.Wq); add(b.Wk); add(b.Wv); }
        for (auto& b : mlp_blocks)  { add(b.W1); add(b.W2); }
        return ps;
    }
};
class Tokenize
{
public:
    vector<char> vocab;
    map<char, int> char_to_id;
    int BOS , EOS ,UNK;
    vector<int> encode(const string& text) {
        vector<int> tokens;
        tokens.push_back(BOS); 
        for (const auto& ch : text) tokens.push_back((char_to_id[ch])?char_to_id[ch]:UNK);
        tokens.push_back(EOS);  // 结尾加 EOS
        return tokens;
    }
    string decode(const vector<int>& tokens) {
        string s;
        for (int id : tokens) {
            if (id >= 0 && id < (int)vocab.size() && id != BOS && id != EOS && id != UNK)s += vocab[id];
        }
        return s;
    }
};
int sample(const Vector& probs)
{
    static mt19937 rng(std::random_device{}()); 
    vector<double> p;
    for (size_t i = 0; i < probs.data.size(); ++i) p.push_back(probs.data[i]->data);
    discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng);
}
string generate(GPT& model, Tokenize& tokenizer, int max_len = 16, double temp = 0.5) 
{
    graph_pool.clear();
    int n_layers = (int)model.attn_blocks.size();
    vector<vector<Vector>> layer_keys(n_layers), layer_values(n_layers); 
    int token_id = tokenizer.BOS; 
    vector<int> generated;
    for (int pos = 0; pos < max_len; ++pos) {
        Vector logits = model.forward(token_id, pos, layer_keys, layer_values);
        Vector scaled_logits = scale(temp, logits);
        Vector probs = softmax(scaled_logits);
        token_id = sample(probs);
        if (token_id == tokenizer.EOS) break;
        generated.push_back(token_id);
    }
    return tokenizer.decode(generated);
}

int main()
{
    // ── Read Data ──
    freopen("input_names.txt", "r", stdin);
    vector<string> data;
    string line;
    while (getline(cin, line)) if (!line.empty()) data.push_back(line);
    
    // ── Build Tokenizer ──
    Tokenize tokenizer;
    set<char> chars;
    for(auto& s: data) for(char c: s) chars.insert(c);
    for(char c: chars) {
        tokenizer.char_to_id[c] = tokenizer.vocab.size();
        tokenizer.vocab.push_back(c);
    }
    tokenizer.BOS = tokenizer.vocab.size();
    tokenizer.EOS = tokenizer.vocab.size() + 1;
    tokenizer.UNK = tokenizer.vocab.size() + 2;
    tokenizer.vocab.push_back('#'); // BOS
    tokenizer.vocab.push_back('#'); // EOS
    tokenizer.vocab.push_back('?'); // UNK
    cfg.num_vocab = tokenizer.vocab.size();

    // ── Initialize Model ──
    GPT model;
    model.wte = Matrix(); model.wte.row = cfg.num_vocab; model.wte.col = cfg.dim_embd;
    model.wpe = Matrix(); model.wpe.row = cfg.dim_block; model.wpe.col = cfg.dim_embd;
    model.lm_head = Matrix(); model.lm_head.row = cfg.num_vocab; model.lm_head.col = cfg.dim_embd;
    
    auto init_matrix = [](Matrix& m) {
        static mt19937 rng(42);
        normal_distribution<double> dist(0.0, 0.2);
        m.data.resize(m.row, vector<Value*>(m.col));
        for(int i=0; i<(int)m.row; ++i) {
            for(int j=0; j<(int)m.col; ++j) {
                param_pool.emplace_back(dist(rng));
                m.data[i][j] = &param_pool.back();
            }
        }
    };
    init_matrix(model.wte); init_matrix(model.wpe); init_matrix(model.lm_head);
    
    int head_dim = cfg.dim_embd / cfg.num_head;
    for(int i=0; i<cfg.num_layer; ++i) {
        AttentionBlock attn;
        attn.dim_embd = cfg.dim_embd; attn.num_head = cfg.num_head; attn.head_dim = head_dim;
        attn.Wq = Matrix(); attn.Wq.row = cfg.dim_embd; attn.Wq.col = cfg.dim_embd;
        attn.Wk = Matrix(); attn.Wk.row = cfg.dim_embd; attn.Wk.col = cfg.dim_embd;
        attn.Wv = Matrix(); attn.Wv.row = cfg.dim_embd; attn.Wv.col = cfg.dim_embd;
        init_matrix(attn.Wq); init_matrix(attn.Wk); init_matrix(attn.Wv);
        model.attn_blocks.push_back(attn);
        
        MLPBlock mlp;
        mlp.W1 = Matrix(); mlp.W1.row = cfg.dim_hidden; mlp.W1.col = cfg.dim_embd;
        mlp.W2 = Matrix(); mlp.W2.row = cfg.dim_embd; mlp.W2.col = cfg.dim_hidden;
        init_matrix(mlp.W1); init_matrix(mlp.W2);
        model.mlp_blocks.push_back(mlp);
    }

    // ── Training Loop ──
    cout << "Training..." << endl;
    for(int step = 0; step < cfg.num_training_steps; ++step) {
        graph_pool.clear();
        const string& doc = data[step % data.size()];
        vector<int> tokens = tokenizer.encode(doc);
        
        int n_layers = model.attn_blocks.size();
        vector<vector<Vector>> layer_keys(n_layers), layer_values(n_layers);
        Value* total_loss = Value::make_new(0.0);
        int count = 0;

        for(size_t pos = 0; pos < tokens.size() - 1; ++pos) {
            Vector logits = model.forward(tokens[pos], pos, layer_keys, layer_values);
            Vector probs = softmax(logits);
            int target = tokens[pos + 1];
            Value* prob = probs.data[target];
            Value* log_prob = Value::log(prob);
            total_loss = Value::sub(total_loss, log_prob);
            count++;
        }
        
        Value* mean_loss = Value::mul(total_loss, Value::make_new(1.0 / count));
        mean_loss->backward();
        
        for(auto p : model.params()) {
            p->data -= cfg.lr * p->grad;
            p->grad = 0.0;
        }
        
        if(step % 10 == 0) {
            cout << "Step " << step << " Loss: " << mean_loss->data << endl;
        }
    }
    
    // ── Generation ──
    cout << "\n=== Generated Samples ===" << endl;
    for(int i = 0; i < cfg.num_samples; ++i) {
        string result = generate(model, tokenizer, cfg.dim_block, 0.8);
        cout << "  [" << i << "] " << result << endl;
    }
    return 0;
}
