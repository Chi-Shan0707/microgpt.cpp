Inspired by [Karpathy's MakeMore](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

# Introduction

**microgpt.cpp in 300 lines**: A minimal GPT implementation in pure C++, demonstrating core deep learning concepts: autograd, backpropagation, and text generation.

## Implementation Details

The C++ implementation in `microgpt.cpp` follows a minimalist design:

- **300-Line Core**: The entire engine (Autograd + Transformer) fits in approximately 300 lines of code.
- **Architecture**: 
  - GPT-style Decoder-only Transformer.
  - Multi-head Self-Attention (without key/value caching for simplicity).
  - Positional Encodings (Learned Embedding Table).
  - MLP with ReLU activation.
  - **No Bias Terms**: All linear transformations ($Wq, Wk, Wv, Wo, W1, W2$) are implemented as pure matrix multiplications without additive bias vectors to keep the code compact.
- **Optimizer**: 
  - **Vanilla SGD**: Fixed learning rate Stochastic Gradient Descent. 
  - Direct gradient subtraction: `p->data -= lr * p->grad`.
  - No Momentum or Adam-style adaptive moments.
- **Autograd System**: 
  - Reverse-mode automatic differentiation.
  - Custom `Value` class managing the computation graph and topological sort.

## Core Files

| File | Language | Purpose | Status |
|------|----------|---------|--------|
| `microgpt.py` | Python | Karpathy's original reference implementation | Reference |
| `microgpt_annotated.py` | Python | Educational version with detailed comments | Learning |
| `microgpt_runnable.cpp` | C++ | Full implementation, less polished | Functional |
| `microgpt.cpp` | C++ | Concise, readable, production version | **Recommended** |

## Features

- **Single-file implementation** with no external dependencies (C++ only)
- **Autograd system** with automatic differentiation via computation graphs
- **Multi-head attention** with positional encoding
- **MLP feedforward** layers with ReLU activation
- **Training** with cross-entropy loss and SGD optimization
- **Generation** with temperature scaling for diversity

# Requirements
- **Python**: no dependencies (for `.py` files)
- **C++**: >= C++11 (for `.cpp` files)

## Build & Run (C++)

```bash
g++ -O2 -std=c++14 -o microgpt microgpt.cpp
./microgpt
```

The program reads from `input_names.txt`, trains for 1000 steps, and generates 20 sample names.

## Training Target: Generate Names

The model learns to generate plausible English names from `input_names.txt` during training.

## Interesting tasks #0 
- Generate English names

Example:
```bash
=== Generated samples ===
  [0] halin
  [1] pavalee
  [2] aliiah
  [3] zariay
  [4] eimola
  [5] alia
  [6] karana
  [7] arelia
  [8] kalina
  [9] jona
  [10] ekana
  [11] aibiya
  [12] kaulii
  [13] ary
  [14] belia
  [15] sin
  [16] jaza
  [17] maranah
  [18] milena
  [19] kano
```

## Interesting tasks #1
- 1+1=?

Example:
```bash
=== Generated samples ===
  [0] 883+514=1046
  [1] 932+854=1621
  [2] 786+045=1536
  [3] 606+202=446
  [4] 0=2+432=987
  [5] 18+527=501
  [6] 256+341=853
  [7] 988+86=368
  [8] 281+408=1429
  [9] 726+308=636
  [10] 392+59=169
  [11] 888+538=1358
  [12] 354+457=1338
  [13] 57+124=743
  [14] 34+139=125
  [15] 543+870=1158
  [16] 851+298=1926
  [17] 536+342=869
  [18] 350+256=1622
  [19] 228+984=1220
```