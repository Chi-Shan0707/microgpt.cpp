# MicroGPT（纯 Python 最小 GPT）

这个目录现在包含：
- `microgpt.py`：训练 + 推理一体的最小 GPT 实现。
- `microgpt_explainer.py`：完整可运行的 microGPT（详注版），包含【代码意图】+【LLM算法原理】+【Transformer思想】注释。

## 1) 详注版 microGPT 使用

在 `MicroGPT/` 目录下运行：

```bash
python microgpt_explainer.py
```

说明：
- 该文件本身就是"完整 microGPT 训练 + 推理代码"，不是外部解释器。
- 你可以直接在这个文件上继续改造（例如后续对齐 C++ 重写逻辑）。

## 2) microgpt.py 训练方法

在 `MicroGPT/` 目录下运行：

```bash
python microgpt.py
```

训练流程说明：
1. 若当前目录没有 `input.txt`，脚本会自动下载 names 数据集。
2. 按字符级 tokenizer 构建词表，加入 BOS token。
3. 初始化 1 层、16 维 embedding 的极简 GPT 参数。
4. 用自定义 `Value` 自动求导 + Adam 优化器训练 `num_steps=1000`。
5. 训练结束后执行采样推理，生成 20 个新名字。

你可以修改这些超参数进行实验：
- `n_layer`, `n_embd`, `block_size`, `n_head`
- `learning_rate`, `beta1`, `beta2`, `num_steps`, `temperature`

## 3) 未来 C++ 重写计划（To-Do）

- [ ] 设计 `Value` 标量自动求导类（含计算图、拓扑反传、运算符重载）。
- [ ] 实现张量/矩阵容器与基础算子（matmul、softmax、rmsnorm、relu）。
- [ ] 按当前结构实现 GPT 前向：embedding -> attention -> MLP -> lm_head。
- [ ] 实现训练循环：tokenize、loss、backward、Adam 更新、梯度清零。
- [ ] 对齐 Python 版本超参，确保 loss 曲线趋势一致。
- [ ] 加入推理采样（temperature、BOS 截断）。
- [ ] 提供最小可复现脚本：build + run + 对比输出。
- [ ] 后续再做性能版：批处理、向量化、并行、内存池。


## 4 ）链接

- [网页版可视化microgpt](https://microgpt.boratto.ca/)


## 5 ) 第一个小尝试

- 做数学题？

layer = 1  -> layer =4 

```bash
(base) chishan@LAPTOP-7N8BKOTJ:/mnt/d/CS/CandC++/Hone-My-C-Plus-Plus-/MicroGPT$ python3 microgpt_annotate.py 
num docs: 10000
vocab size: 13
num params: 3744
step 1000 / 1000 | loss 1.8863
--- inference (new, hallucinated names) ---
sample  1: 126+40=310
sample  2: 652+62=874
sample  3: 130+685=187
sample  4: 192+886=111
sample  5: 827+476=115
sample  6: 962+122=192
sample  7: 120+787=1290
sample  8: 454+268=179
sample  9: 641+424=141
sample 10: 49+965=814
sample 11: 229+21=753
sample 12: 463+28=744
sample 13: 854+494=124
sample 14: 936+468=1
sample 15: 689+357=158
sample 16: 588+483=731
sample 17: 351+97=514
sample 18: 913+10=897
sample 19: 427+767=152
sample 20: 883+566=117
(base) chishan@LAPTOP-7N8BKOTJ:/mnt/d/CS/CandC++/Hone-My-C-Plus-Plus-/MicroGPT$ python3 microgpt_annotate.py 
num docs: 10000
vocab size: 13
num params: 12960
step 1000 / 1000 | loss 1.8350
--- inference (new, hallucinated names) ---
sample  1: 383+768=1333
sample  2: 822+267=1303
sample  3: 692+34=757
sample  4: 933+280=1050
sample  5: 289+142=1165
sample  6: 793+936=1760
sample  7: 723+55=563
sample  8: 424+83=932
sample  9: 860+161=1061
sample 10: 415+10=913
sample 11: 449+70=842
sample 12: 906+221=1661
sample 13: 25+862=411
sample 14: 483+225=1249
sample 15: 822+604=1113
sample 16: 985+222=1027
sample 17: 524+756=1908
sample 18: 428+509=1415
sample 19: 134+85=243
sample 20: 508+253=1667
```
