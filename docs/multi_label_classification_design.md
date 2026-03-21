# AMD vs DGM4 多标签分类设计对比

## 结论先行

**AMD 和 DGM4 的多标签分类设计有本质差异。** AMD 当前的实现并非真正的 4 维多标签分类，而是一个简化的方案。

---

## 1. DGM4 的设计（原始论文定义）

### 标签体系

4 维独立二分类，每维对应一种篡改类型：

| 维度 | 含义             |
|------|------------------|
| 0    | Face Swap (FS)   |
| 1    | Face Attribute (FA) |
| 2    | Text Swap (TS)   |
| 3    | Text Attribute (TA) |

共 9 种标签组合：

| 类别                | 标签向量        |
|---------------------|-----------------|
| orig                | [0, 0, 0, 0]   |
| face_swap           | [1, 0, 0, 0]   |
| face_attribute      | [0, 1, 0, 0]   |
| text_swap           | [0, 0, 1, 0]   |
| text_attribute      | [0, 0, 0, 1]   |
| face_swap+text_swap | [1, 0, 1, 0]   |
| face_swap+text_attr | [1, 0, 0, 1]   |
| face_attr+text_swap | [0, 1, 1, 0]   |
| face_attr+text_attr | [0, 1, 0, 1]   |

### 损失函数

`binary_cross_entropy_with_logits`：4 个维度各自独立计算 BCE。

### 评估指标

- 每类独立 F1（F1_FS, F1_FA, F1_TS, F1_TA）
- 微平均：OP / OR / OF1
- 宏平均：CP / CR / CF1
- MAP：4 类 AP 的均值

---

## 2. AMD 的当前设计

### 2.1 标签体系：只用 3 维，text_attribute 被合并

AMD 虽然定义了 4 维 multi_label，但 **第 4 维（text_attribute）永远为 0**：

```python
# scripts/data.py:300
label = ann['fake_cls'].replace('text_attribute','text_swap')
```

所有 `text_attribute` 样本被统一归为 `text_swap`，因此实际只有 6 种标签组合：

| 类别                     | AMD 标签向量    | DGM4 原始        |
|--------------------------|-----------------|-------------------|
| orig                     | [0, 0, 0, 0]   | [0, 0, 0, 0]     |
| face_swap                | [1, 0, 0, 0]   | [1, 0, 0, 0]     |
| face_attribute           | [0, 1, 0, 0]   | [0, 1, 0, 0]     |
| text_swap                | [0, 0, 1, 0]   | [0, 0, 1, 0]     |
| face_swap & text_swap    | [1, 0, 1, 0]   | [1, 0, 1, 0]     |
| face_attribute & text_swap | [0, 1, 1, 0] | [0, 1, 1, 0]     |

缺失的 3 种 DGM4 组合：`text_attribute [0,0,0,1]`、`FS+TA [1,0,0,1]`、`FA+TA [0,1,0,1]`。

MAP 计算也只取前 3 维：
```python
# scripts/test_on_MDSM.py:209
MAP = multi_label_meter.value()[:3].mean()
```

### 2.2 分类头：二分类（real/fake），而非 4 维多标签

AMD 的辅助分类头输出 **2 类（real vs fake）**，不是 4 维独立二分类：

```python
# models/modeling_florence2.py:2080-2082
self.classifier = nn.Linear(config.d_model, 2)       # learnable token 分类
self.Second_classifier = nn.Linear(config.d_model, 2) # 图像/文本模态分类
```

训练时的标签也是单一的 binary label：

```python
# scripts/train.py:508-515
Binary_lables = []
for tt, label in enumerate(answers):
    if label.startswith('A'):
        Binary_lables.append(1)   # real
    else:
        Binary_lables.append(0)   # fake（所有篡改类型都是 0）
```

### 2.3 损失函数：CrossEntropyLoss，而非 BCE

```python
# scripts/train.py:357
criterion = torch.nn.CrossEntropyLoss()
```

用的是 2 类 CrossEntropy（softmax-based），不是 4 个维度各自 BCE（sigmoid-based）。

### 2.4 多标签评估：间接方式（文本生成 → TF-IDF 匹配）

AMD 的多标签指标并非从分类头直接获取，而是：
1. 模型**生成自然语言回答**（VQA 风格）
2. 用 **TF-IDF + cosine similarity** 将生成文本匹配到最近的选项
3. 将匹配到的选项转换为 multi_label 向量
4. 用 `AveragePrecisionMeter` 计算 MAP/OP/OR/OF1/CP/CR/CF1

option_labels 中还使用了非标准的 soft label（包含 -0.33 和 0.5），而非 DGM4 的纯 0/1 binary：

```python
# scripts/train.py:362-369
option_labels = [
    torch.tensor([0, 0, 0, 0]),           # A. No.
    torch.tensor([1, -0.33, -0.33, -0.33]), # B. face_swap
    torch.tensor([-0.33, 1, -0.33, -0.33]), # C. face_attribute
    torch.tensor([-0.33, -0.33, 1, -0.33]), # D. text_swap
    torch.tensor([0.5, -0.5, 0.5, -0.5]),   # E. FS + TS
    torch.tensor([-0.5, 0.5, 0.5, -0.5]),   # F. FA + TS
]
```

---

## 3. 核心差异总结

| 维度           | DGM4                            | AMD                                    |
|----------------|--------------------------------|----------------------------------------|
| 标签维度       | 4 维（FS/FA/TS/TA）            | 实际 3 维（TA 合并入 TS）              |
| 标签组合       | 9 种                           | 6 种                                   |
| 分类头输出     | 4 维各自独立 sigmoid           | 2 类（real/fake）softmax               |
| 损失函数       | BCE (binary_cross_entropy)     | CrossEntropyLoss (2-class)             |
| 多标签信号来源 | 分类头直接输出                 | 文本生成 → TF-IDF 匹配 → 间接推断     |
| MAP 计算       | 4 类均值                       | 前 3 类均值                            |
| option_labels  | 纯 0/1 binary                  | soft label（含 -0.33, 0.5 等）         |

---

## 4. AMD 不一致的原因分析

AMD 的核心思路不同于 DGM4：
- **DGM4** 是一个经典的多标签分类器，直接从分类头预测 4 维标签
- **AMD** 基于 Florence2 (VLM)，采用生成式范式——模型通过生成文本回答"是否存在篡改"这个 VQA 问题，辅助分类头只做 real/fake 二分类作为额外监督信号

因此 AMD 的多标签能力是隐式的（通过生成式回答间接体现），而非显式的（通过分类头直接输出 4 维标签）。

---

## 5. 如果要对齐 DGM4 的多标签设计，需要改什么

若要让 AMD 完全对齐 DGM4 的 4 维多标签分类：

### 5.1 数据层
- 恢复 `text_attribute` 类别，不再合并到 `text_swap`
- 添加缺失的 3 种组合标签
- 新增选项 G/H/I 对应 text_attribute 相关组合

### 5.2 模型层
- 分类头从 `nn.Linear(d_model, 2)` 改为 `nn.Linear(d_model, 4)` + sigmoid
- 或者保留多头但每头输出 4 维

### 5.3 损失函数
- 从 `CrossEntropyLoss` 改为 `BCEWithLogitsLoss`
- 标签从 `Binary_lables`（0/1 real/fake）改为 4 维 `multi_label`

### 5.4 评估层
- option_labels 改回纯 0/1 binary
- MAP 计算改为 4 类均值
- 可直接从分类头获取多标签预测，不必依赖 TF-IDF 匹配
