# Stage 1 & Stage 2 Function Cheatsheet

This note summarizes the main functions, classes, and methods used in Stage 1 and Stage 2 of the `multi-method-ml-research-agent` project.

It is written for first-time readers of scikit-learn and PyTorch code.

---

## Quick Clarification: What is `nn`?

```python
import torch.nn as nn
```

`nn` is short for **neural network**, not neutral network.

`torch.nn` is PyTorch's neural network module library. You can think of it as a toolbox of neural network building blocks:

- `nn.Module`: base class for neural network models
- `nn.Embedding`: token id to vector lookup layer
- `nn.Linear`: fully connected linear layer, `xW + b`
- `nn.ReLU`: non-linear activation function
- `nn.Dropout`: randomly drops hidden signals during training
- `nn.Sequential`: wraps layers into a sequential pipeline
- `nn.CrossEntropyLoss`: classification loss for logits and label ids

---

# 1. Stage 1: Traditional ML Baseline

Stage 1 pipeline:

```text
raw text
→ train/test split
→ TF-IDF vectorization
→ Logistic Regression / Naive Bayes / Random Forest
→ prediction
→ metrics and reports
```

## 1.1 Script and Path Utilities

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `argparse.ArgumentParser` | argument parser | 命令行参数解析器 | `parser = argparse.ArgumentParser(...)` | Allows the script to accept CLI parameters such as `--data`, `--text_col`, `--label_col` |
| `parser.add_argument()` | add argument | 添加命令行参数 | `parser.add_argument("--data", type=Path)` | Defines one CLI argument |
| `parser.parse_args()` | parse arguments | 解析命令行参数 | `args = parser.parse_args()` | Converts terminal input into `args.xxx` variables |
| `Path` | path object | 路径对象 | `Path("reports/stage1")` | Handles file and directory paths |
| `path.exists()` | check existence | 检查路径是否存在 | `if not data_path.exists(): ...` | Checks whether the input CSV exists |
| `path.mkdir()` | make directory | 创建目录 | `output_dir.mkdir(parents=True, exist_ok=True)` | Creates report output directories |
| `path.write_text()` | write text | 写入文本文件 | `path.write_text(text, encoding="utf-8")` | Saves report text or summary text |
| `sys.path.append()` | append system path | 添加 Python 模块搜索路径 | `sys.path.append(str(PROJECT_ROOT))` | Lets scripts import modules from `src/` |

## 1.2 Data Split and TF-IDF

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `train_test_split` | train-test split | 训练集/测试集切分 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | Splits the dataset into training and test sets |
| `stratify=` | stratified sampling | 分层抽样 | `stratify=labels` | Keeps label distribution similar between train and test sets |
| `TfidfVectorizer` | TF-IDF vectorizer | TF-IDF 文本向量器 | `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` | Converts raw text into sparse TF-IDF feature vectors |
| `fit_transform()` | fit and transform | 学习规则并转换 | `X_train_tfidf = vectorizer.fit_transform(X_train)` | Learns vocabulary and IDF from training data, then transforms training text |
| `transform()` | transform only | 只转换，不重新学习 | `X_test_tfidf = vectorizer.transform(X_test)` | Applies training vocabulary/IDF to test data; avoids data leakage |
| `build_tfidf_features()` | custom TF-IDF builder | 自定义 TF-IDF 构造函数 | `build_tfidf_features(X_train, X_test)` | Wraps `TfidfVectorizer`, `fit_transform`, and `transform` |

## 1.3 Traditional ML Models

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `LogisticRegression` | logistic regression | 逻辑回归分类器 | `LogisticRegression(max_iter=1000, random_state=42)` | Strong linear baseline for sparse TF-IDF text features |
| `MultinomialNB` | multinomial naive Bayes | 多项式朴素贝叶斯 | `MultinomialNB()` | Fast probabilistic baseline for text classification |
| `RandomForestClassifier` | random forest classifier | 随机森林分类器 | `RandomForestClassifier(n_estimators=200, random_state=42)` | Non-linear tree ensemble baseline |
| `get_baseline_models()` | custom model factory | 获取 baseline 模型字典 | `models = get_baseline_models()` | Returns Logistic Regression, Naive Bayes, and Random Forest |
| `model.fit()` | fit / train model | 训练模型 | `model.fit(X_train, y_train)` | Trains a scikit-learn model |
| `model.predict()` | predict labels | 预测标签 | `y_pred = model.predict(X_test)` | Predicts labels on test data |

## 1.4 Metrics and Saving

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `accuracy_score` | accuracy score | 准确率 | `accuracy_score(y_true, y_pred)` | Measures overall correctness |
| `precision_recall_fscore_support` | precision/recall/F-score support | 精确率、召回率、F1 和样本数 | `precision_recall_fscore_support(..., average="weighted")` | Calculates weighted precision, recall, and F1 |
| `f1_score` | F1 score | F1 分数 | `f1_score(y_true, y_pred, average="macro")` | Calculates macro F1 for model comparison |
| `classification_report` | classification report | 分类报告 | `classification_report(y_true, y_pred, zero_division=0)` | Produces per-class precision/recall/F1 report |
| `evaluate_classification_model()` | custom evaluator | 自定义分类评估函数 | `evaluate_classification_model(y_test, y_pred)` | Returns accuracy, weighted precision, weighted recall, and weighted F1 |
| `get_classification_report_text()` | custom report generator | 自定义分类报告文本函数 | `get_classification_report_text(y_true, y_pred, target_names)` | Generates report text for saving |
| `joblib.dump()` | save Python object | 保存 Python 对象 | `joblib.dump(model, "best_model.joblib")` | Saves trained model/vectorizer artifacts |
| `save_outputs()` | custom output saver | 自定义输出保存函数 | `save_outputs(...)` | Saves model results, reports, summary, and optional joblib artifacts |

---

# 2. Stage 2: PyTorch Neural Baseline

Stage 2 pipeline:

```text
raw text
→ tokenizer
→ vocabulary
→ token ids
→ padding
→ Dataset
→ DataLoader
→ Embedding
→ masked mean pooling
→ MLP classifier
→ logits
→ CrossEntropyLoss
→ backward
→ optimizer.step
```

## 2.1 Text Processing and Dataset

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `re.findall()` | regex find all | 正则匹配所有符合规则的 token | `re.findall(r"[a-z0-9]+", text)` | Extracts simple English/digit tokens from text |
| `Counter` | counter | 计数器 | `counter.update(tokens)` | Counts token frequencies |
| `simple_tokenize()` | simple tokenizer | 简单分词函数 | `tokens = simple_tokenize(text)` | Lowercases text and extracts tokens |
| `build_vocab()` | build vocabulary | 建词表 | `vocab = build_vocab(X_train)` | Builds token-to-id mapping from training texts only |
| `encode_text()` | encode text | 文本编码 | `ids = encode_text(text, vocab)` | Converts tokens into token ids |
| `pad_or_truncate()` | pad or truncate | 补齐或截断 | `pad_or_truncate(ids, max_len)` | Makes every input sequence the same length |
| `build_label_mapping()` | build label mapping | 建标签映射 | `label_to_id, id_to_label = build_label_mapping(labels)` | Converts string labels into integer label ids |
| `Dataset` | PyTorch Dataset | PyTorch 数据集基类 | `class MyDataset(Dataset)` | Base class for custom datasets |
| `ResearchTextDataset` | custom text dataset | 自定义文本分类数据集 | `ResearchTextDataset(texts, labels, vocab, label_to_id, max_len)` | Returns `input_ids` and `label` tensors for each sample |
| `__len__()` | length method | 返回数据集长度 | `len(dataset)` | Tells DataLoader how many samples exist |
| `__getitem__()` | get item method | 按索引取样本 | `dataset[idx]` | Returns one processed sample |
| `torch.tensor()` | create tensor | 创建 PyTorch 张量 | `torch.tensor(input_ids, dtype=torch.long)` | Converts Python lists into tensors |
| `torch.long` | long integer dtype | 长整型 | `dtype=torch.long` | Required for token ids and class labels |

## 2.2 DataLoader and Device Management

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `DataLoader` | data loader | 批量数据加载器 | `DataLoader(dataset, batch_size=8, shuffle=True)` | Groups dataset samples into mini-batches |
| `batch_size` | batch size | 每批样本数 | `batch_size=8` | Controls how many samples are trained together |
| `shuffle=True` | shuffle data | 打乱数据 | `DataLoader(train_ds, shuffle=True)` | Randomizes training sample order |
| `shuffle=False` | no shuffle | 不打乱数据 | `DataLoader(val_ds, shuffle=False)` | Keeps validation order stable |
| `torch.device()` | device object | 设备对象 | `torch.device("cpu")` / `torch.device("cuda")` | Represents CPU/GPU device |
| `torch.cuda.is_available()` | check CUDA | 检查 CUDA/GPU 是否可用 | `torch.cuda.is_available()` | Helps choose CPU or GPU automatically |
| `.to(device)` | move to device | 移动到设备 | `model.to(device)` / `tensor.to(device)` | Places model and tensors on the same device |
| `torch.manual_seed()` | set random seed | 设置随机种子 | `torch.manual_seed(42)` | Improves reproducibility |
| `torch.cuda.manual_seed()` | set CUDA seed | 设置 GPU 随机种子 | `torch.cuda.manual_seed(seed)` | Controls CUDA randomness when GPU is used |
| `torch.cuda.manual_seed_all()` | set all CUDA seeds | 设置所有 GPU 随机种子 | `torch.cuda.manual_seed_all(seed)` | Used when multiple CUDA devices exist |

## 2.3 `torch.nn` Neural Network Building Blocks

| Function / Class | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `torch.nn` / `nn` | neural network module | 神经网络模块库 | `import torch.nn as nn` | Provides neural network layers and loss functions |
| `nn.Module` | module base class | 神经网络模型基类 | `class Model(nn.Module)` | Base class for custom PyTorch models |
| `super().__init__()` | initialize parent class | 初始化父类 | `super().__init__()` | Lets PyTorch register model parameters correctly |
| `nn.Embedding` | embedding layer | 词向量查表层 | `nn.Embedding(vocab_size, embedding_dim, padding_idx=0)` | Converts token ids into trainable token vectors |
| `padding_idx=0` | padding index | PAD token 的索引 | `nn.Embedding(..., padding_idx=0)` | Prevents PAD from behaving like a normal semantic token |
| `nn.Linear` | linear / fully connected layer | 线性层 / 全连接层 | `nn.Linear(in_dim, out_dim)` | Computes `xW + b`; maps vectors to hidden states or logits |
| `nn.ReLU` | rectified linear unit | ReLU 激活函数 | `nn.ReLU()` | Converts negative values to zero and adds non-linearity |
| `nn.Dropout` | dropout layer | 随机丢弃层 | `nn.Dropout(0.2)` | Randomly drops hidden signals during training to reduce overfitting |
| `nn.Sequential` | sequential container | 顺序容器 | `nn.Sequential(layer1, layer2, ...)` | Chains layers in order |
| `forward()` | forward pass | 正向传播函数 | `def forward(self, input_ids): ...` | Defines how input tensors become logits |
| `model(input_ids)` | model call | 调用模型 | `logits = model(input_ids)` | Automatically calls `forward()` |
| `nn.CrossEntropyLoss` | cross entropy loss | 交叉熵损失函数 | `criterion = nn.CrossEntropyLoss()` | Computes classification loss from logits and label ids |

## 2.4 Tensor Operations and Masked Mean Pooling

| Function / Method | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `input_ids != 0` | boolean mask | 布尔掩码 | `mask = input_ids != 0` | Marks real tokens and excludes PAD tokens |
| `.unsqueeze(-1)` | add dimension | 增加一个维度 | `mask.unsqueeze(-1)` | Changes shape from `(batch, max_len)` to `(batch, max_len, 1)` |
| `.float()` | convert to float | 转为浮点数 | `mask.float()` | Lets the mask multiply embedding vectors |
| `embeddings * mask` | elementwise multiplication | 逐元素相乘 | `masked_embeddings = embeddings * mask` | Zeros out PAD positions |
| `.sum(dim=1)` | sum along dimension | 沿指定维度求和 | `masked_embeddings.sum(dim=1)` | Sums token vectors within each sentence |
| `.clamp(min=1.0)` | clamp minimum | 限制最小值 | `valid_counts.clamp(min=1.0)` | Prevents division by zero |
| `/` | tensor division | 张量除法 | `sum_embeddings / valid_counts` | Computes mean pooled sentence vectors |
| `torch.argmax()` | argmax | 最大值索引 | `torch.argmax(logits, dim=1)` | Converts logits into predicted class ids |
| `.cpu()` | move to CPU | 移回 CPU | `tensor.cpu()` | Moves tensor from GPU to CPU before evaluation |
| `.numpy()` | convert to NumPy | 转 NumPy 数组 | `tensor.cpu().numpy()` | Makes tensor compatible with sklearn metrics |
| `.tolist()` | convert to list | 转 Python list | `tensor.tolist()` | Produces `y_true` / `y_pred` lists |
| `.item()` | scalar tensor to Python number | 标量张量转普通数字 | `loss.item()` | Records loss as a normal Python float |

## 2.5 Training Loop

| Function / Method | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `model.train()` | training mode | 训练模式 | `model.train()` | Enables training behavior such as Dropout |
| `model.eval()` | evaluation mode | 验证/推理模式 | `model.eval()` | Disables training randomness such as Dropout |
| `torch.no_grad()` | no gradient context | 不计算梯度环境 | `with torch.no_grad(): ...` | Speeds up validation and saves memory |
| `torch.optim.Adam` | Adam optimizer | Adam 优化器 | `optimizer = torch.optim.Adam(model.parameters(), lr=0.01)` | Updates parameters using gradients |
| `model.parameters()` | model parameters | 模型参数 | `model.parameters()` | Passes embedding and classifier parameters to optimizer |
| `optimizer.zero_grad()` | zero gradients | 清空梯度 | `optimizer.zero_grad()` | Clears old gradients before a new batch |
| `loss.backward()` | backpropagation | 反向传播 | `loss.backward()` | Computes gradients for trainable parameters |
| `optimizer.step()` | optimizer step | 参数更新 | `optimizer.step()` | Updates embedding and MLP weights |
| `train_one_epoch()` | train one epoch | 训练一轮 | `train_one_epoch(model, loader, criterion, optimizer, device)` | Runs one full pass over the training set |
| `evaluate()` | evaluate model | 验证模型 | `evaluate(model, val_loader, criterion, device)` | Computes validation loss and predictions |
| `compute_metrics()` | compute metrics | 计算指标 | `compute_metrics(y_true, y_pred)` | Computes accuracy, precision, recall, and F1 |
| `save_stage2_outputs()` | save outputs | 保存输出 | `save_stage2_outputs(...)` | Saves result CSV, classification report, training log, and summary |

## 2.6 Metrics Used in Stage 2

| Function | English Name | 中文理解 | Common Usage | Project Role |
|---|---|---|---|---|
| `accuracy_score` | accuracy score | 准确率 | `accuracy_score(y_true, y_pred)` | Measures overall prediction correctness |
| `precision_score` | precision score | 精确率 | `precision_score(..., average="weighted")` | Measures weighted precision |
| `recall_score` | recall score | 召回率 | `recall_score(..., average="weighted")` | Measures weighted recall |
| `f1_score` | F1 score | F1 分数 | `f1_score(..., average="macro")` | Measures macro F1 or weighted F1 |
| `classification_report` | classification report | 分类报告 | `classification_report(y_true, y_pred, zero_division=0)` | Generates per-class metrics |

---

# 3. Minimal Memory List

If you are new to PyTorch, memorize these first:

| Function / Class | One-line Meaning |
|---|---|
| `nn` | PyTorch neural network toolbox |
| `nn.Module` | base class for custom models |
| `nn.Embedding` | token id to trainable token vector |
| `nn.Linear` | matrix multiplication plus bias, `xW + b` |
| `nn.ReLU` | non-linear activation; negative values become zero |
| `nn.Dropout` | randomly drops signals during training |
| `nn.Sequential` | chains layers in order |
| `nn.CrossEntropyLoss` | logits plus label ids to loss |
| `Dataset` | defines how to retrieve one sample |
| `DataLoader` | groups samples into batches |
| `torch.tensor` | converts Python data into tensors |
| `.to(device)` | moves model/tensors to CPU or GPU |
| `model.train()` | training mode |
| `model.eval()` | evaluation mode |
| `torch.no_grad()` | validation without gradients |
| `torch.argmax` | chooses predicted class from logits |
| `optimizer.zero_grad()` | clears old gradients |
| `loss.backward()` | computes gradients |
| `optimizer.step()` | updates parameters |
| `TfidfVectorizer` | Stage 1 text to TF-IDF features |

---

# 4. Big Picture

```text
Stage 1 = scikit-learn world
TF-IDF + traditional classifiers + classification metrics

Stage 2 = PyTorch world
Dataset/DataLoader + nn.Module + Embedding/MLP + CrossEntropyLoss + backward/optimizer
```

The most important conceptual difference:

```text
Stage 1:
Text features are created by TF-IDF rules, and the classifier learns a decision boundary.

Stage 2:
Token embeddings and classifier weights are trained together through gradient descent.
```
