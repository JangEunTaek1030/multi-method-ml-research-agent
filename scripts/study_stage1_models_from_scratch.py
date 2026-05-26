#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 传统机器学习模型原理学习脚本（从零实现版）

说明：
1) 本脚本仅用于学习、面试讲解、理解算法原理。
2) 不替代生产脚本 scripts/run_stage1_ml_baseline.py。
3) 为了便于讲解，示例中会将小规模 TF-IDF 稀疏矩阵转为稠密矩阵。
4) 本脚本不会保存任何模型文件、报告文件，只在终端打印结果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class SoftmaxLogisticRegressionScratch:
    """从零实现的 Softmax 逻辑回归（多分类）。

    为什么名字里有“回归”却是分类模型？
    - 历史命名原因：它对“线性打分”做了一个可微变换（sigmoid/softmax），
      最终输出的是类别概率，而不是连续实值回归目标。
    - 在二分类时常见 sigmoid；多分类时常用 softmax。

    形状约定（非常重要）：
    - X: (n_samples, n_features)
    - W: (n_features, n_classes)
    - b: (n_classes,)
    - scores = X @ W + b: (n_samples, n_classes)
    - probs = softmax(scores): (n_samples, n_classes)
    """

    def __init__(self, lr=0.1, epochs=500, reg=0.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.verbose = verbose
        self.W = None
        self.b = None
        self.classes_ = None
        self.class_to_idx_ = None

    def _softmax(self, scores):
        """对每个样本做 softmax，得到每个类别的概率。

        为了数值稳定性，会先减去每行最大值，避免 exp 溢出。
        """
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def _one_hot(self, y_idx, num_classes):
        """把类别索引向量转成 one-hot 矩阵。"""
        n_samples = y_idx.shape[0]
        Y = np.zeros((n_samples, num_classes), dtype=float)
        Y[np.arange(n_samples), y_idx] = 1.0
        return Y

    def fit(self, X, y):
        """训练 Softmax 逻辑回归。

        核心公式：
        1) scores = X @ W + b
        2) probs = softmax(scores)
        3) 交叉熵损失：-mean(sum(Y * log(probs)))
        4) 梯度：
           d_scores = (probs - Y) / n_samples
           dW = X.T @ d_scores + reg * W
           db = sum(d_scores)
        5) 参数更新：
           W -= lr * dW
           b -= lr * db

        直觉解释：
        - 交叉熵越小，表示模型给“真实类别”的概率越大。
        - probs - Y 反映了“预测概率和真实分布”的偏差。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[v] for v in y], dtype=int)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        rng = np.random.default_rng(42)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=float)

        Y = self._one_hot(y_idx, n_classes)

        for epoch in range(self.epochs):
            scores = X @ self.W + self.b
            probs = self._softmax(scores)

            eps = 1e-12
            ce_loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))
            reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
            loss = ce_loss + reg_loss

            d_scores = (probs - Y) / n_samples
            dW = X.T @ d_scores + self.reg * self.W
            db = np.sum(d_scores, axis=0)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"[SoftmaxLR] epoch={epoch:4d}, loss={loss:.6f}")

        return self

    def predict_proba(self, X):
        """输出每个样本属于各类别的概率。"""
        X = np.asarray(X, dtype=float)
        scores = X @ self.W + self.b
        return self._softmax(scores)

    def predict(self, X):
        """输出类别预测。"""
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]


class MultinomialNaiveBayesScratch:
    """从零实现的多项式朴素贝叶斯（Multinomial NB）。

    贝叶斯公式核心：
    P(class | text) ∝ P(class) * P(text | class)

    “朴素（naive）”的含义：
    - 假设在给定类别后，各特征（词项）条件独立。
    - 这个假设在现实中不完全成立，但在文本任务里常常依然有效。

    为什么用对数概率？
    - 很多小概率连乘会下溢（接近 0）。
    - 取对数后，乘法变加法，更稳定、更高效。
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """训练 Multinomial NB。

        X 要求非负（例如词频、TF-IDF），形状为 (n_samples, n_features)。
        Laplace 平滑：count + alpha，避免某特征在某类中未出现导致概率为 0。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if np.any(X < 0):
            raise ValueError("MultinomialNaiveBayesScratch 要求 X 非负。")

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        class_count = np.bincount(y_idx, minlength=n_classes).astype(float)
        self.class_log_prior_ = np.log(class_count / n_samples)

        feature_count = np.zeros((n_classes, n_features), dtype=float)
        for c in range(n_classes):
            X_c = X[y_idx == c]
            feature_count[c] = np.sum(X_c, axis=0)

        smoothed = feature_count + self.alpha
        smoothed_sum = np.sum(smoothed, axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed / smoothed_sum)
        return self

    def predict_log_proba(self, X):
        """计算每个样本对应各类别的对数后验“未归一化分数”。"""
        X = np.asarray(X, dtype=float)
        return self.class_log_prior_ + X @ self.feature_log_prob_.T

    def predict(self, X):
        """预测类别。"""
        log_proba = self.predict_log_proba(X)
        pred_idx = np.argmax(log_proba, axis=1)
        return self.classes_[pred_idx]


def gini(y):
    """计算一个标签集合的 Gini impurity（基尼不纯度）。

    直觉：
    - 若一个节点里全是同一类，Gini=0（最“纯”）。
    - 类别越混杂，Gini 越大。
    """
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def weighted_gini(y_left, y_right):
    """计算一次划分后的加权基尼不纯度。"""
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)


class SimpleDecisionTreeScratch:
    """简化版决策树（分类）从零实现。

    说明：
    - 递归寻找最优划分（特征 + 阈值），目标是最小化加权 Gini。
    - 停止条件：
      1) 达到最大深度
      2) 节点样本标签全相同
      3) 样本数小于 min_samples_split
      4) 找不到有效划分
    - 叶子节点预测“多数类”。
    """

    def __init__(self, max_depth=3, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.tree_ = None
        self.classes_ = None

    def fit(self, X, y):
        """训练决策树。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _choose_feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, int):
            k = max(1, min(self.max_features, n_features))
            return self.rng_.choice(n_features, size=k, replace=False)
        return np.arange(n_features)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        feature_indices = self._choose_feature_indices(n_features)

        best_feature = None
        best_threshold = None
        best_score = np.inf

        for f in feature_indices:
            values = np.unique(X[:, f])
            if values.shape[0] < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                score = weighted_gini(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        node = {"is_leaf": False, "prediction": self._majority_class(y)}

        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or X.shape[0] < self.min_samples_split
        ):
            node["is_leaf"] = True
            return node

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            node["is_leaf"] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.update({"feature": best_feature, "threshold": best_threshold})
        node["left"] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        while not node["is_leaf"]:
            f = node["feature"]
            t = node["threshold"]
            node = node["left"] if x[f] <= t else node["right"]
        return node["prediction"]

    def predict(self, X):
        """对样本集合做预测。"""
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree_) for x in X])


class SimpleRandomForestScratch:
    """简化版随机森林（分类）从零实现。

    为什么随机森林要“随机”？
    1) 对每棵树做 bootstrap 抽样（有放回采样），让树看到不同数据子集。
    2) 每次分裂只看随机子特征，降低树与树之间的相关性。

    多棵“低相关”树投票，通常比单棵树更稳健。
    但在高维稀疏 TF-IDF 文本上，线性模型（如逻辑回归）常更有优势，
    因为文本线性可分性较强且特征维度高，树模型可能难以充分利用稀疏信号。
    """

    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.trees_ = []

    def _resolve_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        return n_features

    def fit(self, X, y):
        """训练随机森林：多次 bootstrap + 训练多棵树。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        self.trees_ = []
        max_features_per_split = self._resolve_max_features(n_features)

        for i in range(self.n_estimators):
            indices = self.rng_.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = SimpleDecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features_per_split,
                random_state=int(self.rng_.integers(0, 1_000_000)),
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """对每棵树预测后做多数投票。"""
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        final_preds = []
        for j in range(X.shape[0]):
            col = all_preds[:, j]
            values, counts = np.unique(col, return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        return np.array(final_preds)


def evaluate_and_print(model_name, y_true, y_pred):
    """打印分类指标。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n==== {model_name} ====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")


def main():
    """小型可运行演示：加载样例数据，训练并评估三个 scratch 模型。"""
    data_path = "data/sample/research_queries_sample.csv"
    df = pd.read_csv(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("数据文件必须包含 text 和 label 两列。")

    texts = df["text"].astype(str).values
    labels = df["label"].astype(str).to_numpy()

    # TF-IDF 是文本任务常见特征工程：
    # - TF 体现词在文档内的重要程度
    # - IDF 抑制在全语料中“过于常见”的词
    # 对稀疏文本特征，TF-IDF + 逻辑回归通常是很强的基线。
    vectorizer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # 教学脚本里数据很小，因此转成稠密矩阵便于从零实现与调试。
    X = X_tfidf.toarray()
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    softmax_lr = SoftmaxLogisticRegressionScratch(lr=0.5, epochs=500, reg=1e-4, verbose=False)
    softmax_lr.fit(X_train, y_train)
    y_pred_lr = softmax_lr.predict(X_test)
    evaluate_and_print("Softmax Logistic Regression (Scratch)", y_test, y_pred_lr)

    nb = MultinomialNaiveBayesScratch(alpha=1.0)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    evaluate_and_print("Multinomial Naive Bayes (Scratch)", y_test, y_pred_nb)

    rf = SimpleRandomForestScratch(
        n_estimators=15,
        max_depth=4,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    evaluate_and_print("Random Forest (Scratch)", y_test, y_pred_rf)

    print("\n说明：本脚本仅用于学习与面试讲解（算法原理演示）。")
    print("生产环境 Stage 1 训练仍应使用 sklearn 的成熟实现，以保证可靠性与可维护性。")


if __name__ == "__main__":
    main()
