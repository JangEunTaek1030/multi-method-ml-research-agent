def confusion_matrix_binary(y_true, y_pred):
    """
    手撕二分类混淆矩阵。

    y_true: 真实标签列表，例如 [1, 1, 0, 1]
    y_pred: 预测标签列表，例如 [1, 0, 0, 1]

    返回：
    TP, FP, FN, TN
    """

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 0:
            tn += 1

    return tp, fp, fn, tn


def accuracy_score_scratch(y_true, y_pred):
    """
    Accuracy = 预测正确的样本数 / 总样本数
    """

    correct = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    return correct / len(y_true)


def precision_score_scratch(y_true, y_pred):
    """
    Precision = TP / (TP + FP)

    含义：
    在所有被模型预测为正类的样本里，有多少是真的正类。
    """

    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score_scratch(y_true, y_pred):
    """
    Recall = TP / (TP + FN)

    含义：
    在所有真实正类样本里，有多少被模型成功找出来。
    """

    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score_scratch(y_true, y_pred):
    """
    F1 = 2 * Precision * Recall / (Precision + Recall)

    含义：
    Precision 和 Recall 的调和平均数。
    """

    precision = precision_score_scratch(y_true, y_pred)
    recall = recall_score_scratch(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    # 构造你刚才的例子：
    # TP = 80
    # FP = 10
    # FN = 2
    # TN = 8

    y_true = []
    y_pred = []

    # 80 个真实怀孕，预测怀孕：TP
    y_true += [1] * 80
    y_pred += [1] * 80

    # 10 个真实不怀孕，预测怀孕：FP
    y_true += [0] * 10
    y_pred += [1] * 10

    # 2 个真实怀孕，预测不怀孕：FN
    y_true += [1] * 2
    y_pred += [0] * 2

    # 8 个真实不怀孕，预测不怀孕：TN
    y_true += [0] * 8
    y_pred += [0] * 8

    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)

    print("Confusion Matrix:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")

    print("\nMetrics:")
    print(f"Accuracy:  {accuracy_score_scratch(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score_scratch(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score_scratch(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score_scratch(y_true, y_pred):.4f}")