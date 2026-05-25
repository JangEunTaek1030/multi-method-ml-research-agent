import math
# 导入 math，因为后面计算 IDF 时要用 log 对数函数


def build_vocab(docs):
    # build_vocab 的作用：
    # 根据训练文档 docs 建立一个词表 vocabulary
    # 词表的形式是：单词 -> 编号
    # 例如 {'cat': 0, 'likes': 1, 'milk': 2}

    vocab = {}
    # 创建一个空字典，用来存储词表

    for doc in docs:
        # 遍历每一篇文档
        # doc 是一个字符串，比如 "cat likes milk"

        for word in doc.split():
            # doc.split() 会按空格切词
            # "cat likes milk" -> ["cat", "likes", "milk"]
            # 然后逐个遍历里面的 word

            if word not in vocab:
                # 如果这个词还没有进入词表

                vocab[word] = len(vocab)
                # 给这个词分配一个编号
                # len(vocab) 当前是多少，就给它编号多少
                # 第一个词编号 0，第二个词编号 1，以此类推

    return vocab
    # 返回最终词表


def compute_tf(doc, vocab):
    # compute_tf 的作用：
    # 计算一篇文档的 TF 向量
    # TF = Term Frequency，词频
    # 简单说就是：每个词在当前文档中出现的频率

    words = doc.split()
    # 把当前文档按空格切成单词列表

    vector = [0.0] * len(vocab)
    # 创建一个全 0 向量
    # 向量长度等于词表大小
    # 例如 vocab 有 6 个词，那么 vector = [0, 0, 0, 0, 0, 0]

    for word in words:
        # 遍历当前文档中的每个词

        if word in vocab:
            # 如果这个词在训练集词表里
            # 注意：测试集中可能出现训练集没见过的词，这种词会被忽略

            idx = vocab[word]
            # 找到这个词在词表中的编号

            vector[idx] += 1.0
            # 对应位置加 1
            # 表示这个词出现了一次

    total_words = len(words)
    # 当前文档总共有多少个词

    if total_words == 0:
        # 如果文档是空的，为了避免除以 0，直接返回原始全 0 向量
        return vector

    vector = [count / total_words for count in vector]
    # 把“出现次数”转换成“出现频率”
    # 例如 "cat likes milk" 一共 3 个词
    # cat 出现 1 次，所以 TF(cat) = 1 / 3

    return vector
    # 返回当前文档的 TF 向量


def compute_idf(docs, vocab):
    # compute_idf 的作用：
    # 根据训练集计算每个词的 IDF
    # IDF = Inverse Document Frequency，逆文档频率
    # 它衡量一个词在整个语料库中有多稀有

    N = len(docs)
    # N 是训练文档总数

    idf = [0.0] * len(vocab)
    # 创建一个全 0 向量，用来存每个词的 IDF
    # 长度等于词表大小

    for word, idx in vocab.items():
        # 遍历词表中的每个词
        # word 是单词
        # idx 是这个单词对应的编号

        df = 0
        # df = document frequency
        # 表示有多少篇文档包含这个词

        for doc in docs:
            # 遍历训练集中的每篇文档

            words = set(doc.split())
            # 把当前文档切词后转成 set
            # set 的作用是去重
            # 因为 df 只关心“一篇文档里有没有这个词”
            # 不关心这个词在这篇文档里出现了几次

            if word in words:
                # 如果当前文档包含这个词

                df += 1
                # 文档频率加 1

        idf[idx] = math.log((N + 1) / (df + 1)) + 1
        # 计算 IDF
        #
        # 公式：
        # idf = log((N + 1) / (df + 1)) + 1
        #
        # N 是总文档数
        # df 是包含这个词的文档数
        #
        # df 越大，说明这个词越常见，IDF 越低
        # df 越小，说明这个词越稀有，IDF 越高
        #
        # +1 是为了平滑，避免除以 0，也让数值更稳定

    return idf
    # 返回每个词的 IDF 向量


def transform_tfidf(docs, vocab, idf):
    # transform_tfidf 的作用：
    # 把一组文档转换成 TF-IDF 向量

    X = []
    # X 用来存储所有文档的 TF-IDF 向量

    for doc in docs:
        # 遍历每一篇文档

        tf = compute_tf(doc, vocab)
        # 先计算当前文档的 TF 向量

        tfidf = [tf[i] * idf[i] for i in range(len(vocab))]
        # TF-IDF = TF * IDF
        # 对词表里的每一个词，都用：
        # 当前文档中的词频 × 这个词在训练集中的稀有程度

        X.append(tfidf)
        # 把当前文档的 TF-IDF 向量加入 X

    return X
    # 返回所有文档的 TF-IDF 特征矩阵


if __name__ == "__main__":
    # 这行的意思是：
    # 只有当这个文件被直接运行时，下面的代码才会执行
    # 如果这个文件被别的文件 import，下面的代码不会自动执行

    train_docs = [
        "cat likes milk",
        "dog likes bone",
        "cat likes fish",
    ]
    # 训练集文档
    # 注意：词表和 IDF 都只能从训练集学习

    test_docs = [
        "cat likes bone",
    ]
    # 测试集文档
    # 测试集只能使用训练集学到的 vocab 和 idf

    vocab = build_vocab(train_docs)
    # 用训练集建立词表
    # 这相当于 sklearn 里的 fit

    idf = compute_idf(train_docs, vocab)
    # 用训练集计算每个词的 IDF
    # 这也属于 fit 的一部分

    X_train = transform_tfidf(train_docs, vocab, idf)
    # 把训练集转换成 TF-IDF 向量
    # 这相当于 fit_transform 里的 transform 部分

    X_test = transform_tfidf(test_docs, vocab, idf)
    # 把测试集转换成 TF-IDF 向量
    # 注意：这里不能重新 build_vocab，也不能重新 compute_idf
    # 否则就是数据泄露

    print("Vocabulary:")
    print(vocab)
    # 打印词表

    print("\nIDF:")
    print(idf)
    # 打印每个词的 IDF

    print("\nTrain TF-IDF:")
    for row in X_train:
        print(row)
    # 打印训练集每篇文档的 TF-IDF 向量

    print("\nTest TF-IDF:")
    for row in X_test:
        print(row)
    # 打印测试集文档的 TF-IDF 向量