import pandas as pd
from pathlib import Path
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# 选择本次实验；批量对比使用同一份数据和TF-IDF特征
print("1. SVM：自己选择C")
print("2. 朴素贝叶斯：自己选择alpha")
print("3. 批量对比：SVM四组C + 朴素贝叶斯三组alpha")
print("4. 逻辑回归：自己选择C")
print("5. MLP：自己选择alpha（一个100神经元隐藏层）")
print("6. 批量补测：逻辑回归三组C + MLP两组alpha")
print("7. 批量对比全部四种模型")
choice = input("请输入编号：").strip()

try:
    if choice == "1":
        value = float(input("请输入C（直接回车使用1）：").strip() or "1")
        experiments = [("SVM", value)]
    elif choice == "2":
        value = float(input("请输入alpha（直接回车使用1）：").strip() or "1")
        experiments = [("NB", value)]
    elif choice == "3":
        experiments = [("SVM", c) for c in [0.1, 1, 10, 100]]
        experiments += [("NB", a) for a in [1, 0.1, 0.01]]
    elif choice == "4":
        value = float(input("请输入C（直接回车使用1）：").strip() or "1")
        experiments = [("LR", value)]
    elif choice == "5":
        value = float(input("请输入MLP的alpha（直接回车使用0.0001）：").strip() or "0.0001")
        experiments = [("MLP", value)]
    elif choice in ["6", "7"]:
        experiments = [("LR", c) for c in [0.1, 1, 10]]
        experiments += [("MLP", a) for a in [0.0001, 0.01]]
        if choice == "7":
            experiments = [("SVM", c) for c in [0.1, 1, 10, 100]] + [("NB", a) for a in [1, 0.1, 0.01]] + experiments
    else:
        raise ValueError("请选择1到7")
    if any(value <= 0 for name, value in experiments):
        raise ValueError("参数必须大于0")
except ValueError as error:
    raise SystemExit(f"输入有误：{error}")

data_dir = Path(__file__).resolve().parent

# 1. 读取原始有标签数据
data = pd.read_csv(data_dir / "train_data.csv")

X = data["text"].astype(str)  # 文本：模型要读的内容
y = data["target"]           # 标签：每篇文本的正确类别

# 2. 分成训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,       # 留出20%用于验证
    random_state=42,    # 固定随机划分，方便复现
    stratify=y          # 尽量保持各类别的比例一致
)

# 3. 检查划分结果
print("原始数据数量：", len(X))
print("训练集数量：", len(X_train))
print("验证集数量：", len(X_val))

# 4. TF-IDF：把文本转换成数字特征，最多保留5000个词语特征
vectorizer = TfidfVectorizer(max_features=5000)

# 只从训练集学习词表和IDF，并转换训练文本
X_train_tfidf = vectorizer.fit_transform(X_train)

# 验证集使用训练集学到的同一套规则，不重新学习
X_val_tfidf = vectorizer.transform(X_val)

print("训练集特征形状（文本数，特征数）：", X_train_tfidf.shape)
print("验证集特征形状（文本数，特征数）：", X_val_tfidf.shape)

# 5. 按实验要求中的参考代码创建模型，再训练和验证
current_results = []
for model_name, value in experiments:
    if model_name == "SVM":
        classifier = SVC(kernel="linear", C=value, random_state=42)
        parameter = f"C={value}"
    elif model_name == "NB":
        # alpha=1对应参考代码MultinomialNB()的默认设置
        classifier = MultinomialNB(alpha=value)
        parameter = f"alpha={value}"
    elif model_name == "LR":
        classifier = LogisticRegression(C=value, max_iter=1000, random_state=42)
        parameter = f"C={value}"
    else:
        # 参考代码的100神经元隐藏层和300次最大迭代；这里比较L2正则化强度
        # MLP的alpha是正则化系数，与朴素贝叶斯的平滑参数含义不同
        classifier = MLPClassifier(hidden_layer_sizes=(100,), alpha=value, max_iter=300, random_state=42)
        parameter = f"alpha={value}"

    print(f"\n--- 开始训练 {model_name}：{parameter} ---", flush=True)
    start = perf_counter()
    classifier.fit(X_train_tfidf, y_train)
    train_seconds = perf_counter() - start
    print("模型训练完成，正在检查训练集和验证集成绩……", flush=True)

    y_pred = classifier.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train_tfidf))
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    correct = int((y_pred == y_val).sum())
    print(f"训练准确率：{train_accuracy:.2%}；验证准确率：{accuracy:.2%}")
    print(f"验证Macro-F1：{macro_f1:.4f}；预测正确：{correct}/{len(y_val)}")
    print(f"训练耗时：{train_seconds:.2f}秒", flush=True)

    # 6. 每完成一组就保存，避免后面的实验中断导致前面的成绩丢失
    result = {
        "model": model_name, "kernel": "linear" if model_name == "SVM" else "",
        "C": value if model_name in ["SVM", "LR"] else None,
        "max_features": vectorizer.max_features, "test_size": 0.2,
        "random_state": 42, "train_size": len(X_train), "val_size": len(X_val),
        "train_accuracy": train_accuracy, "val_accuracy": accuracy,
        "val_correct": correct, "train_seconds": train_seconds,
        "alpha": value if model_name in ["NB", "MLP"] else None, "val_macro_f1": macro_f1,
        "hidden_layer_sizes": "(100,)" if model_name == "MLP" else "",
        "max_iter": classifier.max_iter if model_name in ["LR", "MLP"] else None,
        "n_iter": int(max(classifier.n_iter_)) if model_name == "LR" else (classifier.n_iter_ if model_name == "MLP" else None)
    }
    result_path = data_dir / "experiment_results.csv"
    old_results = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    all_results = pd.DataFrame(old_results.to_dict("records") + [result])
    all_results.to_csv(result_path, index=False)
    current_results.append({"模型": model_name, "参数": parameter, "验证准确率": accuracy})
    if model_name == "MLP":
        # 留下真实的逐轮损失，方便报告绘图和检查训练过程
        pd.DataFrame({"epoch": range(1, len(classifier.loss_curve_) + 1), "loss": classifier.loss_curve_}).to_csv(
            data_dir / f"mlp_loss_alpha_{value}.csv", index=False)
    # 留下验证预测，供分类错误分析；不是最终测试集提交文件
    pd.DataFrame({"row_index": X_val.index, "true_label": y_val.values, "predicted_label": y_pred}).to_csv(
        data_dir / f"validation_{model_name}_{value}.csv", index=False)

print("\n本次实验汇总（按验证准确率排序）：")
print(pd.DataFrame(current_results).sort_values("验证准确率", ascending=False).to_string(index=False))
print("成绩已保存到：", result_path)
