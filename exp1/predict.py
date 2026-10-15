"""按验证集选定的MLP配置重新训练，生成最终测试集提交文件。"""
from pathlib import Path
import json
import hashlib
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

data_dir = Path(__file__).resolve().parent
train_path = data_dir / "train_data.csv"
test_path = data_dir / "test_data_unlabeled.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 最终方案已通过验证集选定，此处使用全部有标签数据，不再调参。
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train["text"].astype(str))
X_test = vectorizer.transform(test["text"].astype(str))
classifier = MLPClassifier(
    hidden_layer_sizes=(100,), alpha=0.0001, max_iter=300, random_state=42
)
print(f"全量训练：{len(train)}条；测试预测：{len(test)}条", flush=True)
start = perf_counter()
# 固定数值计算的线程数，避免过多线程拖慢训练。
with threadpool_limits(limits=2):
    classifier.fit(X_train, train["target"])
    predictions = classifier.predict(X_test)
elapsed = perf_counter() - start

# 保持测试集原始顺序；不输出表头和索引。
assert len(predictions) == len(test), "预测行数与测试集不一致"
assert set(predictions).issubset(set(train["target"])), "预测包含未知类别"
output = data_dir / "prediction.csv"
pd.DataFrame(predictions).to_csv(output, index=False, header=False)

# 另存生成记录和训练损失，便于复查；不放进课程平台的两文件压缩包。
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metadata = {
    "model": "MLPClassifier", "hidden_layer_sizes": [100], "alpha": 0.0001,
    "max_iter": 300, "random_state": 42, "max_features": 5000,
    "train_rows": len(train), "test_rows": len(test), "features": X_train.shape[1],
    "n_iter": classifier.n_iter_, "seconds_fit_and_predict": elapsed, "threads": 2,
    "python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
    "prediction_sha256": sha256(output),
    "test_accuracy": None, "note": "测试集无真实标签，无法计算测试准确率。"
}
(data_dir / "final_training.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
pd.DataFrame({"epoch": range(1, len(classifier.loss_curve_) + 1),
              "loss": classifier.loss_curve_}).to_csv(data_dir / "final_mlp_loss.csv", index=False)
print(f"已保存 {output.name}：{len(predictions)}行，训练{classifier.n_iter_}轮，耗时{elapsed:.1f}秒")
