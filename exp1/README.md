# 实验一 文本分类

作者：鲍鑫宇 10243330404
使用TF-IDF表示新闻文本，比较线性SVM、朴素贝叶斯、逻辑回归和MLP。模型实现参考课程《实验一要求》中的示例代码。

## 文件说明

- `main.py`：从原始数据划分训练集和验证集，按菜单运行模型和参数对比。
- `predict.py`：用选定的MLP配置在全部有标签数据上重新训练，生成测试预测 `prediction.csv`。
- `requirements.txt`：本实验使用的依赖版本。
- `results/`：已完成实验的成绩、两组MLP损失、最佳MLP验证预测，以及最终全量训练的生成记录。

## 准备数据和环境

使用Python 3.10（本实验为3.10.21）。在本目录打开终端，安装依赖：

```powershell
python -m pip install -r requirements.txt
```

使用老师提供的原始数据，将以下两个文件放在本目录，与 `main.py` 同级：

```text
exp1/
  main.py
  predict.py
  train_data.csv
  test_data_unlabeled.csv
```

`train_data.csv` 包含 `text` 和 `target` 列，共7368条；
`test_data_unlabeled.csv` 包含 `text` 列，共2457条。

## 训练和验证

```powershell
python main.py
```

输入菜单编号：

1. SVM，自选C。
2. 朴素贝叶斯，自选alpha。
3. 批量比较SVM和朴素贝叶斯。
4. 逻辑回归，自选C。
5. MLP，自选L2正则化alpha。
6. 批量比较逻辑回归和MLP。
7. 批量运行全部四种模型，共12组配置。

训练/验证按80%/20%分层划分，random_state=42。TF-IDF最多5000个词项。SVM使用线性核；逻辑回归max_iter=1000；MLP为100个神经元的单隐藏层、max_iter=300，其余采用库默认设置。

终端显示训练准确率、验证准确率、Macro-F1、正确数和训练耗时。`experiment_results.csv` 在本目录保留各次成绩，重复实验追加记录；验证预测与MLP损失也保存在本目录，同一配置再次运行会更新对应文件。`results/` 中的历史记录保留供报告复查。

## 已有验证结果

| 模型 | 最佳已测参数 | 验证准确率 |
| --- | --- | --- |
| SVM | C=100 | 91.04% |
| 朴素贝叶斯 | alpha=0.1 | 89.15% |
| 逻辑回归 | C=10 | 91.32% |
| MLP | alpha=0.0001 | 91.45% |
