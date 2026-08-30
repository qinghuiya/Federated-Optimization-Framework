# Federated Optimization Framework（联邦优化框架）

[English](README.md) · [入门教程](docs/GETTING_STARTED.md) · [算法说明](docs/ALGORITHMS.md) · [参与贡献](CONTRIBUTING.md)

这是一个面向联邦学习初学者、基于 PyTorch 的联邦优化教学与研究框架。它把“本地优化”和
“服务器端联邦优化”拆成清晰的独立模块，既方便第一次运行 FL 实验，也方便研究者替换算法。

> 这是一次干净的通用化重写。仓库刻意排除了 FedHodge、FedMZ、DriVE、未发表方法、历史实验
> 输出和任何私有研究资产。

## 为什么做这个框架

一个初学者要跑通第一次联邦学习实验，往往需要先处理客户端抽样、非 IID 数据划分、本地训练、
参数聚合、评估、配置和复现等大量基础工作。这个项目把这些通用部分准备好，让使用者能把时间
花在理解算法和验证想法上。

## 已支持的方法

- 经典联邦优化：FedSGD、FedAvg、FedAvgM、FedProx、SCAFFOLD、FedNova、FedAdagrad、
  FedAdam、FedYogi。
- 鲁棒聚合：坐标中位数、截尾均值、Krum。
- 本地优化器：SGD、Momentum、Nesterov、Adam、AdamW、Adagrad、Adadelta、RMSprop、
  Rprop、ASGD、RAdam、NAdam。
- 本地学习率调度：Step、MultiStep、Exponential、Cosine、Linear。
- 数据：无需下载的合成数据，以及 MNIST、FashionMNIST、CIFAR-10、CIFAR-100。
- 划分：IID、Dirichlet 非 IID、标签分片（pathological non-IID）。
- 模型：MLP、轻量 CNN、适用于小图像的 ResNet-18。

每一个对外宣称支持的联邦算法都有端到端冒烟测试。更详细的公式、限制和论文来源见
[算法说明](docs/ALGORITHMS.md)。

## 三分钟运行

```bash
git clone https://github.com/qinghuiya/Federated-Optimization-Framework.git
cd Federated-Optimization-Framework
python -m venv .venv
```

激活虚拟环境后：

```bash
pip install -e .
fedopt-train --config configs/synthetic_fedavg.yaml --output outputs/first-run
```

合成数据示例不需要联网下载数据。切换联邦算法或本地优化器也不需要改代码：

```bash
fedopt-train --config configs/synthetic_fedavg.yaml \
  --algorithm fedprox --optimizer momentum --rounds 10 \
  --output outputs/fedprox
```

运行测试：

```bash
pip install -e ".[dev]"
pytest
```

## 项目结构

- `federated_optimization/local.py`：本地优化器和学习率调度器注册表；
- `federated_optimization/client.py`：客户端训练，以及 FedProx、SCAFFOLD 的客户端逻辑；
- `federated_optimization/server.py`：聚合与服务器优化方法；
- `federated_optimization/data.py`：数据集与 IID/非 IID 划分；
- `federated_optimization/runner.py`：完整通信轮次和评估；
- `configs/`：可以直接运行的配置；
- `tests/`：公式级单元测试和端到端测试。

每次运行会生成 `resolved_config.yaml`、`metrics.jsonl`、`summary.json` 和
`final_model.pt`，便于复现和后续分析。

## 使用边界

本项目是单进程模拟框架，客户端依次在一个 CPU 或 GPU 上训练。它追求可读、可改、可验证，
不是生产环境的分布式安全聚合系统。目前主要支持监督分类，不宣称提供差分隐私、通信加密、
真实网络容错或大规模分布式性能。

SCAFFOLD 和 FedNova 当前明确要求无动量、无本地调度器的普通 SGD，以避免用不完整公式给出
看似可运行但数值含义错误的结果。

## 一起完善它

欢迎增加算法、数据集、模型、测试和中文教程。新增算法时，请同时提供原论文来源和最小数值
测试。详细流程见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [扩展指南](docs/EXTENDING.md)。

项目采用 [MIT License](LICENSE) 开源。
