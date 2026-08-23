# Dataset

本项目使用聚合物玻璃化转变温度（Tg）数据进行机器学习建模。

数据包含：

- 7,367 个聚合物样本
- Experimental Tg
- PSMILES / BIGSMILES
- Polymer Class
- Backbone / Side-chain / Full-polymer molecular descriptors

模型最终使用 99 个数值型分子描述符预测 Tg。

## Data Source

本项目数据来源于 LAMALAB 的 **PolyMetriX Curated Glass Transition Temperature Dataset**。

官方数据集文档：

https://lamalab-org.github.io/PolyMetriX/datasets/

该数据集整合多个公开 Tg 数据源，并提供聚合物结构、实验 Tg、聚合物类别以及分层结构描述符等信息。

出于数据许可和仓库体积考虑，本仓库不直接重新分发原始数据。

请从官方来源获取数据，并放置于：

```text
data/raw/