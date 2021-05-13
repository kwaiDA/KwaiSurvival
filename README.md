1.软件简介

Python KwaiSurvival生存分析软件，简称 Python ksurv Package，Python—— Kwai Survival Model Package，简称 ksurv package，是一款基于 Python 编程环境开发的生存分析包，可用于基于深度学习的生存分析模型， 帮助使用者在 Python 中更高效地使用生存模型实现大数据分析。

2.安装需求

使用该软件需要安装 Python3及TensorFlow2.0+版本（推荐2.2）

3.软件功能

DeepSurv model;

DeepHit model;

Neural Multitask logistic model;

基于KM的纠偏生存曲线

4.使用介绍

生存分析是统计的一个分支，用于分析直到一个或多个事件（例如生物体死亡和机械系统故障）发生之前的预期持续时间。该主题称为工程学中的可靠性理论或可靠性分析，经济学中的持续时间分析或持续时间建模以及社会学中的事件历史分析。生存分析试图回答某些问题，例如将在一定时间内生存的人口比例是多少？在那些幸存者中，他们将以何种速度死亡或失败？可以考虑多种死亡或失败原因吗？特定情况或特征如何增加或降低生存概率？

为了回答这样的问题，有必要定义“寿命”。在生物生存的情况下，死亡是明确的，但是对于机械可靠性而言，故障可能无法明确定义，因为可能存在机械系统，其中部分故障，程度问题或未及时定位。即使在生物学问题上，某些事件（例如心脏病发作或其他器官衰竭）也可能具有相同的歧义。下面概述的理论假设在特定时间定义明确的事件；对于其他情况，可以使用明确说明模棱两可事件的模型来更好地处理。

更普遍地说，生存分析涉及到事件数据时间的建模。在这种情况下，死亡或衰竭被视为生存分析文献中的“事件” ——传统上，每个受试者仅发生一个事件，此后该生物或机制死亡或破裂。重复事件或重复事件模型可以放宽该假设。重复事件的研究与系统可靠性以及社会科学和医学研究的许多领域有关。

本软件集成了三种利用深度学习完成生存分析的前沿模型，论文标题和对应的模块如下：

DeepSurv- Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network —— DeepSurv

DeepHit- A Deep Learning Approach to Survival Analysis with Competing Risks—— DeepHit

Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework—— DeepMultiTasks
