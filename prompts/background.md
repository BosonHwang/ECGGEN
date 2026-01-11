# ecggen —— 项目背景与概念澄清（Project Background）

## 1. 项目核心问题（What problem are we solving）

本项目关注的不是“如何预测 ECG 波形”，  
而是一个更抽象的问题：

**如何用一个稳定、可解释的 latent representation 来刻画“心脏本身的状态”。**

传统 ECG 建模通常把 ECG 当作：
- 一维或多维 time series
- 目标是 forecasting、generation 或 classification

但在本项目中，我们采用完全不同的视角：

> **ECG 不是一个序列本体，而是对心脏这一“物理对象”的观测结果。**

因此，本项目的核心不是 sequence modeling，而是 **object-centric modeling**。

---

## 2. Heart representation W 的概念定位

### 2.1 W 是什么（What W represents）

在本项目中，**W 表示一个 latent heart state**，具有以下特征：

- 是latent representation
- 在短时间窗口内是 quasi-static（相对静态）
- 能同时解释：
  - 多个 beats
  - 多个 leads
- 是从观测中被“估计（estimated）”出来的

W 试图捕捉的是：

- 稳定的形态学倾向（morphological tendencies）
- 心脏的整体电活动状态
- 在多个 beats 之间共享的信息

例如（概念层面）：
- ST level 的整体偏移趋势
- T-wave 形态的系统性变化
- 个体层面的心脏特征

---

### 2.2 W 不是什么（What W does NOT represent）

非常重要的是，W **不是**：

- 某一个时间点的 waveform embedding
- 一个随时间自动演化的 hidden state
- 一个 RNN / Transformer 中的 recurrent memory
- 对未来 ECG 的预测变量

关键澄清一句话：

> **W 不会“自己变”，只能在看到新观测后被重新估计。**

---

## 3. Estimation vs Forecasting（核心区分）

本项目在概念上**严格区分**以下两种范式：

### Forecasting（我们不做的）
- 输入过去 → 输出未来
- 需要 time dynamics
- 通常依赖 autoregressive 或 recurrent 结构
- 误差会沿时间传播

### Estimation / Filtering（我们在做的）
- 输入当前观测 → 更新对系统状态的理解
- 不预测未来
- 不进行 rollout
- 每一步都在“解释已观测数据”

在 ecggen 中：

- 当新 beat 被观察到
- 模型通过 **Test-Time Training (TTT)** 更新 W
- 这个过程是 **online state estimation**
- 而不是 time-series prediction

---

## 4. Test-Time Training (TTT) 的语义澄清

在本项目中，TTT **不是一种 trick**，而是核心建模机制。

TTT 的作用是：

- 在不更新主模型参数的情况下
- 仅通过梯度更新 latent state W
- 使 W 更好地解释当前观测到的 ECG

重要的是：

- TTT 更新的是“解释变量”（W）
- 而不是“生成模型”或“编码器”

因此：

> **TTT 在这里应被理解为一种 state estimation 过程，而不是 training continuation。**

---

## 5. Beat-wise Tokenization 的设计动机

ECG 的自然结构单位是 **beat**，而不是 sample 或 time step。

因此本项目采用：

- beat-wise tokenization
- 每个 beat 被归一化到等长表示
- token 的 index 是 beat index，而不是时间 index

这意味着：


Beat-to-beat 的变化不是通过 sequence dynamics 建模，
而是通过 **W 在不同 beat 上的重新估计**体现。

---

## 6. Multi-lead ECG 的几何视角

在本项目中，多导联 ECG 被理解为：

> **同一个 latent cardiac source 在不同观测方向下的投影结果。**

具体而言：

- 存在一个低维的 latent source（VCG-like）
- ECG leads 是对该 source 的线性观测
- 每个 lead 对应一个固定的空间方向（lead vector）

因此：

- lead parameters 描述的是观测几何（geometry）
- 而不是语义特征（semantics）

---

## 7. Geometry as Nuisance Variable

与心脏状态 W 不同，几何因素（geometry）包括：

- lead orientation
- electrode placement
- global cardiac axis rotation

这些因素在本项目中被视为 **nuisance variables**：

- 它们影响观测结果
- 但不应承载心脏语义
- 不应主导 representation learning

任何 Angle Calibration / rotation 模块都应：

- 维度极低
- 幅度受限
- 不干扰 W 的估计

---

## 8. Rendering 而非 Generation

ecggen 的输出过程应被理解为 **rendering**：

- 给定 W
- 通过 latent source（VCG）
- 再通过 lead projection
- 得到 ECG waveform

这里不存在：

- autoregressive generation
- sampling future
- sequence rollout

一句话总结：

> **模型不是在“生成 ECG”，而是在“渲染 ECG”。**

---

## 9. 项目刻意避免的建模范式

本项目明确避免以下做法：

- RNN / LSTM / GRU
- Autoregressive decoding
- Next-token prediction
- Phase-based continuous-time modeling
- 把 ECG 当作普通 time series

这些并非工程限制，而是概念选择。

---

## 10. 给实现者（Cursor / LLM）的思维准则

在实现任何模块时，请始终自问：

1. 这个模块是在估计 heart state，还是在预测未来？
2. 是否引入了隐式的时间递归？
3. 是否把 nuisance geometry 当成了语义？
4. 是否破坏了 W 的“相对静态”假设？

如果答案偏向 forecasting 或 recurrence，说明设计方向是错误的。

---

## 11. 一句话总结整个项目

> **ecggen 将 ECG 建模为对一个缓慢变化的 latent heart state 的渲染观测，而不是一个需要被预测的时间序列。**