你正在实现一个研究型代码库 ecggen（Python + PyTorch）。

项目的背景参考这个文件 /home/gbsguest/Research/boson/BIO/ecggen/prompts/background.md
一些技术上的定义参考这个文件 /home/gbsguest/Research/boson/BIO/ecggen/prompts/theory.md
主要tensor的 shape要参考这个 /home/gbsguest/Research/boson/BIO/ecggen/prompts/core/shape.md

研究目标：
- 通过 Test-Time Training (TTT) 学习一个相对静态的心脏表示 W
- 通过 VCG（3D latent source）+ Lead Projection 渲染 ECG
- 支持 multi-lead 预训练、角度校准（Angle Calibration）和下游分类任务

非目标（非常重要）：
- 不追求严格的生理 VCG 模型
- 不做 autoregressive / forecasting / next-token prediction

建模原则：
- Heart-as-object：心脏是一个 latent object（W），不是时间序列
- Beat-wise tokenization：token 以 beat 为 index
- Beat-to-beat 漂移通过 TTT 的在线状态估计完成，而不是时间动力学
- 所有模块都必须是 non-recurrent 的

# 代码要求：

- **明确写出 tensor shape, 对关键部分的 tensor shape 和每个维度的具体含义注释得非常清晰**
- 使用 type hints + docstring
- 英文写 注释，每个文件开头和每个class要写清楚注释
