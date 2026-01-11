
⸻



你正在实现一个研究型代码库 ecggen（Python + PyTorch）。

项目的背景参考这个文件 /home/gbsguest/Research/boson/BIO/ecggen/prompts/background.md
一些技术上的定义参考这个文件 /home/gbsguest/Research/boson/BIO/ecggen/prompts/theory.md
主要tensor的 shape要参考这个 /home/gbsguest/Research/boson/BIO/ecggen/prompts/shape.md

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

代码要求：
- 只使用 PyTorch + numpy + Python 标准库
- 明确写出 tensor shape，并使用 assert
- 使用 type hints + docstring
- 英文写 注释，每个文件开头和每个class要写清楚注释

⸻
# 交互和 run_id_管理

所有的运行都通过main这个入口，main主要接受configs 的yaml参数，也接受少量的命令行参数修改 ,注意其中的模型参数和训练的参数要严格分离
每个run要对应runid，格式是 m#s#k# 
然后config要有一个 runid的对应表格，例如m1是对应什么，s2是对应什么 这个是由yaml中的config文件设置的,目前是 ecggen/configs/train/v1.yam

然后所有的model checkpoints 和 results 的保存格式都是 /runid_step#/ 下面保存


🧩 src/data/tokenizer.py

目标文件：src/data/tokenizer.py

实现 ECGTokenizer。

核心设计（必须严格遵守）：
- 只做 beat-wise tokenization
- 每个 beat 插值到等长 beat_len
- token 的 index 就是 beat index
- 额外保存每个 beat 的时间间隔信息

输入：
- ecg: torch.Tensor, shape [B, L, T]

输出：
- X: torch.Tensor, shape [B, L, N, d]
  - B: batch
  - L: lead 数（默认 12）
  - N: beat 数
  - d: token 维度
- meta: dict，至少包含：
  - "beat_intervals": torch.Tensor [B, N]
  - "beat_boundaries": list[list[(start, end)]]
  - "beat_len": int

实现要求：
- beat 分割方式：

  - 可选：rr 划分 参考 ecggen/prompts/refs/rr.py
  或者是等距划分 （直接滑动等距窗口划分）
- 每个 beat 使用线性插值变成 beat_len （如果等距就不用插值，直接就是 beat_len）
- token 特征：
  - 最简单版本：flatten waveform + Linear → d
  - 可选：小 Conv1D encoder
- 不需要 phase_map，不需要 warp




⸻

🧩 src/data/pipeline.py

目标文件：src/data/pipeline.py

预处理的部分参考这里 /home/gbsguest/Research/boson/BIO/ecggen/prompts/datapipline.md

实现数据集与 dataloader。

原始数据格式（统一）：
- ecg: torch.Tensor [B, L, T]
- label: torch.Tensor [B]（或 None）

实现内容：
- ECGDataset：
  - __getitem__ 返回：
    {
      "ecg": Tensor [L,T],
      "label": Optional[int],
      "id": str
    }

- make_dataloader(cfg, split) -> DataLoader
- collate_fn：
  - stack 成 [B,L,T]
  - T 一致



要求：
- 代码简单、清晰、可读
- 不引入复杂 augmentation

写清楚注释：这是一个研究用 pipeline，不是工业级


⸻

🧩 src/models/blocks.py

目标文件：src/models/blocks.py

实现可复用模型组件。

(A) TokenEncoder（保留）
- 输入：X [B,L,N,d]
- reshape → [B, L*N, d]
- 输出：H [B, L*N, d_model]
- 使用 TransformerEncoder
- 注释说明：这里只做 token-level interaction，不建模时间

(B) LeadProjection（重点写注释）
- 输入：VCG V [B,3,T’]
- learnable：
  - lead_vectors [L,3]（归一化）
  - gain [L]
  - bias [L]
- 输出：E_hat [B,L,T’]

注释必须解释：
- ECG lead 是 VCG 的线性观测
- lead vector 表示空间方向，而不是 embedding
- gain/bias 的物理直觉

(C) ResidualHead（CNN）
- 输入：base ECG [B,L,T’]
- 输出：residual [B,L,T’]
- 使用 depthwise + pointwise Conv1D
- 容量要小
- 注释说明：residual 只补偿 projection 的不足

(D) AngleCalib
- 不实现逻辑
- 只写 class skeleton + 超详细注释：
  - prefix beats 是什么
  - 角度为什么是 nuisance variable
  - 为什么不能影响 W 的学习
- forward() 直接 raise NotImplementedError

(E) SO3 utils
- rotation matrix / quaternion 工具
- 注释要解释几何意义，而不是只写公式


使用背景（文件头简单说明即可）：
- 这些旋转只用于方向校准（例如 VCG 或 lead direction）
- 旋转不是心脏状态，只是观测几何的修正
- 旋转应是小幅、刚体的（只改方向，不改幅值）

====================
需要的函数（就这几个）
====================

1) axis_angle_to_matrix

签名：
- axis_angle_to_matrix(r: Tensor[...,3]) -> Tensor[...,3,3]

注释要求（很简短）：
- r 的方向是旋转轴
- r 的模长是旋转角度
- 表示整体方向的微调

实现：
- Rodrigues 公式
- 小角度数值保护
- 输出应满足 RᵀR≈I

--------------------

2) apply_rotation

签名：
- apply_rotation(R: Tensor[...,3,3], v: Tensor[...,3] or [...,3,T]) -> Tensor

注释要求：
- 标准向量旋转
- 不改变向量长度
- 用于旋转 VCG 或 lead 向量

--------------------

3) rotation_magnitude（可选但很简单）

签名：
- rotation_magnitude(R: Tensor[...,3,3]) -> Tensor[...]

注释要求：
- 返回旋转“有多大”
- 可用于正则，防止旋转过大

====================
其他要求
====================

注释风格：
- 一两句话说明“几何意义 + 用途”


⸻

🧩 src/models/ttt.py

目标文件：src/models/ttt.py

实现 Test-Time Training (TTT)。

(A) FastState
- 支持两种模式（由 cfg 控制）：
  1) vector: W 是 [B,D]
  2) mlp: W 表示一个小 MLP 的参数
- 用 dataclass 或 nn.Module

注释必须解释：
- vector vs mlp 的建模差异
- 为什么 W 是“状态”，不是 hidden state

(B) TTTUpdater
- 输入：H [B, L*N, d]
- 按 beat chunk 切分
- 每个 chunk：
  - 用当前 chunk 的 reconstruction proxy 更新 W
- 只更新 W，不更新 encoder

不需要实现 self-supervised loss。

必须写清楚的注释：
- 这是 state estimation / filtering
- 不是 forecasting
- 没有 rollout，没有未来预测

实现：
- W_smooth 正则：||W_k - W_{k-1}||^2
- 用 torch.autograd.grad

写 __main__ 做 shape sanity check


⸻

🧩 src/models/vcg.py

目标文件：src/models/vcg.py

实现 VCG generator。

VCGGenerator：
- 输入：W [B,D]
- 输出：V [B,3,T’]

实现方式：
- basis 版本：
  - A(W): [B,3,K]
  - B: [K,T’]
  - V = A @ B

Regularizers（通过 cfg 控制）：
- smoothness
- energy
- loop_closure（可选）

注释解释：
- V 是 latent source，不是 ECG
- 为什么需要这些正则


⸻

🧩 src/models/ecggen.py

目标文件：src/models/ecggen.py

实现 ECGGenModel，总模型组装。

Pipeline（）：
1) ECG [B,L,T]
2) Tokenizer → X [B,L,N,d]
3) TokenEncoder → H [B,L*N,d]
4) TTT → W_final [B,D]
5) VCG → V [B,3,T’]
6) LeadProjection → E_hat’ [B,L,T’]
7) Residual（可选）

提供 forward：
- forward_gen
- forward_cls（只用 W）

注意：
- model 里不写 loss
- forward 返回 dict，包含中间结果


⸻

🧩 src/models/heads.py

目标文件：src/models/heads.py

实现分类头。

输入：
- W [B,D]

输出：
- logits [B,C]

支持：
- linear probe
- small MLP




⸻

🧩 src/eval.py

目标文件：src/eval.py

实现评估逻辑。

包含：
- reconstruction error
- missing-lead robustness
- linear probe accuracy


## 测试文件
- 对于数据获取和处理，模型 forward，要写具体的 test文件在 ecggen/tests 下面


⸻

⸻




# 注释规范

## 总原则
- 注释解释“建模思想”，不是逐行翻译代码
- 多解释为什么这样建模
- 明确区分：
  - representation（W）
  - rendering（VCG → ECG）
  - estimation（TTT）

## 每个 class 顶部必须回答 3 个问题
1. 这个模块在整体 pipeline 中的角色是什么？
2. 输入/输出在建模语义上表示什么？


## TTT 相关注释重点

  - rollout
- 强调：
  - quasi-static state
  - online estimation
  - not forecasting

## LeadProjection 注释重点
- ECG lead = VCG 的线性观测
- lead vector 是空间方向
- gain/bias 是观测尺度与偏置

## VCG 注释重点
- V 不是 ECG
- V 是共享 latent source
- basis 的意义

## AngleCalib 注释重点
- angle 是 nuisance variable
- prefix beats 的作用
- 不参与 W 的学习


