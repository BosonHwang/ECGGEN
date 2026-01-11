**ecgGen**

**pretrain （multi-lead，自监督，一致性约束）**

**输入：**

**E               # 12-lead ECG 信号，shape [B, L=12, T]**

- **-----------------------------------------------**

**1) Lead Dropout / Channel Mask（制造多视角监督）**

- **-----------------------------------------------**

**从 12 个 lead 中随机选择一部分作为上下文：**

**L_ctx** ⊂ **{1..12}            # 例如随机选 4~8 个 lead**

**E_ctx = E[:, L_ctx, :]     # 作为模型输入**

**E_tgt = E                  # 目标始终是重建完整 12-lead**

**目的：**

**- 强制模型不能记忆某个固定 lead**

**- 逼迫其学习一个能解释所有 lead 的 shared heart state**

- **-----------------------------------------------**

**2) ECG Tokenizer（beat-wise，lead space）**

- **-----------------------------------------------**

**对每个 lead 的 ECG 进行 beat 切分：**

**beats = split_into_beats(E_ctx)**

**每个 beat 进行长度归一化 / 幅值缩放，得到心动周期 token**

**同时保存：**

**- beat_boundaries          # 用于后续对齐（不参与反传）**

**将 beat + lead 信息编码成 token：**

**X = encode_beats(beats, lead_id)**

**X shape = [B, N, d]**

**说明：**

**- token 明确定义在 lead / observation space**

**- 不涉及 VCG / angle 的任何假设**

- **-----------------------------------------------**

**3) Encoder（Transformer Encoder，而不是 LM Decoder）**

- **-----------------------------------------------**

**使用 token-wise Transformer Encoder 建模局部形态与时序关系：**

**H = TransformerEncoder(X)**

**H shape = [B, N, d]**

**说明：**

**- 这里不做 next-token language modeling**

**- 主要目的是提取稳定特征，供 TTT 估计 heart state**

- **-----------------------------------------------**

**4) TTT（global heart representation W）**

- **-----------------------------------------------**

**初始化 fast weights：**

**W0                          # 每条 record / window 一个 W**

**按 token chunk 进行 inner-loop 更新：**

**for each chunk c in H:**

**L_self = self_loss(h_c; W)   # 可用 masked regression / cross-lead pred**

**W ← W - η ∇_W L_self**

**得到最终：**

**W*                          # global、quasi-static heart representation**

**关键设计：**

**- W 是 record-level / window-level 的**

**- 不是 per-token 的 hidden state**

**- 通过 TTT 学到“解释整段 ECG 的心脏状态”**

- **-----------------------------------------------**

**5) VCG Generator（从 W 生成 latent 3D source）**

- **-----------------------------------------------**

**使用 time / phase + W 生成 VCG：**

**V = G([Fourier(phase_grid), W*])**

**输出：**

**V** ∈ **R[B, 3, T’]             # 3D Vectorcardiogram trajectory**

**说明：**

**- VCG 是 latent physical source**

**- time/phase 负责动态**

**- W 负责 heart identity / physiology**

- **-----------------------------------------------**

**6) Lead Projection（低容量，近似物理投影）**

- **-----------------------------------------------**

**为每个 lead 学一个 3D 方向向量：**

**u_ℓ** ∈ **R^3**

**l_ℓ = normalize(u_ℓ)        # 强制 ||l_ℓ|| = 1**

**每个 lead 的 ECG 由 VCG 投影得到：**

**ê_ℓ(t) = a_ℓ * (l_ℓᵀ V(t)) + b_ℓ**

**得到：**

**E_hat’ = stack(ê_1 ... ê_12)    # shape [B, 12, T’]**

**设计动机：**

**- 强制所有 lead 共享同一个 VCG**

**- 防止黑盒 GEO Transformer 吃掉物理结构**

- **-----------------------------------------------**

、

- **-----------------------------------------------**

**8) Loss 设计（Phase 1 够用版）**

- **-----------------------------------------------**

**-----**

**输出：**

**W*        # 全局 heart representation（后续用于分类 / 生成 / probe）**

**V         # latent VCG（可视化 / 任意角度生成）**

**E_hat     # 重建的 12-lead ECG**

**#####	#####**

**问题！！！！**

**##########**

**【1】VCG 生成质量如何保证（VCG generation quality）**

**核心原则：**

**VCG 不是追求“真实物理完全一致”，**

**而是作为 multi-lead 的 shared latent source，被结构性约束出来。**

**关键设计：**

**1) 低自由度生成（low-DOF generator）**

**- V(t) = G([phase(t), W])**

**- 使用 Fourier / spline / INR（带强正则）**

**- 避免完全自由的 unconstrained MLP**

**2) 强投影约束（strong projection bottleneck）**

**- ECG_lead(t) ≈ l_leadᵀ · V(t)**

**- projection 必须是低容量（dot product / small linear）**

**- 防止 decoder 吃掉 VCG 语义**

**3) 平滑与能量正则（regularization）**

**- temporal smoothness / curvature penalty**

**- energy / norm constraint，避免 scale 漂移**

**4) 允许但限制 residual（inevitable imperfection）**

**- Ê = projection(VCG) + residual**

**- residual 必须低容量 + 强正则**

**- residual 只补细节，不能主导重建**

**判据（不是视觉像不像）：**

**- VCG 是否稳定**

**- 是否能一致解释多个 lead**

**- missing-lead / domain shift 下是否仍有效**

- **-----------------------------------------------------------**

**【2】角度校准（angle calibration）如何做才不破坏结构**

**核心原则：**

**angle / lead geometry 是 nuisance，**

**只能用于几何对齐，不能承担表征能力。**

**合理做法：**

**1) 输入：**

**- 少量 prefix tokens（1–2 beats / 短窗口）**

**- 不使用长时间上下文，避免泄露心脏形态**

**2) 输出（必须低维、受约束）：**

**- 最优：3D rotation ΔR（SO(3) / axis-angle / quaternion）**

**- 可选：per-lead gain / bias**

**- 禁止：高维 angle embedding**

**3) 作用位置：**

**- 修正 lead direction： l ← ΔR · l**

**或**

**- 对齐 VCG 坐标系： V ← ΔR · V**

**4) 训练约束：**

**- small-correction prior（偏好小旋转）**

**- 同一 record 的 angle 一致性**

**- 有 angle 标注时仅监督 calibration 模块**

**必须避免：**

**- angle 模块直接参与分类**

**- angle 输出作为 W 的替代**

**- angle 模块吸收 waveform 解释能力**

- **-----------------------------------------------------------**

**一句话总结（implementation mental model）：**

**VCG：**

**用“低自由度 + 强投影 + 正则”被迫学成 shared source**

**Angle calibration：**

**只做低维几何对齐，不做 representation**

**#########**

**问题**

**#############**

**【问题核心】**

**phase 只描述 beat 内坐标，但真实 ECG 中存在 beat-to-beat 的因果变化（如 ST 漂移）。**

**单一、完全静态的 W 确实不足以表达这种中时间尺度变化。**

- **-----------------------------------------------**

**【关键澄清】**

**1) beat-to-beat 的 ST 漂移 ≠ forecasting（预测未来）**

**- forecasting：不看未来观测，rollout 未来状态**

**- 这里：每看到一个新 beat，用观测“重新估计”当前状态**

**2) 正确视角：**

**这是 state estimation / filtering，而不是 time-series forecasting**

- **-----------------------------------------------**

**【正确的状态分层（多时间尺度）】**

- **phase：**

**- beat 内坐标（快变量）**

**- 只负责 intra-beat 形态展开**

**- 不携带跨 beat 记忆**

- **W_k（heart state）：**

**- beat-level / chunk-level 的慢变量**

**- 表示 repolarization、ST level 等中尺度状态**

**- 允许随 beat 缓慢变化（piecewise-slow）**

- **-----------------------------------------------**

**【W 如何随 beat 演化（关键点）】**

**不是：**

**W_{k+1} = f(W_k)        # 预测式动力学（不要）**

**而是：**

**W_k = argmin_W  L(E^{(k)}, render(W))**

**+ λ ||W - W_{k-1}||^2**

**即：**

- **每个 beat / chunk 用当前观测通过 TTT 更新 W**
- **加慢变正则，得到平滑的 W_k 轨迹**
- **没有 forward rollout → 不是预测**
- **-----------------------------------------------**

**【ST 漂移在模型中的来源】**

- **ST 漂移 = W_k 的 gradual movement**
- **不是 phase 累积**
- **不是 hidden state 传播**
- **而是“持续用新观测修正心脏状态”**
- **-----------------------------------------------**

**【一句话总括（核心认知）】**

**ST 漂移不是时间外推问题，**

**而是一个“随着新观测不断被修正的潜在状态”问题。**

**phase 负责坐标展开，W 负责跨 beat 的心脏状态，TTT 负责在线估计。**