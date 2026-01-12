# ECG 多导联重构任务 - 完整实现规范

## 1. 项目背景与目标

### 1.1 核心任务
实现一个基于 VCG（心向量图）的多导联 ECG 重构模型。具体来说：
- **输入**：12 导联 ECG 信号，其中随机选择 3 个导联作为可见输入，其余 9 个导联被 mask（置零）
- **输出**：重构完整的 12 导联 ECG 信号
- **核心思想**：通过 3 个可见导联恢复 VCG（三维心电向量），再从 VCG 投影回 12 导联

### 1.2 建模哲学
- **Heart-as-Object**：心脏是一个物理对象，ECG 是对其电活动在不同方向上的投影观测
- **VCG 作为桥梁**：VCG 是 12 导联 ECG 的共享 latent source
- **几何约束**：每个导联对应一个固定的三维方向向量，由方位角和仰角决定

### 1.3 非目标
- **不做时间预测**：不做 autoregressive / forecasting / next-token prediction
- **不做 TTT**：TTT 模块暂不激活，作为可插拔插件保留
- **不加空间正则**：不对 VCG 添加 smoothness / energy / loop_closure 等正则

---

## 2. 完整架构设计

### 2.1 数据流总览

```
[Input: 12 Lead ECG, shape [B, 12, T]]
         ↓
[Random Mask: 选择 3 个可见导联，9 个置零]
         ↓
[Patch Embedding: 等距分割 + Linear 投影]
   输出: [B, 12, N, d]  (N = T // patch_size)
         ↓
[Reshape: 展平 Lead 维度]
   输出: [B*12, N, d]
         ↓
[Temporal Transformer: 时序建模，参数共享]
   输出: [B*12, N, d']
         ↓
[Reshape: 恢复 Lead 维度]
   输出: [B, 12, N, d']
         ↓
[Linear Unpatch: 恢复时间维度]
   输出: [B, 12, T]  (T 必须与输入对齐)
         ↓
[Select 3 Visible Leads]
   输出: [B, 3, T]
         ↓
[VCG Pseudo-Inverse: 几何恢复 VCG，无可学习参数]
   输出: [B, 3, T]  (VCG 三维轨迹)
         ↓
[Geometric Lead Projection: 几何投影到 12 导联]
   输出: [B, 12, T]
         ↓
[Decoder/Refinement: 可学习的细化网络]
   输出: [B, 12, T]  (最终重构)
         ↓
[Loss: 只监督 9 个被 mask 的导联]
```

### 2.2 关键设计决策

| 设计点 | 决策 |
|--------|------|
| Encoder 处理范围 | 处理全部 12 导联（9 个 masked 为 0） |
| Lead 独立性 | 每个 Lead 独立编码，Transformer 只处理时序关系 |
| Transformer 参数 | 所有 12 个 Lead 共享同一个 Transformer |
| VCG 模块 | 无可学习参数，纯几何运算（伪逆） |
| Decoder 角色 | 在几何投影后做可学习的细化 |
| Mask 策略 | 随机选择 3 个导联作为可见输入 |
| Loss 监督范围 | 只监督 9 个被 mask 的导联 |
| 时间维度 | 输出 T 必须与输入 T 严格对齐 |

---

## 3. 各模块详细实现规范

### 3.1 导联角度模块 (`src/data/angle.py`)

#### 3.1.1 功能
- 定义 12 导联的方位角 (φ) 和仰角 (θ)
- 提供不同数据集的导联顺序映射
- 计算导联方向向量 u = [cos(θ)cos(φ), cos(θ)sin(φ), sin(θ)]

#### 3.1.2 导联角度定义
参考 `ner_PTBXL.py`，角度数组格式为 `[θ, φ]`（注意顺序）：

```python
# 角度数组，顺序为: [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]
# 共 12 个导联
LEAD_ANGLES_PTBXL_ORDER = np.array([
    [np.pi / 2, np.pi / 2],           # I
    [np.pi * 5 / 6, np.pi / 2],       # II
    [np.pi / 2, -np.pi / 18],         # V1
    [np.pi / 2, np.pi / 18],          # V2
    [np.pi * (19 / 36), np.pi / 12],  # V3
    [np.pi * (11 / 20), np.pi / 6],   # V4
    [np.pi * (16 / 30), np.pi / 3],   # V5
    [np.pi * (16 / 30), np.pi / 2],   # V6
    [np.pi * (5 / 6), -np.pi / 2],    # III
    [np.pi * (1 / 3), -np.pi / 2],    # aVR
    [np.pi * (1 / 3), np.pi / 2],     # aVL
    [np.pi * 1, np.pi / 2],           # aVF
], dtype=np.float32)
```

#### 3.1.3 数据集重排序
不同数据集的导联顺序不同，需要提供统一的 reorder 函数：

```python
# MIMIC 标准顺序: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
# PTBXL 顺序:     [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]

# MIMIC → PTBXL 的索引映射
MIMIC_TO_PTBXL_ORDER = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]

def reorder_leads(ecg: np.ndarray, source: str, target: str) -> np.ndarray:
    """
    重排导联顺序。
    
    Args:
        ecg: shape [B, 12, T] 或 [12, T]
        source: 源数据集名称 ('mimic', 'ptbxl', etc.)
        target: 目标顺序 ('ptbxl', 'standard', etc.)
    
    Returns:
        重排后的 ECG，shape 与输入相同
    """
    pass
```

#### 3.1.4 方向向量计算

```python
def compute_lead_directions(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    从角度计算导联方向向量。
    
    Args:
        theta: 仰角 [L] 或 [B, L]
        phi: 方位角 [L] 或 [B, L]
    
    Returns:
        方向向量 u，shape [L, 3] 或 [B, L, 3]
        u = [cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
    """
    u_x = torch.cos(theta) * torch.cos(phi)
    u_y = torch.cos(theta) * torch.sin(phi)
    u_z = torch.sin(theta)
    return torch.stack([u_x, u_y, u_z], dim=-1)
```

---

### 3.2 Patch Embedding 模块

#### 3.2.1 功能
将连续的 ECG 信号分割成等距 Patch 并投影到 Token 空间

#### 3.2.2 实现要求

```python
class PatchEmbedding(nn.Module):
    """
    等距 Patch 分割 + Linear 投影。
    
    Pipeline 角色：将原始 ECG 信号转换为 Transformer 可处理的 Token 序列
    
    输入语义：原始 ECG 波形 [B, L, T]
    输出语义：Token 序列 [B, L, N, d]，每个 Token 代表一个时间窗口
    """
    
    def __init__(self, patch_size: int, input_dim: int, embed_dim: int):
        """
        Args:
            patch_size: 每个 Patch 的长度（采样点数）
            input_dim: 输入通道数（对于单导联为 1）
            embed_dim: Token 嵌入维度 d
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, T] - L 个导联，每个长度 T
        
        Returns:
            tokens: [B, L, N, d] - N = T // patch_size
        """
        B, L, T = x.shape
        assert T % self.patch_size == 0, f"T={T} must be divisible by patch_size={self.patch_size}"
        
        N = T // self.patch_size
        # [B, L, T] -> [B, L, N, patch_size]
        x = x.view(B, L, N, self.patch_size)
        # [B, L, N, patch_size] -> [B, L, N, d]
        tokens = self.proj(x)
        return tokens
```

#### 3.2.3 Shape 合约
- 输入：`[B, 12, T]`，T 必须能被 `patch_size` 整除
- 输出：`[B, 12, N, d]`，其中 `N = T // patch_size`

---

### 3.3 Temporal Transformer 模块

#### 3.3.1 功能
对 Token 序列进行时序建模，捕捉心电周期内的时间依赖关系

#### 3.3.2 关键约束
- **Lead 独立**：每个 Lead 的 Token 序列独立处理
- **参数共享**：所有 12 个 Lead 共享同一个 Transformer
- **只做时序**：不做 Cross-Lead Attention

#### 3.3.3 实现要求

```python
class TemporalTransformer(nn.Module):
    """
    时序 Transformer Encoder。
    
    Pipeline 角色：对每个 Lead 的 Token 序列进行时序建模
    
    输入语义：Patch 后的 Token 序列 [B, L, N, d]
    输出语义：增强后的 Token 序列 [B, L, N, d']
    
    关键设计：
    - Lead 独立：不做 Cross-Lead Attention
    - 参数共享：所有 Lead 使用同一套 Transformer 参数
    """
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, N, d] - L 个导联，每个有 N 个 Token
        
        Returns:
            out: [B, L, N, d'] - 增强后的 Token
        """
        B, L, N, d = x.shape
        
        # Lead 独立处理：reshape 成 [B*L, N, d]
        x = x.view(B * L, N, d)
        
        # Transformer 时序建模
        out = self.transformer(x)  # [B*L, N, d']
        
        # 恢复 Lead 维度
        out = out.view(B, L, N, -1)  # [B, L, N, d']
        return out
```

---

### 3.4 Linear Unpatch 模块

#### 3.4.1 功能
将 Token 序列恢复为原始时间长度的信号

#### 3.4.2 关键约束
- **时间对齐**：输出 T 必须等于原始输入 T
- **Lead 独立**：每个 Lead 独立处理

#### 3.4.3 实现要求

```python
class LinearUnpatch(nn.Module):
    """
    将 Token 序列恢复为连续信号。
    
    Pipeline 角色：Encoder 输出 → 原始时间分辨率信号
    
    输入语义：Token 序列 [B, L, N, d']
    输出语义：连续信号 [B, L, T]，T = N * patch_size
    """
    
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, N, d']
        
        Returns:
            out: [B, L, T] where T = N * patch_size
        """
        B, L, N, d = x.shape
        
        # [B, L, N, d'] -> [B, L, N, patch_size]
        x = self.proj(x)
        
        # [B, L, N, patch_size] -> [B, L, T]
        out = x.view(B, L, N * self.patch_size)
        return out
```

---

### 3.5 VCG 伪逆模块

#### 3.5.1 功能
从 3 个可见导联通过几何伪逆恢复 VCG

#### 3.5.2 关键约束
- **无可学习参数**：纯几何运算
- **数值稳定**：添加正则化 `1e-6 * I` 防止奇异

#### 3.5.3 数学公式
每个 ECG 导联是 VCG 的线性投影：
$$s_i(t) = u_i^T \cdot v(t)$$

其中 $u_i = [\cos\theta_i \cos\phi_i, \cos\theta_i \sin\phi_i, \sin\theta_i]$

给定 3 个导联 $S \in \mathbb{R}^{3 \times T}$ 和对应的方向矩阵 $U \in \mathbb{R}^{3 \times 3}$：
$$VCG = U^+ \cdot S$$

其中 $U^+ = (U^T U + \epsilon I)^{-1} U^T$

#### 3.5.4 实现要求

```python
class VCGPseudoInverse(nn.Module):
    """
    VCG 伪逆恢复模块。
    
    Pipeline 角色：从 3 个可见导联恢复 VCG 三维轨迹
    
    输入语义：3 个可见导联的信号 [B, 3, T] + 对应角度 [B, 3, 2]
    输出语义：VCG 三维轨迹 [B, 3, T]
    
    关键设计：
    - 无可学习参数
    - 纯几何运算（伪逆）
    - 添加正则化防止奇异
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, S: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: [B, 3, T] - 3 个可见导联的信号
            theta: [B, 3] - 3 个导联的仰角
            phi: [B, 3] - 3 个导联的方位角
        
        Returns:
            VCG: [B, 3, T] - 恢复的 VCG 三维轨迹
        """
        B, _, T = S.shape
        device = S.device
        
        # 计算方向向量 U: [B, 3, 3]
        u_x = torch.cos(theta) * torch.cos(phi)  # [B, 3]
        u_y = torch.cos(theta) * torch.sin(phi)  # [B, 3]
        u_z = torch.sin(theta)                    # [B, 3]
        U = torch.stack([u_x, u_y, u_z], dim=-1)  # [B, 3, 3]
        
        # 计算伪逆 U_pinv: [B, 3, 3]
        Ut = U.transpose(1, 2)                    # [B, 3, 3]
        UtU = Ut @ U                              # [B, 3, 3]
        UtU_reg = UtU + self.eps * torch.eye(3, device=device).unsqueeze(0)
        UtU_inv = torch.linalg.inv(UtU_reg)       # [B, 3, 3]
        U_pinv = UtU_inv @ Ut                     # [B, 3, 3]
        
        # 恢复 VCG
        VCG = U_pinv @ S                          # [B, 3, T]
        
        return VCG
```

---

### 3.6 几何投影模块

#### 3.6.1 功能
将 VCG 投影到任意导联方向

#### 3.6.2 实现要求

```python
class GeometricLeadProjection(nn.Module):
    """
    VCG → 多导联几何投影。
    
    Pipeline 角色：将 VCG 投影到 12 个导联方向
    
    输入语义：VCG [B, 3, T] + 12 导联角度
    输出语义：12 导联 ECG [B, 12, T]
    
    关键设计：
    - 无可学习参数
    - 纯几何投影：ECG_l = u_l^T @ VCG
    """
    
    def __init__(self, lead_angles: torch.Tensor):
        """
        Args:
            lead_angles: [12, 2] - 12 导联的 [theta, phi] 角度
        """
        super().__init__()
        # 预计算 12 导联的方向向量
        theta = lead_angles[:, 0]  # [12]
        phi = lead_angles[:, 1]    # [12]
        
        u_x = torch.cos(theta) * torch.cos(phi)
        u_y = torch.cos(theta) * torch.sin(phi)
        u_z = torch.sin(theta)
        U = torch.stack([u_x, u_y, u_z], dim=-1)  # [12, 3]
        
        self.register_buffer('U', U)  # 不可学习
    
    def forward(self, VCG: torch.Tensor) -> torch.Tensor:
        """
        Args:
            VCG: [B, 3, T]
        
        Returns:
            ECG: [B, 12, T]
        """
        # U: [12, 3], VCG: [B, 3, T]
        # ECG_l = sum_c(U[l, c] * VCG[:, c, :])
        ECG = torch.einsum('lc,bct->blt', self.U, VCG)  # [B, 12, T]
        return ECG
```

---

### 3.7 Decoder/Refinement 模块

#### 3.7.1 功能
对几何投影后的 ECG 进行可学习的细化

#### 3.7.2 实现要求

```python
class ECGRefinementDecoder(nn.Module):
    """
    ECG 细化解码器。
    
    Pipeline 角色：在几何投影基础上学习残差修正
    
    输入语义：几何投影后的 12 导联 ECG [B, 12, T]
    输出语义：细化后的 12 导联 ECG [B, 12, T]
    
    关键设计：
    - 可学习参数
    - 轻量级（残差补偿，不主导重构）
    - Lead 独立处理
    """
    
    def __init__(self, num_leads: int = 12, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        layers = []
        in_dim = 1
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 12, T] - 几何投影后的 ECG
        
        Returns:
            out: [B, 12, T] - 细化后的 ECG
        """
        B, L, T = x.shape
        
        # Lead 独立处理
        x = x.view(B * L, 1, T)      # [B*L, 1, T]
        residual = self.net(x)        # [B*L, 1, T]
        x = x + residual              # 残差连接
        x = x.view(B, L, T)           # [B, L, T]
        
        return x
```

---

### 3.8 主模型组装 (`src/models/ecggen.py`)

```python
class ECGGenModel(nn.Module):
    """
    ECG 多导联重构模型。
    
    Pipeline 总览：
    1) 输入 12 导联 ECG（3 可见 + 9 masked）
    2) Patch Embedding → Token 序列
    3) Temporal Transformer → 时序建模
    4) Linear Unpatch → 恢复信号
    5) 选择 3 可见导联 → VCG 伪逆 → 恢复 VCG
    6) 几何投影 → 12 导联
    7) Decoder → 细化输出
    
    关键约束：
    - Lead 独立：Encoder 参数共享，不做 Cross-Lead
    - VCG 无参数：纯几何运算
    - 时间对齐：输出 T == 输入 T
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # 配置参数
        self.patch_size = cfg.model.patch_size
        self.embed_dim = cfg.model.embed_dim
        self.num_leads = cfg.model.num_leads  # 12
        
        # 模块
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            input_dim=1,
            embed_dim=self.embed_dim
        )
        
        self.transformer = TemporalTransformer(
            d_model=self.embed_dim,
            nhead=cfg.model.nhead,
            num_layers=cfg.model.num_encoder_layers,
            dim_feedforward=cfg.model.dim_feedforward
        )
        
        self.unpatch = LinearUnpatch(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size
        )
        
        self.vcg_inverse = VCGPseudoInverse()
        
        # 加载 12 导联角度
        lead_angles = torch.tensor(LEAD_ANGLES_PTBXL_ORDER)
        self.lead_projection = GeometricLeadProjection(lead_angles)
        
        self.decoder = ECGRefinementDecoder(
            num_leads=self.num_leads,
            hidden_dim=cfg.model.decoder_hidden,
            num_layers=cfg.model.decoder_layers
        )
        
        # 注册角度 buffer
        self.register_buffer('lead_angles', lead_angles)
    
    def forward(
        self,
        ecg: torch.Tensor,
        visible_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ecg: [B, 12, T] - 原始 12 导联（已 masked）
            visible_indices: [B, 3] - 可见导联的索引
        
        Returns:
            dict with keys:
                - 'recon': [B, 12, T] - 重构的 12 导联
                - 'VCG': [B, 3, T] - 恢复的 VCG
                - 'geom_proj': [B, 12, T] - 几何投影（Decoder 之前）
        """
        B, L, T = ecg.shape
        assert L == 12, f"Expected 12 leads, got {L}"
        
        # 1) Patch Embedding
        tokens = self.patch_embed(ecg)  # [B, 12, N, d]
        
        # 2) Temporal Transformer
        tokens = self.transformer(tokens)  # [B, 12, N, d']
        
        # 3) Linear Unpatch
        encoded = self.unpatch(tokens)  # [B, 12, T]
        
        # 4) 选择 3 可见导联
        # visible_indices: [B, 3]
        batch_idx = torch.arange(B, device=ecg.device).unsqueeze(1).expand(-1, 3)
        visible_ecg = encoded[batch_idx, visible_indices]  # [B, 3, T]
        
        # 5) 获取可见导联的角度
        visible_angles = self.lead_angles[visible_indices]  # [B, 3, 2]
        visible_theta = visible_angles[..., 0]  # [B, 3]
        visible_phi = visible_angles[..., 1]    # [B, 3]
        
        # 6) VCG 伪逆
        VCG = self.vcg_inverse(visible_ecg, visible_theta, visible_phi)  # [B, 3, T]
        
        # 7) 几何投影到 12 导联
        geom_proj = self.lead_projection(VCG)  # [B, 12, T]
        
        # 8) Decoder 细化
        recon = self.decoder(geom_proj)  # [B, 12, T]
        
        return {
            'recon': recon,
            'VCG': VCG,
            'geom_proj': geom_proj
        }
```

---

## 4. 训练逻辑规范

### 4.1 Mask 策略

```python
def random_lead_mask(batch_size: int, num_leads: int = 12, num_visible: int = 3, device='cuda'):
    """
    随机选择可见导联。
    
    Args:
        batch_size: Batch 大小
        num_leads: 总导联数（12）
        num_visible: 可见导联数（3）
        device: 设备
    
    Returns:
        visible_indices: [B, 3] - 可见导联索引
        mask: [B, 12] - 布尔 mask，True 表示被 mask（不可见）
    """
    visible_indices = []
    for _ in range(batch_size):
        indices = torch.randperm(num_leads)[:num_visible]
        visible_indices.append(indices)
    
    visible_indices = torch.stack(visible_indices).to(device)  # [B, 3]
    
    # 生成 mask
    mask = torch.ones(batch_size, num_leads, dtype=torch.bool, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_visible)
    mask[batch_idx, visible_indices] = False  # 可见的置为 False
    
    return visible_indices, mask
```

### 4.2 数据准备

```python
def apply_lead_mask(ecg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将 masked 导联置零。
    
    Args:
        ecg: [B, 12, T] - 原始 ECG
        mask: [B, 12] - 布尔 mask，True 表示被 mask
    
    Returns:
        masked_ecg: [B, 12, T] - masked 后的 ECG
    """
    masked_ecg = ecg.clone()
    # mask: [B, 12] -> [B, 12, 1]
    mask_expanded = mask.unsqueeze(-1)
    masked_ecg = masked_ecg.masked_fill(mask_expanded, 0.0)
    return masked_ecg
```

### 4.3 Loss 计算

```python
def masked_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    只在被 mask 的导联上计算 MSE Loss。
    
    Args:
        recon: [B, 12, T] - 重构的 ECG
        target: [B, 12, T] - 原始 ECG（Ground Truth）
        mask: [B, 12] - 布尔 mask，True 表示被 mask（需要监督）
    
    Returns:
        loss: 标量 - 平均 MSE
    """
    # 只选择被 mask 的导联
    # mask: [B, 12] -> [B, 12, 1]
    mask_expanded = mask.unsqueeze(-1).float()  # [B, 12, 1]
    
    # 计算 MSE
    mse = (recon - target) ** 2  # [B, 12, T]
    
    # 只在 masked 导联上求和
    masked_mse = mse * mask_expanded  # [B, 12, T]
    
    # 平均
    num_masked = mask.sum() * recon.shape[-1]  # 总的 masked 采样点数
    loss = masked_mse.sum() / (num_masked + 1e-8)
    
    return loss
```

### 4.4 训练循环核心

```python
def train_step(model, batch, optimizer, device):
    """
    单步训练。
    """
    ecg = batch['ecg'].to(device)  # [B, 12, T]
    B = ecg.shape[0]
    
    # 1) 生成随机 mask
    visible_indices, mask = random_lead_mask(B, num_visible=3, device=device)
    
    # 2) 应用 mask
    masked_ecg = apply_lead_mask(ecg, mask)
    
    # 3) 前向传播
    outputs = model(masked_ecg, visible_indices)
    recon = outputs['recon']
    
    # 4) 计算 Loss（只在 masked 导联上）
    loss = masked_reconstruction_loss(recon, ecg, mask)
    
    # 5) 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 5. 配置文件规范 (`configs/train/v1.yaml`)

```yaml
run:
  checkpoint_root: /path/to/checkpoints
  log_root: /path/to/logs
  runs_root: /path/to/runs
  save_every: 100
  force_cuda: true
  m: m1
  s: s1
  k: k1

data:
  dataset_type: mimic_raw
  meta_root: /path/to/manifest.json
  num_leads: 12
  time_len: 512
  fs: 100
  lead_order: ptbxl  # 导联顺序标准

model:
  # Patch Embedding
  patch_size: 16
  embed_dim: 128
  
  # Transformer
  nhead: 8
  num_encoder_layers: 4
  dim_feedforward: 512
  
  # Decoder
  decoder_hidden: 64
  decoder_layers: 2
  
  # 其他
  num_leads: 12
  num_visible_leads: 3

train:
  batch_size: 64
  num_workers: 4
  lr: 0.001
  epochs: 100
  max_steps: 1000000
```

---

## 6. Shape 合约总表

| 模块 | 输入 Shape | 输出 Shape |
|------|------------|------------|
| 原始输入 | - | `[B, 12, T]` |
| Masked 输入 | `[B, 12, T]` | `[B, 12, T]`（9 导联为 0） |
| PatchEmbedding | `[B, 12, T]` | `[B, 12, N, d]`，N=T/patch_size |
| Temporal Transformer | `[B*12, N, d]` | `[B*12, N, d']` |
| LinearUnpatch | `[B, 12, N, d']` | `[B, 12, T]` |
| Select Visible | `[B, 12, T]` | `[B, 3, T]` |
| VCGPseudoInverse | `[B, 3, T]` | `[B, 3, T]` (VCG) |
| GeometricLeadProjection | `[B, 3, T]` | `[B, 12, T]` |
| ECGRefinementDecoder | `[B, 12, T]` | `[B, 12, T]` |
| Loss | `[B, 12, T]` × 2 + mask | scalar |

---

## 7. 文件结构

```
ecggen/
├── src/
│   ├── data/
│   │   ├── angle.py          # 导联角度 + reorder
│   │   ├── mimic.py          # MIMIC 数据加载
│   │   └── pipeline.py       # Dataset + DataLoader
│   ├── models/
│   │   ├── ecggen.py         # 主模型组装
│   │   ├── vcg.py            # VCG 伪逆 + 几何投影
│   │   ├── blocks/
│   │   │   ├── patch.py      # Patch Embedding + Unpatch
│   │   │   ├── transformer.py # Temporal Transformer
│   │   │   └── decoder.py    # Refinement Decoder
│   │   └── heads.py          # 分类头（可选）
│   └── utils/
│       ├── trainer.py        # 训练逻辑
│       ├── config.py         # 配置加载
│       └── run_id.py         # Run ID 管理
├── configs/
│   └── train/
│       └── v1.yaml
├── main.py                   # 入口
└── prompts/
    └── scripts/
        └── ecg_reconstruction_v1.md  # 本文档
```

---

## 8. 代码规范

1. **类型标注**：所有函数必须有完整的 type hints
2. **Shape 注释**：每个 Tensor 操作必须注释 shape 变化
3. **Docstring**：每个类/函数必须有 docstring，说明：
   - Pipeline 角色
   - 输入/输出语义
   - 关键设计决策
4. **Assert**：关键 shape 变换处必须有 assert 校验
5. **英文注释**：所有代码注释使用英文

---

## 9. 实现优先级

1. **P0 - 必须首先完成**
   - `src/data/angle.py`：角度定义 + reorder
   - `src/models/vcg.py`：VCG 伪逆 + 几何投影
   - `src/models/blocks/patch.py`：Patch Embedding + Unpatch

2. **P1 - 核心模块**
   - `src/models/blocks/transformer.py`：Temporal Transformer
   - `src/models/blocks/decoder.py`：Refinement Decoder
   - `src/models/ecggen.py`：主模型组装

3. **P2 - 训练支持**
   - `src/utils/trainer.py`：训练循环 + Mask 策略 + Loss
   - `configs/train/v1.yaml`：配置更新

4. **P3 - 验证**
   - 运行训练，确保无报错
   - 检查 Loss 是否正常下降

