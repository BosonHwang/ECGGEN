全项目唯一 shape 规范

# ecggen Shape Contract（唯一真理源）

## 原始输入
- ECG: [B, L, T]

## Tokenization
- Beats:
  - N: beat 数
  - beat_len: 插值后的长度

- Tokens:
  - X: [B, L, N, d]

- Metadata:
  - beat_intervals: [B, N]

## TokenEncoder
- 输入：X [B,L,N,d]
- reshape → [B, L*N, d]
- 输出：H [B, L*N, d_model]

## TTT
- 输入：H [B, L*N, d_model]
- 输出：
  - W_final: [B, D]
  - W_seq（可选）: [B, K, D]

## VCG
- 输入：W [B, D]
- 输出：V [B, 3, T’]

## Projection
- 输入：V [B,3,T’]
- 输出：E_hat’ [B,L,T’]

## Residual
- 输入：base [B,L,T’]
- 输出：residual [B,L,T’]

## 分类
- 输入：W [B,D]
- 输出：logits [B,C]

## 不变量
- 无时间递归
- 无未来预测
- W 是被估计的状态，不是演化的动力学

