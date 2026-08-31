lite_boost.layers.rope_apply
============================

.. py:function:: lite_boost.layers.rope_apply(x, grid_sizes, freqs)

    对 `x` 应用旋转位置编码（RoPE）。

    将 `x` 的交错复对 :math:`(x[..., 2k], x[..., 2k+1])` 按 `freqs` 的旋转角旋转，cos/sin频率表按每样本的 `grid_sizes` 展开并进程内缓存；本rank切片遵循序列并行（SP）切分，当 ``seq_len % sp_size != 0`` 时末rank切片在表外零填充（填充位置输入为零，输出保持为零）。调用前需初始化 ``torch.distributed``（单进程可调用 ``dist.init_process_group(backend="hccl", world_size=1, rank=0)``）。本接口仅支持A2，不支持300I Duo。

    符号说明：`T` 为 `freqs` 频率表长度，`B` 为batch大小，`F`/`H`/`W` 为每样本的（帧数、高、宽）网格（原始序列长度 ``seq_len = F*H*W``），`D` 为head_dim，`N` 为头数，`s` 为按SP切分后的每rank序列长度（``padded_seq_len / sp_size``，``padded_seq_len`` 为各样本SP切分前统一补齐后的长度）。`s` 与 ``F*H*W`` 相关但不完全相同：单卡且无需补齐时 ``s == F*H*W``；一般情况下 ``s >= ceil(seq_len / sp_size)``（按补齐后的长度均分），末rank切片可能超出短样本的表范围，超出位置补零（该处输入亦为零，输出保持为零）。

    参数：
        - **x** (Tensor) - shape为 :math:`(B, s, N, D)` 的输入张量。`B` 为batch大小，`s` 为按SP切分后的序列长度（``padded_seq_len / sp_size``），`N` 为头数，`D` 为head_dim（偶数，按交错复对旋转）。支持float16、float32和bfloat16。
        - **grid_sizes** (Tensor) - shape为 :math:`(B, 3)` 的整型张量，每样本的 :math:`(F, H, W)` 网格（``seq_len = F*H*W``）。
        - **freqs** (Tensor) - shape为 :math:`(T, D//2)` 的复数张量（极坐标 :math:`e^{i\theta}`），须与 `x` 在同一设备上。

    返回：
        Tensor, 与 `x` 同shape，输出固定为float32。

    异常：
        - **RuntimeError** - `x` 非4维、`D` 非偶数、`freqs` 与 `x` 不在同一设备时抛出。
        - **ValueError** - 调用前未初始化 ``torch.distributed`` 时抛出。
        - **TypeError** - `grid_sizes` 非2维 :math:`[B, 3]` 时抛出。
