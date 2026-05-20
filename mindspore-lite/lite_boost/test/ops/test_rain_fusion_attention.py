# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
lite_boost test rain_fusion_attention
"""

import math
import torch
import torch_npu
import lite_boost.ops as lite_ops


class TestRainFusionAttention:
    """
    Test rain fusion attention.
    """

    def __init__(self):
        """
        Test rain fusion attention.
        """
        self.device = torch.device("npu:1")
        torch.npu.set_device(self.device)
        self.batch_size = 1
        self.head = 3
        self.q_seqlen = 4096
        self.kv_seqlen = 4096
        self.headdim = 128
        self.scale = self.headdim**-0.5

        q_shape = (self.batch_size, self.q_seqlen, self.head, self.headdim)
        kv_shape = (self.batch_size, self.kv_seqlen, self.head, self.headdim)
        self.q = torch.randn(q_shape, dtype=torch.float16, device=self.device)
        self.k = torch.randn(kv_shape, dtype=torch.float16, device=self.device)
        self.v = torch.randn(kv_shape, dtype=torch.float16, device=self.device)
        self.q_tnd = self.q.reshape(-1, self.head, self.headdim)
        self.k_tnd = self.k.reshape(-1, self.head, self.headdim)
        self.v_tnd = self.v.reshape(-1, self.head, self.headdim)

        q_blocknum = math.ceil(self.q_seqlen / 128)
        kv_blocknum = math.ceil(self.kv_seqlen / 128)
        self.block_shape = [128, 128]
        self.actual_seq_lengths = [self.q_seqlen for _ in range(self.batch_size)]
        self.actual_seq_lengths_kv = [self.kv_seqlen for _ in range(self.batch_size)]
        self.select_idx, self.select_num_idx = self._generate_sparse_mask(
            q_blocknum, self.head, kv_blocknum, ratio=1.0
        )

    def _generate_sparse_mask(
        self, q_blocknum, head, kv_blocknum, device="npu", ratio=1.0
    ):
        """
        Generate sparse mask.
        """
        select_idx = torch.full(
            (q_blocknum, head, kv_blocknum), -1, dtype=torch.int64, device=device
        )

        select_num_idx = torch.tensor(
            kv_blocknum, dtype=torch.int64, device=device
        ).repeat(q_blocknum, head)

        base_indices = torch.arange(kv_blocknum, dtype=torch.int64, device=device)
        select_idx[...] = base_indices.repeat(q_blocknum, head, 1)

        for q in range(q_blocknum):
            for h in range(head):
                selected_kvs = base_indices[: int(kv_blocknum * ratio)]
                select_idx[q, h, : len(selected_kvs)] = selected_kvs
                select_num_idx[q, h] = len(selected_kvs)

        return select_idx, select_num_idx

    def test_rainfusionattention_vs_fusionattention(self):
        """
        Test rain fusion attention vs fusion attention.
        """
        ra, _ = lite_ops.rain_fusion_attention(
            self.q_tnd,
            self.k_tnd,
            self.v_tnd,
            self.select_idx,
            self.select_num_idx,
            self.block_shape,
            attn_mask=None,
            actual_seq_lengths=self.actual_seq_lengths,
            actual_seq_lengths_kv=self.actual_seq_lengths_kv,
            block_table=None,
            q_input_layout="TND",
            kv_input_layout="TND",
            num_key_value_heads=self.head,
            mask_type=0,
            scale_value=self.scale,
            inner_precise=0,
            block_size=0,
        )
        fascore = torch_npu.npu_fusion_attention(
            self.q,
            self.k,
            self.v,
            input_layout="BSND",
            scale=self.headdim**-0.5,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            head_num=self.head,
        )[0]
        print("rain_fusion_attention shape:", ra.shape)
        print("fusion_attention shape:", fascore.shape)

    def test_sparse_attention_rf_v2(self):
        """
        Test sparse_attention rf_v2 path vs dense fusion attention.
        """
        from lite_boost.ops.sparse_attention import sparse_attention

        latent_shape_q = (1, 64, 64)   # t=1, h=64, w=64 => 4096 tokens
        latent_shape_k = (1, 64, 64)

        out_sparse = sparse_attention(
            self.q, self.k, self.v,
            scale=self.scale,
            head_num=self.head,
            input_layout="BSND",
            inner_precise=0,
            sparse_type="rf_v2",
            txt_len=0,
            block_size=128,
            latent_shape_q=latent_shape_q,
            latent_shape_k=latent_shape_k,
            sparsity=0.0,
        )
        out_dense = torch_npu.npu_fusion_attention(
            self.q, self.k, self.v,
            input_layout="BSND",
            scale=self.scale,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            head_num=self.head,
        )[0]
        print("sparse_attention (rf_v2) shape:", out_sparse.shape)
        print("fusion_attention (dense) shape:", out_dense.shape)
        max_diff = (out_sparse - out_dense).abs().max().item()
        print("max diff:", max_diff)


if __name__ == "__main__":
    test = TestRainFusionAttention()
    test.test_rainfusionattention_vs_fusionattention()
    test.test_sparse_attention_rf_v2()
