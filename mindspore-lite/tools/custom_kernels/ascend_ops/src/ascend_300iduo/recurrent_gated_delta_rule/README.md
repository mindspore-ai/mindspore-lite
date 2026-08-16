# RecurrentGatedDeltaRule for Atlas 300I Duo

This directory provides the Ascend C implementation used when
the built-in `aclnnRecurrentGatedDeltaRule` kernel is unavailable on
Atlas 300I Duo. It uses a product-specific registration name to avoid a name
collision with CANN's built-in operator metadata.

## Interface

The custom operator follows the LiteBoost op-api inputs. The state is a
reference input that is updated in place:

- `query`: `[T, Nk, Dk]`, FP16
- `key`: `[T, Nk, Dk]`, FP16
- `value`: `[T, Nv, Dv]`, FP16
- `beta`: `[T, Nv]`, FP16
- `state`: `[state_slots, Nv, Dv, Dk]`, FP16
- `actual_seq_lengths`: `[B]`, INT32
- `ssm_state_indices`: `[T]`, INT32
- `g`: `[T, Nv]`, optional FP32
- `gk`: `[T, Nv, Dk]`, optional FP32
- `num_accepted_tokens`: `[B]`, optional INT32
- `scale_value`: optional float attribute
- `out`: `[T, Nv, Dv]`, FP16
- updated state reference: `[state_slots, Nv, Dv, Dk]`, FP16

`Nv` must be divisible by `Nk`. Each key/query head is shared by
`Nv / Nk` consecutive value heads. Decode sequences contain at most eight
tokens per batch.

## Implementation

Work is split by `(batch, value_head)` across at most eight AIV task
slots. State tiles are accumulated in FP32 in Unified Buffer and written back
as FP16. The output and recurrent state use separate queue buffers while
respecting the device eight-queue limit.

## Tests

The LiteBoost test
`lite_boost/test/ops/test_recurrent_gated_delta_rule.py` contains
`ascend_300iduo` cases for the Qwen3.5-4B dimensions
`Nk=16, Nv=32, Dk=Dv=128`, including batch size two.
