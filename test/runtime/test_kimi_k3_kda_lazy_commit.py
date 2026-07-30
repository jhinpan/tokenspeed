"""Lazy KDA speculative commit through the real MambaAttnBackend.

The kernel-level equivalences live in
``tokenspeed-kernel/test/ops/attention/test_kda_fused_replay_verify.py``;
what these tests exercise is the runtime plumbing around them: recording a
pending window after acceptance, composing it into the next verify round
(including a re-packed batch), and flushing it whenever a request leaves the
verify stream.

Ground truth for every scenario is the same backend flow with the pending
flushed eagerly after every round -- lazy and eager must land the same
committed pages (to the fused/standalone kernels' ~1 ulp fp32 FMA daylight).
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import flat_metadata_for as _metadata_for
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from types import SimpleNamespace  # noqa: E402  (after torch guard)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
)

_LOWER_BOUND = -5.0
H, D, D_FA = 4, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 2  # draft tokens per request
DEV = "cuda"


def _backend(pool):
    config = SimpleNamespace(
        device=DEV,
        num_attention_heads=H,
        num_kv_heads=H,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=D,
        is_draft=False,
        speculative_num_draft_tokens=T,
    )
    backend = MambaAttnBackend(config, is_kda=True)
    backend.set_kv_pool(pool)
    if not backend._kda_replay_active():
        pytest.skip("KDA replay kernels unavailable on this platform")
    return backend


class _Harness:
    """Drives verify rounds + accepts through one backend over a real pool."""

    def __init__(self, seed=0, usable_pages=16):
        torch.manual_seed(seed)
        self.pool = _make_kimi_pool(DEV, usable_pages=usable_pages)
        self.contract = self.pool.runtime_contract
        self.backend = _backend(self.pool)
        # Drive EVERY KDA layer, as a real verify forward would.
        self.layer_ids = list(self.backend._flat_mamba_layer_ids())
        self.params = {
            layer_id: dict(
                conv_weights=torch.randn(CONV_DIM, 4, device=DEV, dtype=torch.bfloat16)
                * 0.1,
                f_b_weight=torch.randn(KEY_DIM, D_FA, device=DEV, dtype=torch.bfloat16)
                * 0.05,
                A_log=torch.randn(H, device=DEV, dtype=torch.float32) * 0.1,
                dt_bias=torch.randn(KEY_DIM, device=DEV, dtype=torch.float32) * 0.1,
            )
            for layer_id in self.layer_ids
        }

    def window(self, bs, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float32).to(
                DEV, torch.bfloat16
            )

        return dict(
            mixed_qkv=rnd(bs * T, CONV_DIM),
            f_a_out=rnd(bs * T, D_FA),
            beta_raw=rnd(bs * T, H),
        )

    def verify_round(self, rpis, pages, seq_lens, window):
        """One target-verify forward over all three KDA layers."""
        bs = len(rpis)
        tables = {
            gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
            for gid in _STATE_GROUPS
        }
        metadata, op = _metadata_for(self.contract, tables, DEV)
        op.request_pool_indices = list(rpis)
        self.backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            flat_cache_metadata=metadata,
            flat_cache_forward_op=op,
        )
        outs = {}
        for layer_id in self.layer_ids:
            p = self.params[layer_id]
            outs[layer_id] = self.backend.forward_decode(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=self.pool,
                bs=bs,
                mixed_qkv=window["mixed_qkv"].clone(),
                f_a_out=window["f_a_out"],
                beta_raw=window["beta_raw"],
                g_raw=None,
                conv_weights=p["conv_weights"],
                bias=None,
                activation="silu",
                key_dim=KEY_DIM,
                value_dim=KEY_DIM,
                attention_tp_size=1,
                head_k_dim=D,
                head_v_dim=D,
                A_log=p["A_log"],
                dt_bias=p["dt_bias"],
                f_b_weight=p["f_b_weight"],
                lower_bound=_LOWER_BOUND,
                layer_id=layer_id,
                seq_len=bs * T,
                a=None,
                b=None,
            )
        return outs

    def accept(self, accepted):
        self.backend.flat_commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV)
        )

    def flush(self):
        self.backend.flush_kda_pending_commits()

    def state_of(self, layer_id, page):
        conv = self.pool.get_component(layer_id, "conv_state")[page].clone()
        ssm = self.pool.get_component(layer_id, "recurrent_state")[page].clone()
        return conv, ssm

    def pending(self):
        return getattr(self.backend, "_kda_pending", None)


def _pages_for(h, rpis):
    """One distinct page per (request, group), stable across rounds."""
    return {
        gid: [2 + g * len(rpis) + i for i, _ in enumerate(rpis)]
        for g, gid in enumerate(_STATE_GROUPS)
    }


def _run_rounds(h, rounds, accepts, rpis, eager):
    """Drive verify+accept rounds; eager mode flushes after every accept."""
    pages = _pages_for(h, rpis)
    seq_lens = [4 + T] * len(rpis)
    outs = []
    for k, (window_seed, accepted) in enumerate(zip(rounds, accepts)):
        w = h.window(len(rpis), window_seed)
        outs.append(h.verify_round(rpis, pages, seq_lens, w))
        h.accept(accepted)
        if eager:
            h.flush()
        seq_lens = [s + a for s, a in zip(seq_lens, accepted)]
    h.flush()  # commit the last window either way
    return outs, pages


def _assert_pools_match(h_lazy, h_eager, pages, rpis):
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i, _ in enumerate(rpis):
            page = pages[gid][i]
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)


def test_lazy_rounds_match_eager_flush_every_round():
    """Three fused rounds == the same rounds with an eager flush after each.

    A flush-counting probe pins down that the steady-state rounds really
    commit through the fused kernel: if composing fell back to flushing,
    lazy and eager would trivially agree and the test would prove nothing.
    """
    rpis = [0, 1, 2]
    rounds, accepts = [11, 12, 13], [[1, 2, 1], [2, 1, 2], [1, 1, 2]]
    h_lazy = _Harness(seed=5)
    flushes = []
    inner = h_lazy.backend._flush_kda_pending

    def _counting_flush(only_rpis=None):
        flushes.append(only_rpis)
        inner(only_rpis)

    h_lazy.backend._flush_kda_pending = _counting_flush
    outs_lazy, pages = _run_rounds(h_lazy, rounds, accepts, rpis, eager=False)
    # Rounds 2 and 3 must have fused their pending; only the final explicit
    # flush (of round 3's window) may run the standalone kernels.
    assert flushes == [None], flushes
    h_eager = _Harness(seed=5)
    outs_eager, _ = _run_rounds(h_eager, rounds, accepts, rpis, eager=True)

    # Round k's outputs come from identical committed inputs in both modes.
    for lo, eo in zip(outs_lazy, outs_eager):
        for layer_id in h_lazy.layer_ids:
            torch.testing.assert_close(
                lo[layer_id].float(), eo[layer_id].float(), atol=1e-3, rtol=1e-2
            )
    _assert_pools_match(h_lazy, h_eager, pages, rpis)
    assert h_lazy.pending() is None


def test_departed_request_is_flushed_and_survivors_fuse():
    """Round 2 drops the middle request and re-packs the batch.

    The departed request's pending must be flushed at the next arm; the
    survivors replay from payload captured at their OLD slots (row-base
    indirection), so their states must still match the eager run.
    """
    h_lazy = _Harness(seed=7)
    h_eager = _Harness(seed=7)
    all_rpis = [0, 1, 2]
    pages3 = _pages_for(h_lazy, all_rpis)
    w1 = {}
    for h in (h_lazy, h_eager):
        w1[h] = h.window(3, 21)
        h.verify_round(all_rpis, pages3, [6, 6, 6], w1[h])
        h.accept([2, 1, 2])
    h_eager.flush()

    # Request 1 leaves; 0 and 2 re-pack into slots 0 and 1.
    survivors = [0, 2]
    pages2 = {gid: [pages3[gid][0], pages3[gid][2]] for gid in _STATE_GROUPS}
    for h in (h_lazy, h_eager):
        w2 = h.window(2, 22)
        h.verify_round(survivors, pages2, [8, 8], w2)
        h.accept([1, 2])
        h.flush()

    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i in range(3):  # including the departed request's final state
            page = pages3[gid][i]
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)


def test_non_verify_forward_flushes_the_pending():
    """A pending commit must not survive into a non-verify KDA forward."""
    h = _Harness(seed=9)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    w = h.window(2, 31)
    h.verify_round(rpis, pages, [5, 5], w)
    h.accept([1, 2])
    assert h.pending() is not None

    # Metadata prep for a plain decode (non-verify) must flush.
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = rpis
    h.backend.spec_num_tokens = 1  # plain-decode shape for this metadata call
    try:
        h.backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor([6, 7], dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            flat_cache_metadata=metadata,
            flat_cache_forward_op=op,
        )
    finally:
        h.backend.spec_num_tokens = T
    assert h.pending() is None


def test_all_rejected_window_still_commits():
    """accepted = 0 clamps to one committed token (the target's own sample);
    the lazy path must carry that clamp exactly like the eager path."""
    rpis = [0, 1]
    rounds, accepts = [41, 42], [[0, 0], [2, 1]]
    h_lazy = _Harness(seed=11)
    _, pages = _run_rounds(h_lazy, rounds, accepts, rpis, eager=False)
    h_eager = _Harness(seed=11)
    _run_rounds(h_eager, rounds, accepts, rpis, eager=True)
    _assert_pools_match(h_lazy, h_eager, pages, rpis)
