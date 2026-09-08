"""Receiver expected/generation lowering tests for Rank communication."""

from collections import Counter

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.sunmmio.inter_rank.lowering import lower_to_device_tir


@tilelang.jit(target="sunmmio")
def all_signal_state_kinds_kernel_factory(world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            local_M, local_N = A.local_shape
            src = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_dst = T.alloc_shared((local_M, local_N), T.bfloat16)
            sram_inc = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_INC)
            dram_inc = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_INC)
            sram_value = T.dist.signal(kind=T.dist.SignalKind.SRAM_FLAGREG_VALUE)
            dram_value = T.dist.signal(kind=T.dist.SignalKind.DRAM_FLAGREG_VALUE)
            sram_memory = T.dist.signal(kind=T.dist.SignalKind.SRAM_MEMORY)
            dram_memory = T.dist.signal(kind=T.dist.SignalKind.DRAM_MEMORY)
            peer_rank = (rank_id + 1) % world_size

            T.copy(A, src)
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_inc)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_inc)
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_value)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_value)
            T.dist.put(src, sram_dst, dst_rank=peer_rank, signal=sram_memory)
            T.dist.put(src, B, dst_rank=peer_rank, signal=dram_memory)
            T.dist.wait_signal(sram_inc, dst=sram_dst)
            T.dist.wait_signal(dram_inc, dst=B)
            T.dist.wait_signal(sram_value, dst=sram_dst)
            T.dist.wait_signal(dram_value, dst=B)
            T.dist.wait_signal(sram_memory, dst=sram_dst)
            T.dist.wait_signal(dram_memory, dst=B)

    return main


@tilelang.jit(target="sunmmio")
def multi_sender_sram_signal_kernel_factory(signal_kind, world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((32,), T.bfloat16)
            signal = T.dist.signal(kind=signal_kind)

            T.dist.put(src, dst, dst_rank=0, signal=signal)
            T.dist.wait_signal(signal, dst=dst)

    return main


@tilelang.jit(target="sunmmio")
def multi_sender_dram_signal_kernel_factory(signal_kind, world_size: int = 1):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def main(
        A: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        B: T.MeshTensor((32, 32), placement, T.bfloat16),  # type: ignore
        rank_id: T.dist.RankId,
    ):
        with T.Kernel():
            signal = T.dist.signal(kind=signal_kind)
            T.dist.put(A, B, dst_rank=0, signal=signal)
            T.dist.wait_signal(signal, dst=B)

    return main


@tilelang.jit(target="sunmmio")
def interleaved_wait_all_kernel_factory(world_size: int = 1):
    @T.prim_func
    def main(rank_id: T.dist.RankId):
        with T.Kernel():
            src0 = T.alloc_shared((32,), T.bfloat16)
            src1 = T.alloc_shared((32,), T.bfloat16)
            dst = T.alloc_shared((64,), T.bfloat16)
            signals = T.dist.signals(2)
            peer_rank = (rank_id + 1) % world_size

            T.dist.put(src0, dst[0:32], dst_rank=peer_rank, signal=signals[0])
            T.dist.put(src1, dst[32:64], dst_rank=peer_rank, signal=signals[1])
            T.dist.put(src0, dst[0:32], dst_rank=peer_rank, signal=signals[0])
            T.dist.wait_all(signals, dst=dst)

    return main


def _single_device_func(mod):
    funcs = [func for func in mod.functions.values() if isinstance(func, tvm.tir.PrimFunc)]
    assert len(funcs) == 1
    return funcs[0]


def _collect_leaf_states(func, op_name, kind_position, index_position, state_position):
    states = []

    def visit(node):
        if not isinstance(node, tvm.tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name != op_name:
            return
        state = node.args[state_position]
        assert isinstance(state, tvm.tir.BufferLoad)
        states.append(
            (
                str(node.args[kind_position].value),
                int(node.args[index_position]),
                str(state.dtype),
                state.buffer,
            )
        )

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return states


def _collect_local_state_stores(func):
    stores = Counter()

    def visit(node):
        if not isinstance(node, tvm.tir.BufferStore) or node.buffer.scope() != "local.var":
            return
        stores[node.buffer] += 1
        assert str(node.value.dtype) == str(node.buffer.dtype)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return stores


def test_all_signal_kinds_use_their_declared_state_dtype():
    func = all_signal_state_kinds_kernel_factory.get_tir(world_size=4)
    device_func = _single_device_func(lower_to_device_tir(func).device_mod)

    puts = _collect_leaf_states(device_func, "tl.dist_put_", 3, 4, 5)
    waits = _collect_leaf_states(device_func, "tl.dist_wait_signal_", 0, 1, 2)
    expected_dtypes = {
        "sram_flagreg_inc": "uint8",
        "dram_flagreg_inc": "uint8",
        "sram_flagreg_value": "uint32",
        "dram_flagreg_value": "uint32",
        "sram_memory": "uint32",
        "dram_memory": "uint32",
    }

    assert {kind: dtype for kind, _, dtype, _ in puts} == expected_dtypes
    assert {kind: dtype for kind, _, dtype, _ in waits} == expected_dtypes

    stores = _collect_local_state_stores(device_func)
    for _, _, _, state_buffer in puts + waits:
        assert stores[state_buffer] == 1


@pytest.mark.parametrize(
    "factory, signal_kind",
    [
        (multi_sender_sram_signal_kernel_factory, T.dist.SignalKind.SRAM_FLAGREG_INC),
        (multi_sender_dram_signal_kernel_factory, T.dist.SignalKind.DRAM_FLAGREG_INC),
    ],
)
def test_inc_signal_aggregates_multiple_physical_senders(factory, signal_kind):
    func = factory.get_tir(signal_kind=signal_kind, world_size=4)
    result = lower_to_device_tir(func, capture_passes="tl.LowerDistCommunication")
    script = result.pass_snapshot("tl.LowerDistCommunication").mod.script()

    assert "T.Select(rank_id == 0, 3, 0)" in script
    assert "T.dist_expect_(" in script


@pytest.mark.parametrize(
    "factory, signal_kind",
    [
        (multi_sender_sram_signal_kernel_factory, T.dist.SignalKind.SRAM_FLAGREG_VALUE),
        (multi_sender_sram_signal_kernel_factory, T.dist.SignalKind.SRAM_MEMORY),
        (multi_sender_dram_signal_kernel_factory, T.dist.SignalKind.DRAM_FLAGREG_VALUE),
        (multi_sender_dram_signal_kernel_factory, T.dist.SignalKind.DRAM_MEMORY),
    ],
)
def test_value_and_memory_signals_reject_multiple_physical_senders(factory, signal_kind):
    func = factory.get_tir(signal_kind=signal_kind, world_size=4)
    with pytest.raises(tvm.error.InternalError, match="multiple physical senders"):
        lower_to_device_tir(func)


def test_wait_all_reads_each_signals_current_expected_state():
    func = interleaved_wait_all_kernel_factory.get_tir(world_size=4)
    device_func = _single_device_func(lower_to_device_tir(func).device_mod)
    puts = _collect_leaf_states(device_func, "tl.dist_put_", 3, 4, 5)
    waits = _collect_leaf_states(device_func, "tl.dist_wait_signal_", 0, 1, 2)
    stores = _collect_local_state_stores(device_func)

    generation_by_index = {index: name for _, index, _, name in puts}
    expected_by_index = {index: name for _, index, _, name in waits}
    assert stores[generation_by_index[0]] == 2
    assert stores[expected_by_index[0]] == 2
    assert stores[generation_by_index[1]] == 1
    assert stores[expected_by_index[1]] == 1
    assert [index for _, index, _, _ in waits] == [0, 1]
