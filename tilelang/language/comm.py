"""Communication intrinsics wrappers for TileLang.

This module provides small helper functions that prepare arguments and
emit TIR intrinsics for inter-core communication on a target mesh. 
"""
from typing import Tuple, Iterable

from tvm import tir
import tilelang.language as T
from tilelang.utils.language import to_buffer_region


def CoreId(core_id: int | tuple[int, int]):
    """Create a TIR handle representing a core identifier.

    Parameters
    ----------
    core_id : int or tuple[int, int]
        Either a linear core id (int) or a (row, col) tuple specifying the
        core's coordinates on the target mesh. When a tuple is provided it
        is converted to a linear id using the mesh shape.

    Returns
    -------
    tir.Call
        A TIR intrinsic handle produced by `tir.call_intrin(..., tl.CoreId, ...)`.

    Raises
    ------
    AssertionError
        If any coordinate is outside the mesh bounds determined by
        `T.get_target_mesh_shape("auto")`.
    ValueError
        If `core_id` is neither an int nor a two-element tuple.

    Examples
    --------
    >>> CoreId((0, 1))
    >>> CoreId(3)
    """
    mesh_shape = T.get_target_mesh_shape("auto")
    if isinstance(core_id, tuple):
        row, col = core_id
        assert (
            0 <= row < mesh_shape["x"]
        ), f"Row {row} out of bounds for mesh shape {mesh_shape}"
        assert (
            0 <= col < mesh_shape["y"]
        ), f"Col {col} out of bounds for mesh shape {mesh_shape}"
        # Convert 2D coordinates into a linear core id.
        core_id_value = row * mesh_shape["x"] + col
    elif isinstance(core_id, int):
        core_id_value = core_id
        assert (
            0 <= core_id_value < mesh_shape["x"] * mesh_shape["y"]
        ), f"Core ID {core_id_value} out of bounds for mesh shape {mesh_shape}"
    else:
        raise ValueError("core_id must be either a tuple[int, int] or an int.")
    return tir.call_intrin("handle", tir.op.Op.get("tl.CoreId"), core_id_value)


def put(
    src_buffer: T.Buffer,
    dst_buffer: T.Buffer,
    dst_core: tuple[int, int],
    size: int | None = None,
):
    """Emit a remote `put` (copy) intrinsic from `src_buffer` to `dst_buffer`.

    Parameters
    ----------
    src_buffer : T.Buffer
        Source buffer region to send.
    dst_buffer : T.Buffer
        Destination buffer region on the remote core.
    dst_core : tuple[int, int]
        Target core coordinates (row, col) to which data will be sent.
    size : int | None = None
        Optional size (in elements) to limit the copy. If omitted the full 
        buffer region is used.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_put`.

    Examples
    --------
    >>> put(A, B, (1, 3))
    >>> put(A, B, (2, 3), size=1024)
    """
    src_buffer_region = to_buffer_region(src_buffer)
    dst_buffer_region = to_buffer_region(dst_buffer)
    dst_core_id = CoreId(dst_core)
    if size is None:
        args = (src_buffer_region, dst_buffer_region, dst_core_id)
    else:
        args = (src_buffer_region, dst_buffer_region, dst_core_id, size)
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_put"), *args)


def broadcast(
    buffer: T.Buffer,
    src_core: tuple[int, int],
    group: Iterable[tuple[int, int]] | None = None,
):
    """Broadcast a buffer from `src_core` to a set of cores.

    Parameters
    ----------
    buffer : T.Buffer
        Buffer region to broadcast.
    src_core : tuple[int, int]
        Source core coordinates that will broadcast the buffer.
    group : iterable of tuple[int, int] | None
        Optional iterable of core coordinates specifying the recipients.
        If omitted, the runtime's default participant set is used.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_broadcast`.

    Examples
    --------
    >>> broadcast(A, (0, 0))
    >>> broadcast(A, (1, 2), group=[(1, 2), (1, 3)])
    """
    src_core_id = CoreId(src_core)
    buffer_region = to_buffer_region(buffer)
    if group is None:
        args = (buffer_region, src_core_id)
    else:
        group = [CoreId(core_id) for core_id in group]
        args = (buffer_region, src_core_id, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_broadcast"), *args)


def all_gather(
    send_buffer: T.Buffer,
    recv_buffer: T.Buffer,
    group: Iterable[tuple[int, int]] | None = None,
):
    """Perform an all-gather: collect contributions into `recv_buffer`.

    Parameters
    ----------
    send_buffer : T.Buffer
        Local buffer containing this core's contribution.
    recv_buffer : T.Buffer
        Buffer to receive the gathered result.
    group : iterable of tuple[int, int] | None
        Optional participant set for the all-gather.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_allgather`.

    Examples
    --------
    >>> all_gather(A, B)
    >>> all_gather(A, B, group=[(0,0),(0,1)])
    """
    send_buffer_region = to_buffer_region(send_buffer)
    recv_buffer_region = to_buffer_region(recv_buffer)
    if group is None:
        args = (send_buffer_region, recv_buffer_region)
    else:
        group = [CoreId(core_id) for core_id in group]
        args = (send_buffer_region, recv_buffer_region, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_allgather"), *args)


def all_reduce(
    op: str,
    src_buffer: T.Buffer,
    dst_buffer: T.Buffer,
    group: Iterable[tuple[int, int]] | None = None,
    axis: int | None = None,
):
    """Reduce values across cores using the specified operation.

    Parameters
    ----------
    op : str
        Reduction operation name (for example, 'sum', 'max').
    src_buffer : T.Buffer
        Source buffer containing local values to reduce.
    dst_buffer : T.Buffer
        Destination buffer to hold the reduced result.
    group : iterable of tuple[int, int] | None
        Optional participant set for the reduction.
    axis : int | None
        Optional axis parameter forwarded to the intrinsic if supported.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_reduce`.

    Examples
    --------
    >>> all_reduce('sum', A, B)
    >>> all_reduce('sum', A, B, group=[(0,0),(0,1)], axis=0)
    """
    src_buffer_region = to_buffer_region(src_buffer)
    dst_buffer_region = to_buffer_region(dst_buffer)
    if group is None and axis is None:
        args = (op, src_buffer_region, dst_buffer_region)
    elif group is not None and axis is None:
        group = [CoreId(core_id) for core_id in group]
        args = (op, src_buffer_region, dst_buffer_region, *group)
    elif group is None and axis is not None:
        args = (op, src_buffer_region, dst_buffer_region, axis)
    else:
        group = [CoreId(core_id) for core_id in group]
        args = (op, src_buffer_region, dst_buffer_region, axis, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_reduce"), *args)


def barrier(group: Iterable[tuple[int, int]] | None = None):
    """Insert a synchronization barrier among a group of cores.

    Parameters
    ----------
    group : iterable of tuple[int, int] | None
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_barrier`.

    Examples
    --------
    >>> barrier()
    >>> barrier(group=[(0,0),(0,1)])
    """
    if group is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"))
    else:
        group = [CoreId(core_id) for core_id in group]
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"), *group)


def fence():
    """Emit a memory/communication fence intrinsic.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_fence`.

    Examples
    --------
    >>> fence()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_fence"))


def current_core():
    """Get the current core's identifier.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_current_core`.

    Examples
    --------
    >>> current_core()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_current_core"))
