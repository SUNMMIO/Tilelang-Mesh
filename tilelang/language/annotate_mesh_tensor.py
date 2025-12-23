from copy import deepcopy
from typing import Callable, Literal

from tvm import tir
import tilelang.language as T

"""Utilities for working with MeshTensor abstractions.

This module provides a small factory function `mesh_tensor_functions` that
generates helper callables for annotating mesh tensor metadata, computing
per-tile shapes and performing copy operations between mesh tiles.
"""


def get_target_mesh_shape(target: str = "auto") -> dict[str, int]:
    """Get the shape of the target mesh as a dictionary with 'x' and 'y' keys.
    Args:
        target: The target mesh type. Supported values are
            'a4e', 'a4e-lite', and 'auto'. If 'auto' is specified,
            the function defaults to 'a4e'.
    Returns:
        A dictionary with integer keys 'x' and 'y' representing
        the 2D mesh size in each dimension.
    Raises:
        ValueError: If an unknown target is specified.
    """
    if target == "auto":
        target = "a4e"

    if target == "a4e":
        return {"x": 4, "y": 4}
    elif target == "a4e-lite":
        return {"x": 2, "y": 4}
    else:
        raise ValueError(f"Unknown target: {target}")


def mesh_tensor_functions(mesh_shape: dict[str, int]) -> dict[str, Callable]:
    """Create mesh-tensor helper functions for a given mesh shape.

    This factory returns a dictionary with three helpers:
    - `annotate_mesh_tensor_info`: attach mesh metadata to buffers
    - `mesh_tensor_copy`: copy data between mesh tiles (supports tile coords)
    - `get_tile_shape`: compute the per-tile shape for a buffer

    Args:
        mesh_shape: A mapping with integer keys 'x' and 'y' representing
            the 2D mesh size in each dimension.

    Returns:
        A dict mapping helper names to callables.

    Raises:
        ValueError: If `mesh_shape` does not contain required keys.
    """

    # Internal storage for mesh-tensor metadata; keyed by buffer.data
    _mesh_tensor_info = {}
    if isinstance(mesh_shape, dict) and "x" in mesh_shape and "y" in mesh_shape:
        _mesh_shape = deepcopy(mesh_shape)
    else:
        raise ValueError("mesh_shape must be a dict with 'x' and 'y' keys.")

    def init_mesh_tensor_info(
        sharding: dict[str, int] | tuple[int, int],
        block_shape: tuple[int, ...],
        program_id: int | tuple[int, int] | None = None,
        order: Literal["block_wise", "row_major"] = "block_wise",
    ):
        """Initialize a mesh tensor metadata dictionary.
        Args:
            sharding: A dict or tuple specifying which tensor dimensions
                are sharded across the mesh in the 'x' and 'y' dimensions.
            block_shape: A tuple specifying the shape of each tile/block.
            program_id: The core id (int) or (row, col) tuple of the
                program owning this mesh tensor. If None, uses current core.
            order: The memory layout order for the mesh tensor.
        Returns:
            A dict containing the mesh tensor metadata.
        Raises:
            ValueError: If `sharding` or `program_id` are malformed.
        """
        if isinstance(sharding, dict):
            if "x" not in sharding or "y" not in sharding:
                raise ValueError("sharding dict must contain 'x' and 'y' keys.")
        elif isinstance(sharding, tuple):
            if len(sharding) != 2:
                raise ValueError("sharding tuple must have exactly two elements.")
            sharding = {"x": sharding[0], "y": sharding[1]}
        else:
            raise ValueError("sharding must be either a dict or a tuple.")
        if program_id is None:
            program_id = T.comm.current_core()
        elif isinstance(program_id, tuple):
            if len(program_id) != 2:
                raise ValueError("program_id tuple must have exactly two elements.")
            program_id = T.comm.CoreId(program_id)
        elif isinstance(program_id, int):
            program_id = T.comm.CoreId(program_id)
        else:
            raise ValueError("program_id must be either an int, a tuple, or None.")

        return {
            "sharding": sharding,
            "block_shape": block_shape,
            "program_id": program_id,
            "order": order,
        }

    def annotate_mesh_tensor_info(mesh_tensor_info: dict[tir.Buffer, dict]) -> Callable:
        """Annotate buffers with mesh tensor metadata.

        Args:
            mesh_tensor_info: Mapping from `tir.Buffer` -> metadata dict.

            The expected input maps buffer objects to info dicts containing
            at least 'block_shape', 'program_id', and 'sharding'.

        Example:
            mesh_tensor_info = {
                buffer_a: {
                    "block_shape": (16, 16),
                    "program_id": 0,
                    "sharding": {"x": 0, "y": 1},
                },
                buffer_b: {
                    "block_shape": (32, 8),
                    "program_id": 1,
                    "sharding": {"x": 1, "y": 0},
                },
            }

        Returns:
            A callable produced by `T.func_attr` that attaches the collected
            metadata to a TVM function under the attribute name
            ``'mesh_tensor_info'``.

        Raises:
            ValueError: If any metadata value is missing required keys.
        """

        nonlocal _mesh_tensor_info
        _mesh_tensor_info = {}
        for buffer, info in mesh_tensor_info.items():
            # Validate metadata structure for each buffer
            if (
                not isinstance(info, dict)
                or "block_shape" not in info
                or "program_id" not in info
                or "sharding" not in info
            ):
                raise ValueError(f"Invalid mesh tensor info: {info}")
            else:
                # store metadata keyed by `buffer.data` so helpers can lookup
                _mesh_tensor_info[buffer.data] = deepcopy(info)

        return T.func_attr({"mesh_tensor_info": _mesh_tensor_info})

    def get_tile_shape(buffer: tir.Buffer) -> tuple[int, ...]:
        """Compute the per-tile shape for a given buffer.

        This uses stored mesh tensor metadata (sharding) to determine which
        tensor dimensions are split across the mesh and divides those
        dimensions by the mesh size, rounding up using `T.ceildiv`.

        Args:
            buffer: A TVM `tir.Buffer` whose shape is to be partitioned.

        Returns:
            A tuple describing the shape of a single tile for `buffer`.

        Raises:
            ValueError: If metadata for `buffer` is not available.
        """

        tensor_shape = buffer.shape
        nonlocal _mesh_tensor_info
        info = _mesh_tensor_info.get(buffer.data, None)
        if info is None:
            raise ValueError(f"MeshTensor information for buffer {buffer} not found.")

        # indices of the tensor dimensions that are sharded on x and y
        sharding_x = info["sharding"]["x"]
        sharding_y = info["sharding"]["y"]

        # start from the full tensor shape and replace the sharded dims
        tile_shape = list(tensor_shape)
        tile_shape[sharding_x] = T.ceildiv(tile_shape[sharding_x], _mesh_shape["x"])
        tile_shape[sharding_y] = T.ceildiv(tile_shape[sharding_y], _mesh_shape["y"])
        return tuple(tile_shape)

    def mesh_tensor_copy(
        src: tir.Buffer,
        dst: tir.Buffer,
        *,
        src_coord: tuple[int, ...] | None = None,
        dst_coord: tuple[int, ...] | None = None,
    ):
        """Copy data between mesh tensor tiles.

        If `src_coord`/`dst_coord` are provided, this computes the per-tile
        buffer slice by multiplying tile coordinates with the stored
        `block_shape` and indexing the buffer accordingly.

        Args:
            src: Source buffer (may be a full tensor or a sliced view).
            dst: Destination buffer.
            src_coord: Optional tile coordinates for the source buffer.
            dst_coord: Optional tile coordinates for the destination buffer.

        Returns:
            The result of `T.copy(src, dst)` which describes the copy
            operation in the TileLang / TVM IR.

        Raises:
            ValueError: If required mesh metadata for either buffer is missing.
        """

        nonlocal _mesh_tensor_info
        # If a source coordinate is specified, compute the local tile slice
        if src_coord is not None:
            try:
                info = _mesh_tensor_info[src.data]
                block_shape = info["block_shape"]
                src = src[tuple(i * b for i, b in zip(src_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f"MeshTensor information for buffer {src} not found."
                ) from e

        # If a destination coordinate is specified, compute its tile slice
        if dst_coord is not None:
            try:
                info = _mesh_tensor_info[dst.data]
                block_shape = info["block_shape"]
                dst = dst[tuple(i * b for i, b in zip(dst_coord, block_shape))]
            except KeyError as e:
                raise ValueError(
                    f"MeshTensor information for buffer {dst} not found."
                ) from e

        # Delegate to TileLang's copy primitive
        return T.copy(src, dst)

    return {
        "annotate_mesh_tensor_info": annotate_mesh_tensor_info,
        "mesh_tensor_copy": mesh_tensor_copy,
        "get_tile_shape": get_tile_shape,
        "init_mesh_tensor_info": init_mesh_tensor_info,
    }
