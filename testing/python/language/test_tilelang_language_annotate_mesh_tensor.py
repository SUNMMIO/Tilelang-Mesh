import tilelang
import tilelang.testing
import tilelang.language as T


tilelang.env.disable_cache()


def matmul(
    batch,
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float",
):
    mt_funcs = T.mesh_tensor_functions(T.get_target_mesh_shape("a4e"))
    annotate_mesh_tensor_info = mt_funcs["annotate_mesh_tensor_info"]
    mesh_tensor_copy = mt_funcs["mesh_tensor_copy"]
    get_tile_shape = mt_funcs["get_tile_shape"]
    mt_info = mt_funcs["init_mesh_tensor_info"]

    @T.prim_func
    def main(
        A: T.Tensor((batch, M, K), dtype),  # type: ignore
        B: T.Tensor((batch, K, N), dtype),  # type: ignore
        C: T.Tensor((batch, M, N), dtype),  # type: ignore
    ):
        annotate_mesh_tensor_info(
            {
                A: mt_info(sharding=(0, 1), block_shape=(1, block_M, block_K)),
                B: mt_info(sharding=(0, 0), block_shape=(1, block_K, block_N)),
                C: mt_info(sharding=(0, 1), block_shape=(1, block_M, block_N)),
            },
        )
        
        tile_batch, tile_M, tile_N = get_tile_shape(C)

        with T.Kernel(tile_N, tile_M, tile_batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bz, by * block_M, ko * block_K], A_shared)
                mesh_tensor_copy(A, A_shared, src_coord=(bz, by, ko))
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[bz, ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bz, by * block_M, bx * block_N])
            mesh_tensor_copy(C_local, C, dst_coord=(bz, by, bx))

    return main


def test_frontend():
    func = matmul(64, 1024, 1024, 1024, 128, 128, 32)
    print(func.script())
    func_TIR = """# from tvm.script import tir as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.handle("float16", "global")
    B = T.handle("float16", "global")
    C = T.handle("float16", "global")
    T.func_attr({"mesh_tensor_info": {A: {"block_shape": [1, 128, 32], "order": "block_wise", "program_id": T.comm_current_core(), "sharding": {"x": 0, "y": 1}}, B: {"block_shape": [1, 32, 128], "order": "block_wise", "program_id": T.comm_current_core(), "sharding": {"x": 0, "y": 0}}, C: {"block_shape": [1, 128, 128], "order": "block_wise", "program_id": T.comm_current_core(), "sharding": {"x": 0, "y": 1}}}})
    A_1 = T.match_buffer(A_handle, (64, 1024, 1024), "float16", data=A, strides=(1048576, 1024, 1))
    B_1 = T.match_buffer(B_handle, (64, 1024, 1024), "float16", data=B, strides=(1048576, 1024, 1))
    C_1 = T.match_buffer(C_handle, (64, 1024, 1024), "float16", data=C, strides=(1048576, 1024, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 1024)
    by = T.launch_thread("blockIdx.y", 256)
    bz = T.launch_thread("blockIdx.z", 16)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A_1[bz, by * 128, 0:993], B_1[bz, 0:1024, bx * 128:bx * 128 + 128], C_1[bz, by * 128, bx * 128])
        T.writes()
        A_shared = T.alloc_buffer((128, 32), "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 128), "float16", scope="shared.dyn")
        C_local = T.alloc_buffer((128, 128), scope="local.fragment")
        T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
        for ko in T.serial(32, annotations={"num_stages": 3}):
            T.copy(T.region(A_1[bz, by * 128, ko * 32], 1, 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32), -1, T.bool(False), 0)
            T.copy(T.region(A_1[bz, by * 128, ko * 32], 1, 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32), -1, T.bool(False), 0)
            for k in T.parallel(32):
                for j in T.parallel(128):
                    B_shared[k, j] = B_1[bz, ko * 32 + k, bx * 128 + j]
            T.gemm_py(T.region(A_shared[0, 0], 1, 128, 32), T.region(B_shared[0, 0], 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, T.uint32(0), 0, 0)
        T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C_1[bz, by * 128, bx * 128], 2, 1, 128, 128), -1, T.bool(False), 0)
        T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C_1[bz, by * 128, bx * 128], 2, 1, 128, 128), -1, T.bool(False), 0)"""
    assert func.script() == func_TIR


if __name__ == "__main__":
    tilelang.testing.main()
    # test_frontend()
