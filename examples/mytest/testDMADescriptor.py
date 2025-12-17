"""
Docstring for examples.mytest.testDMADescriptor
just to test create_dma_descriptor() intrin
to run this script successfully, you should 
1. remove tilelang/language/dma.py and line "from .dma import dma_load, dma_store  # noqa: F401" in tilelang/__init__.py
2. add `def create_dma_descriptor(*args):
            return tir.call_intrin("handle", tir.op.Op.get("tl.create_dma_descriptor"), *args)`
    in tilelang/builtin.py
"""


from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir




def test_multi_version_buffer(M, N, K, dtype, block_M, block_N, block_K):

    @T.prim_func
    def before(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype)):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        v = T.launch_thread("threadIdx.x", 128)
        with T.block(""):
            T.reads(A[by * 64, 0:481], B[0:481, bx * 64])
            T.writes()
            A_shared = T.alloc_buffer((1, 8, 256), "float16", scope="shared.dyn")
            B_shared = T.alloc_buffer((1, 4, 512), "float16", scope="shared.dyn")
            C_local = T.alloc_buffer((32,), scope="local")
            for i in T.unroll(16, annotations={"pragma_unroll_explicit": T.bool(False)}):
                for vec in T.vectorized(2):
                    C_local[i * 2 + vec] = T.float32(0)
            for k in T.serial(16, annotations={"num_stages": T.int32(3)}):
                if v == 0:
                    T.dma_load(
                        T.create_dma_descriptor(6, 2, A.data, 512, 512, 2, 1024, 32, 64, 1, 1, 0, 2,
                                                2, 0), 0,
                        T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 2048, 2),
                        k * 32, by * 64)
                if v == 0:
                    T.dma_load(
                        T.create_dma_descriptor(6, 2, B.data, 512, 512, 2, 1024, 64, 32, 1, 1, 0, 3,
                                                2, 0), 0,
                        T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 2048, 2),
                        bx * 64, k * 32)
                T.call_extern(
                    "handle", "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                    T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 2048, 1),
                    T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 32, 3))
    return before



M = 512
N = 512
K = 512
dtype = "float16"
block_M = 64
block_N = 64
block_K = 32
test_kernel = test_multi_version_buffer(M, N, K, dtype, block_M, block_N, block_K)
test_kernel.show()
