import tilelang
import tilelang.testing
import tilelang.language as T


tilelang.env.disable_cache()


def matmul(
    batch,
    M,
    N,
    K,
    dtype="float16",
):

    @T.prim_func
    def main(
        A: T.Tensor((batch, M, K), dtype),  # type: ignore
        B: T.Tensor((batch, K, N), dtype),  # type: ignore
        C: T.Tensor((batch, M, N), dtype),  # type: ignore
    ):
        T.comm.CoreId((1, 2))
        T.comm.put(A, B, (1, 3))
        T.comm.put(A, B, (2, 3), size=1024)
        T.comm.broadcast(A, (0, 0))
        T.comm.broadcast(A, (1, 2), group=[(1, 2), (1, 3), (2, 2), (2, 3)])
        T.comm.all_gather(A, B)
        T.comm.all_gather(A, B, group=[(0, 0), (0, 1), (1, 0), (1, 1)])
        T.comm.all_reduce("sum", A, B, group=[(0, 0), (0, 1), (1, 0)], axis=0)
        T.comm.barrier()
        T.comm.barrier(group=[(0, 0), (0, 1), (1, 0), (1, 1)])
        T.comm.fence()
        T.comm.current_core()

    return main

def test_frontend():
    func = matmul(64, 1024, 1024, 1024)
    # print(func.script())

    func_TIR = """# from tvm.script import tir as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.match_buffer(A_handle, (64, 1024, 1024), "float16", strides=(1048576, 1024, 1))
    B = T.match_buffer(B_handle, (64, 1024, 1024), "float16", strides=(1048576, 1024, 1))
    C = T.match_buffer(C_handle, (64, 1024, 1024), "float16", strides=(1048576, 1024, 1))
    T.CoreId(6)
    T.comm_put(A[0:64, 0:1024, 0:1024], B[0:64, 0:1024, 0:1024], T.CoreId(7))
    T.comm_put(A[0:64, 0:1024, 0:1024], B[0:64, 0:1024, 0:1024], T.CoreId(11), 1024)
    T.comm_broadcast(A[0:64, 0:1024, 0:1024], T.CoreId(0))
    T.comm_broadcast(A[0:64, 0:1024, 0:1024], T.CoreId(6), T.CoreId(6), T.CoreId(7), T.CoreId(10), T.CoreId(11))
    T.comm_allgather(A[0:64, 0:1024, 0:1024], B[0:64, 0:1024, 0:1024])
    T.comm_allgather(A[0:64, 0:1024, 0:1024], B[0:64, 0:1024, 0:1024], T.CoreId(0), T.CoreId(1), T.CoreId(4), T.CoreId(5))
    T.comm_reduce("sum", A[0:64, 0:1024, 0:1024], B[0:64, 0:1024, 0:1024], 0, T.CoreId(0), T.CoreId(1), T.CoreId(4))
    T.comm_barrier()
    T.comm_barrier(T.CoreId(0), T.CoreId(1), T.CoreId(4), T.CoreId(5))
    T.comm_fence()
    T.comm_current_core()"""
    
    assert func.script() == func_TIR


if __name__ == '__main__':
    tilelang.testing.main()