
import tilelang
import tilelang.language as T

# from mesh_layout import MeshLayout, ReplicationType
from typing import Tuple

class MeshTensorDescriptor:
    """
    MeshTensorDescriptor类,用在编译时描述MeshTensor的布局信息
    
    参数：
    global_shape: 多元元组，元素个数可以是1, 2, 3, 4等
    mesh_layout: 二维元组，默认值为(4, 4)
    axis_partitions: 元组，元素个数可以是一个或两个
    """
    def __init__(self,
                global_shape: Tuple[int, ...],
                mesh_layout: Tuple[int, int],
                axis_partitions: Tuple[int, ...]):
        self.global_shape = global_shape
        self.mesh_layout = mesh_layout
        self.axis_partitions = axis_partitions
        self.mesh_size = mesh_layout[0]*mesh_layout[1]

    def get_local_shape(self) -> Tuple[int, ...]:
        local_shape = list(self.global_shape)
        if len(self.axis_partitions) == 1:
            dim_idx = self.axis_partitions[0]
            local_shape[dim_idx] = self.global_shape[dim_idx] // self.mesh_size
        elif len(self.axis_partitions) == 2:
            dim_idx_0 = self.axis_partitions[0]
            dim_idx_1 = self.axis_partitions[1]
            local_shape[dim_idx_0] = self.global_shape[dim_idx_0] // self.mesh_layout[0]
            local_shape[dim_idx_1] = self.global_shape[dim_idx_1] // self.mesh_layout[1]
        return tuple(local_shape)
# def make_mesh_tensor_descriptor(): 


def get_gpu_info():
    return (4, 4)  # For example, a 4x4 core mesh

def flashattn_fwd(batch, heads, seq_len, dim, block_M, block_N):

    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]# 32b 16h
    dtype = "bfloat16"
    accum_dtype = "float32"

    currunt_mesh_layout = get_gpu_info() # (4, 4) or (4, 3)...
    shape = MeshTensorDescriptor(
        global_shape=(batch, seq_len, heads, dim),
        mesh_layout=currunt_mesh_layout,
        axis_partitions=(0, 2) # partition along batch and heads dimensions
    ).get_local_shape()
    print(f"\n******Local shape on each core: {shape}*****\n")
    batch, seq_len, heads, dim = shape

    
    @T.prim_func
    def flash_attention(
        Q: T.Tensor((batch, seq_len, heads, dim), dtype),
        K: T.Tensor((batch, seq_len, heads, dim), dtype),
        V: T.Tensor((batch, seq_len, heads, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        # Launch a specialized T.Kernel with 3D mapping: (bx, by, bz)
        #   bx: block index in sequence dimension
        #   by: block index in "heads" dimension
        #   bz: block index in "batch" dimension
        # Assume each core is responsible for a block of size (block_M, dim) for Q and (block_N, dim) for K, V
        
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch) as (bx, by, bz):
            # Allocate shared memory for Q, K, V to reduce global memory accesses
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            # Allocate buffers on register
            # acc_s: buffer to hold intermediate attention scores
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            # acc_s_cast: buffer for storing casted/adjusted scores
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            # acc_o: partial accumulation of output
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            # Buffers to track per-row maximum score and related stats
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

 
            # Copy a block of Q from global memory to Q_shared
            T.dma_load(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

            # Initialize accumulators
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((bx + 1) * block_M, block_N)

            # Pipeline the loop to overlap copies/gemm stages
            for k in T.Pipelined(loop_range, num_stages=3):
                # Copy K block into shared memory
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)


                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
                # Perform the Q*K^T multiplication, Here, transpose_B=True indicates that K_shared is transposed,
                # policy=T.GemmWarpPolicy.FullRow means each warp is responsible for computing an entire row
                # of acc_s, and the resulting acc_s is retained in registers.
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Copy V block into shared memory
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                for i, j in T.Parallel(block_M, dim):
                    acc_s[i, j] *= scale

                # Save old scores_max, then reset scores_max
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Compute the maximum value per row on dimension 1 (block_N)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    
                # Compute the factor by which we need to rescale previous partial sums
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])

                # Rescale the partial output accumulation to keep exponents consistent
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                # Exponentiate (scores - max) for the new block
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

                # Make a cast of acc_s to fp16 for the next GEMM
                T.copy(acc_s, acc_s_cast)

                # Multiply the attention acc_s_cast by V and add to partial output (acc_o)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                # Update the "logsum" tracker with the newly accumulated sum
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            # Final step: divide each partial output by logsum (completing the softmax)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            # Write back the final output block from acc_o to the Output buffer
            # T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
            T.dma_store(acc_o,Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
    return flash_attention

flashattn = flashattn_fwd(batch=32, heads=16, seq_len=4096, dim=64, block_M=64, block_N=64)
flashattn.show()
