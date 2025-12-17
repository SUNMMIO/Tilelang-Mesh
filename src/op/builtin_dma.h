/*
 *\file tl/op/builtin_dma.h
 *\brief DMA builtin intrinscs for SUNMMIO GPU
 *\separated from the origional Tilelang's code

 */

#ifndef TVM_TL_OP_BUILTIN_DMA_H_
#define TVM_TL_OP_BUILTIN_DMA_H_


#include "operator.h"
#include <tvm/ir/transform.h>


namespace tvm {

namespace tl {

/*!
 * \brief tvm intrinsics for DMADescriptor creation for tiled load
 *
 * CuTensorMap* create_dma_descriptor(data_type, rank, global_addr,
 * global_shape..., global_stride..., smem_box..., smem_stride..., interleave,
 * swizzle, l2_promotion, oob_fill)
 *
 */
TVM_DLL const Op &create_dma_descriptor();


/*!
 * \brief tvm intrinsics for loading data from global tensor descriptor to
 * shared memory for DMA
 *

 *
 */
TVM_DLL const Op &dma_load();


/*!
 * \brief tvm intrinsics for storing data from shared memory to global tensor
 * descriptor for DMA
 *
  *
 */
TVM_DLL const Op &dma_store();

} // namespace tl
} // namespace tvm

#endif