/*
 *\file tl/op/builtin_dma.CC
 *\brief DMA builtin intrinscs for SUNMMIO GPU
 *\separated from the origional Tilelang's code

 */

#include "builtin_dma.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/cuda.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {


#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)   

TIR_DEFINE_TL_BUILTIN(create_dma_descriptor)
.set_num_inputs(-1)
.set_attr<TCallEffectKind>("TCallEffectKind",
                            Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(dma_load).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    
TIR_DEFINE_TL_BUILTIN(dma_store).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
