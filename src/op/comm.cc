/*!
 * \file tl/op/comm.cc
 * \brief Communication intrinsics.
 *
 */

#include "comm.h"
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {
#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)
    TIR_DEFINE_TL_BUILTIN(comm_put).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_broadcast).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_allgather).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_reduce).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_barrier).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_fence).set_num_inputs(0).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque)); 
    TIR_DEFINE_TL_BUILTIN(CoreId).set_num_inputs(1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_current_core).set_num_inputs(0).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
} // namespace tl
} // namespace tvm