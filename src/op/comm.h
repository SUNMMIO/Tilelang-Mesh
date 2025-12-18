/*!
 * \file tl/op/comm.h
 * \brief Communication intrinsics.
 *
 */

#ifndef TVM_TL_OP_BUILTIN_H_
#define TVM_TL_OP_BUILTIN_H_

#include "operator.h"

namespace tvm {
namespace tl {
/*!
 * \brief tvm intrinsics for putting data from one core to another.
 *
 * comm_put(src_buffer, dst_buffer, dst_core, size)
 *
 */
TVM_DLL const Op &comm_put();
/*!
 * \brief tvm intrinsics for broadcasting data from one core to a group of cores.
 *
 * comm_broadcast(buffer, src_core, group)
 *
 */
TVM_DLL const Op &comm_broadcast();
/*!
 * \brief tvm intrinsics for gathering data from all cores.
 *
 * comm_allgather(send_buffer, recv_buffer, group)
 *
 */
TVM_DLL const Op &comm_allgather();
/*!
 * \brief tvm intrinsics for reducing data across cores.
 *
 * comm_reduce(reduce_type, send_buffer, recv_buffer, group)
 *
 */
TVM_DLL const Op &comm_reduce();
/*!
 * \brief tvm intrinsics for synchronizing cores.
 *
 * comm_barrier(group)
 *
 */
TVM_DLL const Op &comm_barrier();
/*!
 * \brief tvm intrinsics for fence operations.
 *
 * comm_fence()
 *
 */
TVM_DLL const Op &comm_fence();
/*!
 * \brief tvm intrinsics for getting core id.
 *
 * CoreId(core_index)
 *
 */
TVM_DLL const Op &CoreId();
/*!
 * \brief tvm intrinsics for getting current core id.
 *
 * comm_current_core()
 *
 */
TVM_DLL const Op &comm_current_core();
} // namespace tl
} // namespace tvm
#endif //  TVM_TL_OP_BUILTIN_H_