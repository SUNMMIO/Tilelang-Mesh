/*!
 * \file tl/transform/inject_dist_sync.cc
 * \brief Materialize Rank communication generation updates after scheduling.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

class InjectDistSignalAdvance : public StmtExprMutator {
public:
  static PrimFunc Run(PrimFunc func) {
    InjectDistSignalAdvance injector;
    Stmt body = injector(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *evaluate_node) final {
    const auto *call = evaluate_node->value.as<CallNode>();
    if (!call) {
      return StmtExprMutator::VisitStmt_(evaluate_node);
    }
    if (call->op.same_as(dist_put_())) {
      ICHECK_EQ(call->args.size(), 6U)
          << "tl.dist_put_ must have its stable peer signature before "
             "InjectDistSync";
      const DistSignalKindInfo &kind =
          RequireDistSignalKindInfo(call->args[3], "signal kind");
      BufferLoad generation =
          RequireLocalState(call->args[5], "generation", kind.state_dtype);
      PrimExpr next =
          Cast(kind.state_dtype, generation + IntImm(kind.state_dtype, 1));
      Stmt advance = BufferStore(generation->buffer, next, generation->indices);
      return SeqStmt::Flatten(
          Array<Stmt>{advance, tvm::ffi::GetRef<Stmt>(evaluate_node)});
    }
    if (call->op.same_as(dist_expect_())) {
      ICHECK_EQ(call->args.size(), 2U);
      BufferLoad expected = RequireLocalState(call->args[0], "expected");
      DataType dtype = expected->dtype;
      PrimExpr delta = Cast(dtype, call->args[1]);
      PrimExpr next = Cast(dtype, expected + delta);
      return BufferStore(expected->buffer, next, expected->indices);
    }
    return StmtExprMutator::VisitStmt_(evaluate_node);
  }

  BufferLoad
  RequireLocalState(const PrimExpr &expr, const char *name,
                    std::optional<DataType> expected_dtype = std::nullopt) {
    const auto *load = expr.as<BufferLoadNode>();
    ICHECK(load) << "T.dist " << name << " state must be a BufferLoad, got "
                 << expr;
    ICHECK(load->dtype.is_uint() &&
           (load->dtype.bits() == 8 || load->dtype.bits() == 32))
        << "T.dist " << name << " state must have uint8 or uint32 dtype";
    if (expected_dtype) {
      ICHECK(load->dtype == expected_dtype.value())
          << "T.dist " << name << " state dtype must match its signal kind: "
          << expected_dtype.value() << ", got " << load->dtype;
    }
    return tvm::ffi::GetRef<BufferLoad>(load);
  }
};

PrimFunc Run(PrimFunc func) {
  if (!IsMultiRank(func)) {
    return func;
  }
  DistOpDetector detector(/*high_level=*/false);
  if (!detector.Detect(func->body)) {
    return func;
  }
  return InjectDistSignalAdvance::Run(std::move(func));
}

} // namespace

tvm::transform::Pass InjectDistSync() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectDistSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectDistSync", InjectDistSync);
}

} // namespace tl
} // namespace tvm
