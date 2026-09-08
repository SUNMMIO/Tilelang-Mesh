/*!
 * \file tl/transform/plan_dist_signals.cc
 * \brief Plan and validate Rank-level communication signal resources.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "../op/utils.h"
#include "dist_transform_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;
using namespace dist_transform;

namespace {

void ValidateSignalArgs(const CallNode *call, size_t kind_arg,
                        size_t index_arg) {
  const DistSignalKindInfo &kind =
      RequireDistSignalKindInfo(call->args[kind_arg], "signal kind");
  int64_t index = RequireIntImm(call->args[index_arg], "signal index");
  ICHECK_GE(index, 0);
  if (kind.capacity >= 0) {
    ICHECK_LT(index, kind.capacity)
        << kind.name << " signal index must be in [0, " << kind.capacity
        << "), got " << index;
  }
}

void ValidateStaticRegion(const BufferRegion &region, const char *op_name,
                          const char *operand_name, bool allow_dram) {
  bool valid_scope = region->buffer.scope() == kSunmmioScopeRSRAM ||
                     (allow_dram && IsDramScope(region->buffer.scope()));
  ICHECK(valid_scope) << op_name << " " << operand_name
                      << " must use shared.rsram"
                      << (allow_dram ? " or global/DRAM" : "") << ", got "
                      << region->buffer.scope();
  for (const Range &range : region->region) {
    ICHECK(range->extent.as<IntImmNode>())
        << op_name << " " << operand_name
        << " region must have static extents, got " << region;
  }
}

void ValidateStaticTransfer(const BufferRegion &src, const BufferRegion &dst,
                            const char *op_name) {
  ValidateStaticRegion(src, op_name, "source", /*allow_dram=*/true);
  ValidateStaticRegion(dst, op_name, "destination", /*allow_dram=*/true);
}

int64_t StaticElementCount(const BufferRegion &region) {
  int64_t count = 1;
  for (const Range &range : region->region) {
    count *= range->extent.as<IntImmNode>()->value;
  }
  return count;
}

const char *DestinationScopeName(DistSignalScope scope) {
  switch (scope) {
  case DistSignalScope::kSram:
    return "shared.rsram";
  case DistSignalScope::kDram:
    return "global/DRAM";
  }
  return "unknown";
}

DistSignalScope ClassifyDestinationScope(const BufferRegion &region,
                                         const char *op_name) {
  const ffi::String &scope = region->buffer.scope();
  if (scope == kSunmmioScopeRSRAM) {
    return DistSignalScope::kSram;
  }
  if (IsDramScope(scope)) {
    return DistSignalScope::kDram;
  }
  ICHECK(false) << op_name
                << " destination must use shared.rsram or global/DRAM, got "
                << scope;
  return DistSignalScope::kSram;
}

struct DistSignalRecord {
  using Endpoint = std::pair<int64_t, int64_t>;

  Var var;
  std::string requested_kind;
  int64_t logical_id;
  std::optional<DistSignalScope> destination_scope;
  std::map<Endpoint, std::map<Endpoint, int64_t>> puts_by_receiver_sender;
  std::map<Endpoint, std::set<Endpoint>> receivers_by_sender;
  bool used{false};
  const DistSignalKindInfo *resolved_kind{nullptr};
  int64_t resolved_index{-1};

  bool HasMultiplePhysicalSenders() const {
    for (const auto &[_, senders] : puts_by_receiver_sender) {
      if (senders.size() > 1) {
        return true;
      }
    }
    return false;
  }

  bool CanUseValueForRepeatedPuts() const {
    if (puts_by_receiver_sender.empty() || HasMultiplePhysicalSenders()) {
      return false;
    }
    for (const auto &[_, receivers] : receivers_by_sender) {
      // A physical sender owns one generation counter per signal. Reusing it
      // for different receivers would create generation gaps at each waiter.
      if (receivers.size() != 1) {
        return false;
      }
    }
    for (const auto &[_, senders] : puts_by_receiver_sender) {
      ICHECK_EQ(senders.size(), 1U);
      if (senders.begin()->second < 2) {
        return false;
      }
    }
    return true;
  }
};

class DistSignalUseCollector : public StmtExprVisitor {
public:
  DistSignalUseCollector(Target target, int64_t world_size, Var rank_id)
      : target_(std::move(target)), world_size_(world_size),
        rank_id_(std::move(rank_id)) {
    auto mesh = GetSunmmioMeshConfig(target_);
    mesh_nrows_ = mesh.nrow;
    mesh_ncols_ = mesh.ncol;
  }

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  std::vector<DistSignalRecord> &Records() { return records_; }

private:
  using Endpoint = DistSignalRecord::Endpoint;

  PrimExpr I32(int64_t value) const { return IntImm(DataType::Int(32), value); }

  PrimExpr CurrentCol(const Var &current_core) {
    return analyzer_.Simplify(floormod(current_core, I32(mesh_ncols_)));
  }

  Map<Var, PrimExpr> MakeSourceSubstitution(const Var &current_core,
                                            int64_t src_row, int64_t src_rank) {
    PrimExpr source_core =
        I32(src_row * mesh_ncols_) + CurrentCol(current_core);
    return Map<Var, PrimExpr>{{current_core, source_core},
                              {rank_id_, I32(src_rank)}};
  }

  std::optional<int64_t> ResolveInt(const PrimExpr &expr,
                                    const Map<Var, PrimExpr> &substitution) {
    PrimExpr resolved = analyzer_.Simplify(Substitute(expr, substitution));
    if (const auto *imm = resolved.as<IntImmNode>()) {
      return imm->value;
    }
    return std::nullopt;
  }

  bool SourceMayExecute(const Map<Var, PrimExpr> &substitution) {
    PrimExpr predicate =
        analyzer_.Simplify(Substitute(source_predicate_, substitution));
    return !analyzer_.CanProve(Not(predicate));
  }

  void RecordPhysicalPut(DistSignalRecord &record, int64_t src_rank,
                         int64_t physical_src_row, int64_t dst_rank,
                         int64_t dst_row) {
    if (dst_rank == src_rank) {
      return;
    }
    Endpoint receiver{dst_rank, dst_row};
    Endpoint sender{src_rank, physical_src_row};
    int64_t contribution = repeated_loop_depth_ > 0 ? 2 : 1;
    int64_t &count = record.puts_by_receiver_sender[receiver][sender];
    count = std::min<int64_t>(2, count + contribution);
    record.receivers_by_sender[sender].insert(receiver);
  }

  void RecordRoute(DistSignalRecord &record, const PrimExpr &dst_rank_expr,
                   const PrimExpr &dst_row_expr, const Var &current_core,
                   int64_t origin_src_row, int64_t src_rank) {
    Map<Var, PrimExpr> substitution =
        MakeSourceSubstitution(current_core, origin_src_row, src_rank);
    if (!SourceMayExecute(substitution)) {
      return;
    }
    std::optional<int64_t> dst_rank = ResolveInt(dst_rank_expr, substitution);
    std::optional<int64_t> dst_row = ResolveInt(dst_row_expr, substitution);
    if (!dst_rank || !dst_row || dst_rank.value() < 0 ||
        dst_rank.value() >= world_size_ || dst_row.value() < 0 ||
        dst_row.value() >= mesh_nrows_) {
      return;
    }
    // Cross-row routes first forward within the sender Rank, so the egress row
    // is the remote destination row for every physical peer put.
    RecordPhysicalPut(record, src_rank, dst_row.value(), dst_rank.value(),
                      dst_row.value());
  }

  void RecordLogicalPut(DistSignalRecord &record, const CallNode *call) {
    const auto *current_core_node = call->args[5].as<VarNode>();
    if (!current_core_node) {
      return;
    }
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    for (int64_t src_rank = 0; src_rank < world_size_; ++src_rank) {
      for (int64_t src_row = 0; src_row < mesh_nrows_; ++src_row) {
        RecordRoute(record, call->args[2], call->args[3], current_core, src_row,
                    src_rank);
      }
    }
  }

  void RecordRoutedPut(DistSignalRecord &record, const CallNode *call,
                       bool has_src_rank) {
    size_t routes_arg = 2;
    size_t src_rank_arg = 3;
    size_t current_core_arg = has_src_rank ? 5 : 4;
    const auto *current_core_node = call->args[current_core_arg].as<VarNode>();
    if (!current_core_node) {
      return;
    }
    Var current_core = tvm::ffi::GetRef<Var>(current_core_node);
    std::vector<NormalRouteEntry> routes =
        ParseNormalRouteTable(call->args[routes_arg]);

    int64_t src_rank_begin = 0;
    int64_t src_rank_end = world_size_;
    if (has_src_rank && !call->args[src_rank_arg].same_as(rank_id_)) {
      const auto *src_rank = call->args[src_rank_arg].as<IntImmNode>();
      if (!src_rank) {
        return;
      }
      src_rank_begin = src_rank->value;
      src_rank_end = src_rank_begin + 1;
    }
    for (int64_t src_rank = src_rank_begin; src_rank < src_rank_end;
         ++src_rank) {
      if (src_rank < 0 || src_rank >= world_size_) {
        continue;
      }
      for (const NormalRouteEntry &route : routes) {
        if (route.origin_src_row < 0 || route.origin_src_row >= mesh_nrows_) {
          continue;
        }
        RecordRoute(record, route.dst_rank, route.dst_row, current_core,
                    route.origin_src_row, src_rank);
      }
    }
  }

  void VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_signal_decl())) {
      StmtExprVisitor::VisitStmt_(let_node);
      return;
    }
    ICHECK_EQ(call->args.size(), 2U);
    std::string requested_kind =
        RequireStringImm(call->args[0], "requested signal kind");
    int64_t logical_id = RequireIntImm(call->args[1], "logical signal id");
    ICHECK(requested_kind == kAutoSignalKind ||
           FindDistSignalKindInfo(requested_kind))
        << "Unsupported requested T.dist signal kind " << requested_kind;
    ICHECK_GE(logical_id, 0);
    ICHECK(!logical_ids_.count(logical_id))
        << "Duplicate logical T.dist signal id " << logical_id;

    size_t record_index = records_.size();
    records_.push_back(
        DistSignalRecord{let_node->var, requested_kind, logical_id});
    logical_ids_.emplace(logical_id, record_index);
    active_.emplace(let_node->var.get(), record_index);
    VisitStmt(let_node->body);
    active_.erase(let_node->var.get());
  }

  void VisitStmt_(const IfThenElseNode *if_node) final {
    VisitExpr(if_node->condition);
    PrimExpr old_predicate = source_predicate_;
    source_predicate_ =
        analyzer_.Simplify(And(old_predicate, if_node->condition));
    VisitStmt(if_node->then_case);
    if (if_node->else_case) {
      source_predicate_ =
          analyzer_.Simplify(And(old_predicate, Not(if_node->condition)));
      VisitStmt(if_node->else_case.value());
    }
    source_predicate_ = old_predicate;
  }

  void VisitStmt_(const ForNode *for_node) final {
    VisitExpr(for_node->min);
    VisitExpr(for_node->extent);
    bool repeated = !analyzer_.CanProve(for_node->extent <= I32(1));
    repeated_loop_depth_ += repeated ? 1 : 0;
    VisitStmt(for_node->body);
    repeated_loop_depth_ -= repeated ? 1 : 0;
  }

  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(DistPutOp::Get())) {
      ICHECK_EQ(call->args.size(), 6U);
      DistSignalRecord &record = RegisterUse(
          call->args[4], NormalizeToBufferRegion(call->args[1]), "T.dist.put");
      RecordLogicalPut(record, call);
    } else if (call->op.same_as(DistWaitSignalOp::Get())) {
      ICHECK_EQ(call->args.size(), 2U);
      RegisterUse(call->args[0], NormalizeToBufferRegion(call->args[1]),
                  "T.dist.wait_signal");
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ICHECK_EQ(call->args.size(), 6U);
      DistSignalRecord &record =
          RegisterUse(call->args[4], NormalizeToBufferRegion(call->args[1]),
                      "T.dist.routed_put");
      RecordRoutedPut(record, call, /*has_src_rank=*/true);
    } else if (call->op.same_as(dist_routed_put())) {
      ICHECK_EQ(call->args.size(), 5U);
      DistSignalRecord &record =
          RegisterUse(call->args[3], NormalizeToBufferRegion(call->args[1]),
                      "T.dist.routed_put");
      RecordRoutedPut(record, call, /*has_src_rank=*/false);
    } else if (call->op.same_as(dist_wait_all())) {
      ICHECK_GE(call->args.size(), 2U);
      BufferRegion destination = NormalizeToBufferRegion(call->args[0]);
      for (size_t index = 1; index < call->args.size(); ++index) {
        RegisterUse(call->args[index], destination, "T.dist.wait_all");
      }
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  DistSignalRecord &RegisterUse(const PrimExpr &signal,
                                const BufferRegion &destination,
                                const char *op_name) {
    const auto *var = signal.as<VarNode>();
    ICHECK(var) << op_name << " expected a signal Var, got " << signal;
    auto it = active_.find(var);
    ICHECK(it != active_.end())
        << op_name << " cannot find the corresponding T.dist.signal for "
        << signal;
    DistSignalRecord &record = records_[it->second];
    DistSignalScope scope = ClassifyDestinationScope(destination, op_name);
    if (record.destination_scope && record.destination_scope.value() != scope) {
      ICHECK(false) << "T.dist.signal " << record.var
                    << " is used with inconsistent destination scopes: "
                    << DestinationScopeName(record.destination_scope.value())
                    << " and " << DestinationScopeName(scope);
    }
    record.destination_scope = scope;
    record.used = true;
    return record;
  }

  Target target_;
  int64_t world_size_;
  Var rank_id_;
  int64_t mesh_nrows_{1};
  int64_t mesh_ncols_{1};
  arith::Analyzer analyzer_;
  PrimExpr source_predicate_{const_true()};
  int repeated_loop_depth_{0};
  std::vector<DistSignalRecord> records_;
  std::unordered_map<const VarNode *, size_t> active_;
  std::unordered_map<int64_t, size_t> logical_ids_;
};

class DistSignalPlanRewriter : public StmtExprMutator {
public:
  explicit DistSignalPlanRewriter(
      const std::unordered_map<const VarNode *, DistSignalRecord> &plans)
      : plans_(plans) {}

private:
  Stmt VisitStmt_(const LetStmtNode *let_node) final {
    const auto *call = let_node->value.as<CallNode>();
    if (!call || !call->op.same_as(dist_signal_decl())) {
      return StmtExprMutator::VisitStmt_(let_node);
    }
    auto it = plans_.find(let_node->var.get());
    ICHECK(it != plans_.end());
    Stmt body = VisitStmt(let_node->body);
    const DistSignalRecord &plan = it->second;
    if (!plan.used) {
      return body;
    }
    Array<PrimExpr> args{
        StringImm(plan.resolved_kind->name),
        IntImm(DataType::Int(32), plan.resolved_index),
    };
    PrimExpr value =
        Call(call->dtype, dist_signal(), args, call->annotations, call->span);
    return LetStmt(let_node->var, value, body, let_node->span);
  }

  const std::unordered_map<const VarNode *, DistSignalRecord> &plans_;
};

PrimFunc PlanSignalResources(PrimFunc func) {
  if (func->GetAttr<Map<ffi::String, Integer>>(kDistSignalCountsAttr)) {
    return func;
  }

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()));
  auto world_size_attr = func->GetAttr<Integer>(kDistWorldSizeAttr);
  int64_t world_size = world_size_attr ? world_size_attr.value()->value : 1;
  auto rank_index = func->GetAttr<Integer>("tl.dist.rank_id_param_index");
  ICHECK(rank_index) << "T.dist passes require a T.dist.RankId parameter";
  ICHECK_GE(rank_index.value()->value, 0);
  ICHECK_LT(rank_index.value()->value,
            static_cast<int64_t>(func->params.size()));
  Var rank_id = func->params[rank_index.value()->value];

  DistSignalUseCollector collector(target.value(), world_size, rank_id);
  collector.Collect(func->body);
  std::vector<DistSignalRecord> &records = collector.Records();
  if (records.empty()) {
    return func;
  }

  std::array<int64_t, static_cast<size_t>(DistSignalKind::kCount)> counts{};
  auto allocate = [&](DistSignalRecord &record, const DistSignalKindInfo &kind,
                      bool inferred) {
    ICHECK(record.used);
    ICHECK(record.destination_scope);
    ICHECK(kind.scope == record.destination_scope.value())
        << "T.dist.signal " << record.var << " explicitly requests "
        << kind.name << " but its destination scope is "
        << DestinationScopeName(record.destination_scope.value());
    size_t kind_index = DistSignalKindIndex(kind.kind);
    if (kind.capacity >= 0 && counts[kind_index] >= kind.capacity) {
      ICHECK(false) << kind.name << " signal capacity exceeded: maximum is "
                    << kind.capacity
                    << (inferred ? "; this topology requires an increment "
                                   "signal"
                                 : "");
    }
    record.resolved_kind = &kind;
    record.resolved_index = counts[kind_index]++;
  };

  for (DistSignalRecord &record : records) {
    if (record.used && record.requested_kind != kAutoSignalKind) {
      allocate(record, *FindDistSignalKindInfo(record.requested_kind),
               /*inferred=*/false);
    }
  }
  for (DistSignalRecord &record : records) {
    if (record.used && record.requested_kind == kAutoSignalKind) {
      bool sram = record.destination_scope.value() == DistSignalScope::kSram;
      DistSignalKind kind;
      // INC aggregates competing senders. A repeated one-to-one route can
      // instead publish its sender generation through VALUE, then MEMORY.
      if (record.HasMultiplePhysicalSenders() ||
          !record.CanUseValueForRepeatedPuts()) {
        kind = sram ? DistSignalKind::kSramFlagregInc
                    : DistSignalKind::kDramFlagregInc;
      } else {
        DistSignalKind value_kind = sram ? DistSignalKind::kSramFlagregValue
                                         : DistSignalKind::kDramFlagregValue;
        const DistSignalKindInfo &value_info =
            DistSignalKindInfos()[DistSignalKindIndex(value_kind)];
        if (counts[DistSignalKindIndex(value_kind)] < value_info.capacity) {
          kind = value_kind;
        } else {
          kind =
              sram ? DistSignalKind::kSramMemory : DistSignalKind::kDramMemory;
        }
      }
      allocate(record, DistSignalKindInfos()[DistSignalKindIndex(kind)],
               /*inferred=*/true);
    }
  }

  std::unordered_map<const VarNode *, DistSignalRecord> plans;
  for (const DistSignalRecord &record : records) {
    plans.emplace(record.var.get(), record);
  }
  Stmt body = DistSignalPlanRewriter(plans)(func->body);
  if (!body.same_as(func->body)) {
    func.CopyOnWrite()->body = std::move(body);
  }
  Map<ffi::String, Integer> signal_counts;
  for (const DistSignalKindInfo &info : DistSignalKindInfos()) {
    signal_counts.Set(info.name,
                      Integer(counts[DistSignalKindIndex(info.kind)]));
  }
  func = WithAttr(std::move(func), kDistSignalCountsAttr, signal_counts);
  return func;
}

class DistCommunicationValidator : public StmtExprVisitor {
public:
  void Validate(const PrimFunc &func) {
    auto world_size = func->GetAttr<Integer>(kDistWorldSizeAttr);
    int64_t world_size_value = world_size ? world_size.value()->value : 1;
    ICHECK_GT(world_size_value, 1)
        << "T.dist communication op appears in a single-Rank kernel: "
           "world_size=1 disables Rank communication";
    if (auto signal_counts =
            func->GetAttr<Map<ffi::String, Integer>>(kDistSignalCountsAttr)) {
      ICHECK_EQ(signal_counts.value().size(), DistSignalKindInfos().size());
      for (const DistSignalKindInfo &info : DistSignalKindInfos()) {
        Optional<Integer> count = signal_counts.value().Get(info.name);
        ICHECK(count) << "Missing T.dist signal count for " << info.name;
        ICHECK_GE(count.value()->value, 0);
        if (info.capacity >= 0) {
          ICHECK_LE(count.value()->value, info.capacity);
        }
      }
    }
    VisitStmt(func->body);
  }

private:
  void VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(dist_signal_decl())) {
      ICHECK(false) << "tl.dist_signal_decl must be resolved by "
                       "PlanDistSignals before validation";
    } else if (call->op.same_as(dist_signal())) {
      ICHECK_EQ(call->args.size(), 2);
      ValidateSignalArgs(call, 0, 1);
      ICHECK(call->dtype.is_handle());
    } else if (call->op.same_as(DistPutOp::Get())) {
      ValidatePut(call);
    } else if (call->op.same_as(DistWaitSignalOp::Get())) {
      ValidateWaitSignal(call);
    } else if (call->op.same_as(dist_rank_routed_put())) {
      ValidateRankRoutedPut(call);
    } else if (call->op.same_as(dist_routed_put())) {
      ValidateRoutedPut(call);
    } else if (call->op.same_as(dist_wait_all())) {
      ValidateWaitAll(call);
    } else if (call->op.same_as(dist_wait_send())) {
      ICHECK_EQ(call->args.size(), 0);
    }
    StmtExprVisitor::VisitExpr_(call);
  }

  void ValidatePut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 6);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype)
        << "T.dist.put source and destination dtypes must match";
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst))
        << "T.dist.put source and destination regions must contain the same "
           "number of elements";
    ICHECK(call->args[2].dtype().is_int())
        << "T.dist.put dst_rank must have integer dtype";
    ICHECK(call->args[3].dtype().is_int())
        << "T.dist.put dst_row must have integer dtype";
    ICHECK(call->args[4].dtype().is_handle())
        << "T.dist.put signal must be a signal handle";
    ICHECK(call->args[5].dtype().is_int())
        << "T.dist.put current_core must have integer dtype";
  }

  void ValidateWaitAll(const CallNode *call) {
    ICHECK_GE(call->args.size(), 2U)
        << "T.dist.wait_all requires a destination and at least one signal";
    BufferRegion dst = NormalizeToBufferRegion(call->args[0]);
    ValidateStaticRegion(dst, "T.dist.wait_all", "destination",
                         /*allow_dram=*/true);
    for (size_t index = 1; index < call->args.size(); ++index) {
      ICHECK(call->args[index].dtype().is_handle())
          << "T.dist.wait_all arguments after destination must be signal "
             "handles";
    }
  }

  void ValidateWaitSignal(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 2);
    ICHECK(call->args[0].dtype().is_handle())
        << "T.dist.wait_signal signal must be a signal handle";
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticRegion(dst, "T.dist.wait_signal", "destination",
                         /*allow_dram=*/true);
  }

  void ValidateRoutedPut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 5U);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.routed_put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype);
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst));
    ICHECK(call->args[3].dtype().is_handle());
    ICHECK(call->args[4].dtype().is_int());
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    ICHECK(!routes.empty()) << "T.dist.routed_put route table cannot be empty";
  }

  void ValidateRankRoutedPut(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 6U);
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    ValidateStaticTransfer(src, dst, "T.dist.routed_put");
    ICHECK(src->buffer->dtype == dst->buffer->dtype);
    ICHECK_EQ(StaticElementCount(src), StaticElementCount(dst));
    ICHECK(call->args[3].dtype().is_int())
        << "T.dist.routed_put src_rank must have integer dtype";
    ICHECK(call->args[4].dtype().is_handle());
    ICHECK(call->args[5].dtype().is_int());
    std::vector<NormalRouteEntry> routes = ParseNormalRouteTable(call->args[2]);
    ICHECK(!routes.empty()) << "T.dist.routed_put route table cannot be empty";
  }
};

PrimFunc Run(PrimFunc func) {
  DistOpDetector detector(/*high_level=*/true);
  if (!detector.Detect(func->body)) {
    return func;
  }
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target && TargetIsSunmmio(target.value()))
      << "T.dist operations only support the SunMMIO target";
  func = PlanSignalResources(std::move(func));
  DistCommunicationValidator().Validate(func);
  return func;
}

} // namespace

tvm::transform::Pass PlanDistSignals() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return Run(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PlanDistSignals", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanDistSignals", PlanDistSignals);
}

} // namespace tl
} // namespace tvm
