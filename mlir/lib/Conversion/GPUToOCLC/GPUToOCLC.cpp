//===-- GPUToOCLC.cpp - conversion from GpuOps to OCLC call Func Ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToOCLC/GPUToOCLCPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOOCLCOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static std::string getDim(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return "0";
  case gpu::Dimension::y:
    return "1";
  case gpu::Dimension::z:
    return "2";
  }
  llvm_unreachable("All dimension enum cases handled above");
}

template <typename Op>
struct GPUOpToOCLCCall : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  GPUOpToOCLCCall<Op>(MLIRContext *context, StringRef libFunc)
      : OpRewritePattern<Op>(context), libFunc(libFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
  std::string libFunc;
};

template <typename Op>
struct GPUOpToVoidOCLCCall : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  GPUOpToVoidOCLCCall<Op>(MLIRContext *context, StringRef libFunc)
      : OpRewritePattern<Op>(context), libFunc(libFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
  std::string libFunc;
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, MLIRContext *ctx,
                           StringRef libFunc) {
  patterns.add<GPUOpToOCLCCall<OpTy>>(ctx, libFunc);
}
template <typename OpTy>
void populatePatternsForVoidOp(RewritePatternSet &patterns, MLIRContext *ctx,
                               StringRef libFunc) {
  patterns.add<GPUOpToVoidOCLCCall<OpTy>>(ctx, libFunc);
}
} // namespace

template <typename Op>
LogicalResult
GPUOpToOCLCCall<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  Operation *module = SymbolTable::getNearestSymbolTable(op);
  std::string name = libFunc;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
      SymbolTable::lookupSymbolIn(module, name));

  if (isa<gpu::SubgroupMmaLoadMatrixOp>(op)) {
    // Only AOp type for the loads needed. Otherwise, it shall be passed via
    // function args and will be embedded into the constructed opaque struct

    gpu::MMAMatrixType mmaType = cast<gpu::MMAMatrixType>(op.getType());
    name.append(mmaType.getOperand());
  }

  if (isa<gpu::BlockIdOp>(op)) {
    std::string dim = getDim(cast<gpu::BlockIdOp>(op).getDimension());
    name.append(dim);
  }

  // Forward declare function if it hasn't already been
  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    FunctionType opFunctionTy = FunctionType::get(
        rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name,
                                           opFunctionTy);
    opFunc.setPrivate();
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

  auto callOp = rewriter.create<func::CallOp>(op->getLoc(), name, op.getType(),
                                              op->getOperands());

  op->replaceAllUsesWith(callOp);
  rewriter.eraseOp(op);

  return success();
}

template <typename Op>
LogicalResult
GPUOpToVoidOCLCCall<Op>::matchAndRewrite(Op op,
                                         PatternRewriter &rewriter) const {
  Operation *module = SymbolTable::getNearestSymbolTable(op);
  std::string name = libFunc;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
      SymbolTable::lookupSymbolIn(module, name));

  // Forward declare function if it hasn't already been
  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    FunctionType opFunctionTy = FunctionType::get(
        rewriter.getContext(), op->getOperandTypes(), rewriter.getNoneType());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name,
                                           opFunctionTy);
    opFunc.setPrivate();
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

  auto callOp = rewriter.create<func::CallOp>(
      op->getLoc(), name, rewriter.getNoneType(), op->getOperands());
  // void return ty, shouldn't have any use.
  rewriter.eraseOp(op);

  return success();
}

class retConverter final : public OpConversionPattern<gpu::ReturnOp> {
public:
  using OpConversionPattern<gpu::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto newOp = rewriter.create<func::ReturnOp>(op->getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

class funcConverter final : public OpConversionPattern<gpu::GPUFuncOp> {
public:
  using OpConversionPattern<gpu::GPUFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto name = op.getName();

    FunctionType opFunctionTy = op.getFunctionType();
    func::FuncOp opFunc =
        rewriter.create<func::FuncOp>(op->getLoc(), name, opFunctionTy);
    opFunc->setAttr("gpu.kernel", rewriter.getUnitAttr());
    Region &gpuFuncOpBody = op.getBody();

    IRMapping map;

    Region &opFuncBody = opFunc.getBody();
    opFuncBody.takeBody(gpuFuncOpBody);

    rewriter.eraseOp(op);

    return success();
  }
};

void mlir::populateGpuOpsToOCLCOpsConversionPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  populatePatternsForOp<gpu::SubgroupMmaLoadMatrixOp>(patterns, ctx,
                                                      "coop_load");
  populatePatternsForOp<gpu::SubgroupMmaComputeOp>(patterns, ctx,
                                                   "coop_compute");
  populatePatternsForVoidOp<gpu::SubgroupMmaStoreMatrixOp>(patterns, ctx,
                                                           "coop_store");
  // wrapper to builtin functions
  populatePatternsForOp<gpu::BlockIdOp>(patterns, ctx, "get_block_id");
  patterns.add<retConverter>(ctx);
  patterns.add<funcConverter>(ctx);
}

namespace {
struct ConvertGpuOpsToOCLCOpsPass
    : public impl::ConvertGpuOpsToOCLCOpsBase<ConvertGpuOpsToOCLCOpsPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertGpuOpsToOCLCOpsPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateGpuOpsToOCLCOpsConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, BuiltinDialect, func::FuncDialect,
                         gpu::GPUDialect>();
  target
      .addIllegalOp<gpu::BlockIdOp, gpu::SubgroupMmaComputeOp,
                    gpu::SubgroupMmaLoadMatrixOp, gpu::SubgroupMmaStoreMatrixOp,
                    gpu::GPUFuncOp, gpu::LaunchFuncOp, gpu::ReturnOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createGpuOpsToOCLCOpsPass() {
  return std::make_unique<ConvertGpuOpsToOCLCOpsPass>();
}
