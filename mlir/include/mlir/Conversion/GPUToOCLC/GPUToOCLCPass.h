//===- GPUToOCLCPass.h - Convert GPU Ops to OCLC target Func Ops-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOOCLC_GPUTOOCLCPASS_H_
#define MLIR_CONVERSION_GPUTOOCLC_GPUTOOCLCPASS_H_
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <memory>

namespace mlir {
class Pass;
template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOOCLCOPS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the GPU dialect to OCLC.
void populateGpuOpsToOCLCOpsConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuOpsToOCLCOpsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOOCLC_GPUTOOCLCPASS_H_
