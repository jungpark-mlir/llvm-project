//===----------- mlir-c/Target.h - C API to MLIR Targets ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C APIs for accessing various target supports.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_TARGET_H
#define MLIR_C_TARGET_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

struct MlirSPIRVBinary {
  int32_t *data; ///< Pointer to the first symbol.
  size_t length;   ///< Length of the fragment.
};

/// Serializes the given
/// SPIR-V `module` and writes to `bin->data`. On failure, reports errors to the
/// error handler registered with the MLIR context for `module`.
MLIR_CAPI_EXPORTED bool mlirTargetSpirvSerializedBinCreate(MlirModule module,
                                                           struct MlirSPIRVBinary *bin,
                                                           bool emitSymbolName,
                                                           bool emitDebugInfo);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_H
