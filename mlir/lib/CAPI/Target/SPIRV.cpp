//===----- SPIRV.cpp - C API for SPIRV Target utilities  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

using namespace mlir;

MLIR_CAPI_EXPORTED bool mlirTargetSpirvSerializedBinCreate(MlirModule module,
                                                           MlirSPIRVBinary *bin,
                                                           bool emitSymbolName,
                                                           bool emitDebugInfo) {
  auto mod = unwrap(module);
  spirv::SerializationOptions spirvOptions = {emitSymbolName, emitDebugInfo};

  bool done = false;
  SmallVector<uint32_t, 0> binary;
  for (auto spirvModule : mod.getOps<spirv::ModuleOp>()) {
    if (done) {
      spirvModule.emitError("should only contain one 'spirv.module' op");
      return false;
    }
    done = true;

    if (failed(spirv::serialize(spirvModule, binary, spirvOptions)))
      return false;
  }

  int32_t *binaryShader = new int32_t(binary.size());
  std::memcpy(binaryShader, reinterpret_cast<char *>(binary.data()),
              binary.size());
  bin->data = binaryShader;
  bin->length = binary.size()*sizeof(int32_t);
  return true;
}
