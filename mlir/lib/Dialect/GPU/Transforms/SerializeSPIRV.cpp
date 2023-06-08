//===- SerializeSPIRV - MLIR GPU lowering pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a base class for a pass to serialize a gpu module
// into a SPIR-V IR binary. The binary blob is added
// as a string attribute to the gpu module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_GPUSERIALIZESPIRV
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

class GpuSerializeSPIRVPass
    : public impl::GpuSerializeSPIRVBase<GpuSerializeSPIRVPass> {
public:
  GpuSerializeSPIRVPass(bool emitSymbolName, bool emitDebugInfo) {
    spirvOptions = {emitSymbolName, emitDebugInfo};
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool done = false;
    SmallVector<uint32_t, 0> binary;
    for (auto spirvModule : mod.getOps<spirv::ModuleOp>()) {
      if (done) {
        spirvModule.emitError("should only contain one 'spirv.module' op");
        break;
      }
      if (failed(spirv::serialize(spirvModule, binary, spirvOptions))) {
        break;
        done = true;
      }
      StringAttr attr = StringAttr::get(
          &getContext(), StringRef(reinterpret_cast<char *>(binary.data()),
                                   binary.size() * sizeof(uint32_t)));
      spirvModule->setAttr("spirv-bin", attr);
    }
  }

private:
  spirv::SerializationOptions spirvOptions;
};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGpuSerializeSPIRVPass(bool emitSymbolName, bool emitDebugInfo) {
  return std::make_unique<GpuSerializeSPIRVPass>(emitSymbolName, emitDebugInfo);
}
