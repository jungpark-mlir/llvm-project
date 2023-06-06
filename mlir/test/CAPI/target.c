//===---- target.c - Test of C API for MLIR targets  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-transform-test 2>&1 | FileCheck %s

#include "mlir-c/Target.h"
#include "mlir-c/Dialect/SPIRV.h"
#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

void testSimpleSerializationSPIRV(void) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx,
      mlirStringRefCreateFromCString(
          // clang-format off
"module {\n"
"  spirv.module @__spv__kernels Physical32 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses], []>{ \n"
"    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in(\"WorkgroupId\") : !spirv.ptr<vector<3xi32>, Input> \n"
"    spirv.func @kernel_add(%arg0: !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>) \"None\" attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>, workgroup_attributions = 0 : i64} { \n"
"      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input> \n"
"      %0 = spirv.Load \"Input\" %__builtin_var_WorkgroupId___addr : vector<3xi32> \n"
"      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32> \n"
"      %2 = spirv.AccessChain %arg0[%1] : !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>, i32 \n"
"      %3 = spirv.Load \"CrossWorkgroup\" %2 : f32 \n"
"      %4 = spirv.AccessChain %arg1[%1] : !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>, i32 \n"
"      %5 = spirv.Load \"CrossWorkgroup\" %4 : f32 \n"
"      %6 = spirv.FAdd %3, %5 : f32 \n"
"      %7 = spirv.AccessChain %arg2[%1] : !spirv.ptr<!spirv.array<8 x f32>, CrossWorkgroup>, i32 \n"
"      spirv.Store \"CrossWorkgroup\" %7, %6 : f32 \n"
"      spirv.Return \n"
"    } \n"
"  } \n"
"}"));

  // clang-format on
  struct MlirSPIRVBinary spirv_bin;
  // CHECK-LABEL: target serialized.
  if (mlirTargetSpirvSerializedBinCreate(module, &spirv_bin,
                                         true /*emitSymbolName*/,
                                         false /*emitDebugInfo*/)) {
    fprintf(stderr, "target serialized, binary size : %ld bytes\n",
            spirv_bin.length * sizeof(int32_t));
    free(spirv_bin.data);
  }
}

int main(void) {
  printf("Running test \n");
  testSimpleSerializationSPIRV();
  return EXIT_SUCCESS;
}
