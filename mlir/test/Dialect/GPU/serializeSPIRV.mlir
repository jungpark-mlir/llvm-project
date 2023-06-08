// RUN: mlir-opt -allow-unregistered-dialect --spirv-update-vce --gpu-serialize-spirv -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: "spirv-bin" = "\03\02
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>} {
  spirv.module @__spv__kernels Physical64 OpenCL {
    spirv.func @basic_module_structure(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<12 x f32>, CrossWorkgroup>) "None" attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>, workgroup_attributions = 0 : i64} {
      spirv.Return
    }
  }
  gpu.module @kernels {
    gpu.func @basic_module_structure(%arg0: f32, %arg1: memref<12xf32, #spirv.storage_class<CrossWorkgroup>>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      gpu.return
    }
  }
  func.func @main() {
    %0 = "op"() : () -> f32
    %1 = "op"() : () -> memref<12xf32, #spirv.storage_class<CrossWorkgroup>>
    %c1 = arith.constant 1 : index
    gpu.launch_func  @kernels::@basic_module_structure blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%0 : f32, %1 : memref<12xf32, #spirv.storage_class<CrossWorkgroup>>)
    return
  }
}