module {
  // tc.graph: Convf32
  // tc.node_count: 2
  // tc.source: microsoft/onnxruntime onnxruntime/test/testdata/conv_default_attrs.onnx
  func.func @tc_model(%arg0: tensor<1x2x8x8xf32>) -> tensor<1x2x7x7xf32> {
    // tc.node[0]: op=Conv, name=Conv0
    %cst0 = arith.constant dense<[-1.5, 0.0, 0.200000003, 1.5, -1.5, 0.0, 0.200000003, 1.5, -1.0, 0.0, 0.133300006, 1.0, -1.0, 0.0, 0.133300006, 1.0]> : tensor<2x2x2x2xf32>
    %t0 = tensor.empty() : tensor<1x2x7x7xf32>
    %t1 = linalg.conv_2d_nchw_fchw ins(%arg0, %cst0 : tensor<1x2x8x8xf32>, tensor<2x2x2x2xf32>) outs(%t0 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
    // tc.node[1]: op=Add, name=Conv0_bias
    %cst1 = arith.constant dense<[0.0, 0.0]> : tensor<2xf32>
    %t2 = tensor.empty() : tensor<1x2x7x7xf32>
    %t3 = linalg.broadcast ins(%cst1 : tensor<2xf32>) outs(%t2 : tensor<1x2x7x7xf32>) dimensions = [1]
    %t4 = arith.addf %t1, %t3 : tensor<1x2x7x7xf32>
    return %t4 : tensor<1x2x7x7xf32>
  }
}
