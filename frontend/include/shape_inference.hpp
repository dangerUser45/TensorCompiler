#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tc::frontend {

bool BroadcastDimsCompatible(int64_t lhs_dim, int64_t rhs_dim) noexcept;

int64_t ResolveBroadcastDim(int64_t lhs_dim, int64_t rhs_dim) noexcept;

bool ComputeBroadcastShape(const std::vector<int64_t>& lhs_shape,
                           const std::vector<int64_t>& rhs_shape,
                           std::vector<int64_t>& out_shape);

bool ComputeChannelBiasBroadcastShape(const std::vector<int64_t>& lhs_shape,
                                      const std::vector<int64_t>& rhs_shape,
                                      std::vector<int64_t>& out_shape);

bool HasOnlyStaticDims(const std::vector<int64_t>& shape) noexcept;

int64_t StaticElementCount(const std::vector<int64_t>& shape) noexcept;

bool InferReshapeOutputShape(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& target_shape_spec,
                             std::vector<int64_t>& out_shape,
                             std::string& out_error);

int64_t ComputeSpatialOutputSize(int64_t input_size,
                                 int64_t kernel_size,
                                 int64_t stride,
                                 int64_t pad_begin,
                                 int64_t pad_end,
                                 int64_t dilation) noexcept;

} // namespace tc::frontend
