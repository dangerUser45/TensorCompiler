#include "shape_inference.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace tc::frontend {

bool BroadcastDimsCompatible(int64_t lhs_dim, int64_t rhs_dim) noexcept
{
    if (lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1) {
        return true;
    }
    if (lhs_dim == -1 || rhs_dim == -1) {
        return true;
    }
    return false;
}

int64_t ResolveBroadcastDim(int64_t lhs_dim, int64_t rhs_dim) noexcept
{
    if (lhs_dim == rhs_dim) {
        return lhs_dim;
    }
    if (lhs_dim == 1) {
        return rhs_dim == -1 ? -1 : rhs_dim;
    }
    if (rhs_dim == 1) {
        return lhs_dim == -1 ? -1 : lhs_dim;
    }
    if (lhs_dim == -1 && rhs_dim == -1) {
        return -1;
    }
    if (lhs_dim == -1) {
        return rhs_dim;
    }
    if (rhs_dim == -1) {
        return lhs_dim;
    }
    return -1;
}

bool ComputeBroadcastShape(const std::vector<int64_t>& lhs_shape,
                           const std::vector<int64_t>& rhs_shape,
                           std::vector<int64_t>& out_shape)
{
    const std::size_t rank = std::max(lhs_shape.size(), rhs_shape.size());
    out_shape.assign(rank, -1);

    for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::size_t lhs_offset = rank - lhs_shape.size();
        const std::size_t rhs_offset = rank - rhs_shape.size();
        const int64_t lhs_dim =
            axis < lhs_offset ? 1 : lhs_shape[axis - lhs_offset];
        const int64_t rhs_dim =
            axis < rhs_offset ? 1 : rhs_shape[axis - rhs_offset];

        if (!BroadcastDimsCompatible(lhs_dim, rhs_dim)) {
            return false;
        }
        out_shape[axis] = ResolveBroadcastDim(lhs_dim, rhs_dim);
    }

    return true;
}

bool ComputeChannelBiasBroadcastShape(const std::vector<int64_t>& lhs_shape,
                                      const std::vector<int64_t>& rhs_shape,
                                      std::vector<int64_t>& out_shape)
{
    if (lhs_shape.size() == 4 && rhs_shape.size() == 1 &&
        BroadcastDimsCompatible(lhs_shape[1], rhs_shape[0])) {
        out_shape = lhs_shape;
        return true;
    }
    if (lhs_shape.size() == 1 && rhs_shape.size() == 4 &&
        BroadcastDimsCompatible(lhs_shape[0], rhs_shape[1])) {
        out_shape = rhs_shape;
        return true;
    }
    return false;
}

bool HasOnlyStaticDims(const std::vector<int64_t>& shape) noexcept
{
    return std::all_of(
        shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
}

int64_t StaticElementCount(const std::vector<int64_t>& shape) noexcept
{
    int64_t count = 1;
    for (const int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

bool InferReshapeOutputShape(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& target_shape_spec,
                             std::vector<int64_t>& out_shape,
                             std::string& out_error)
{
    out_shape.clear();
    out_error.clear();
    if (target_shape_spec.empty()) {
        out_error = "target shape is empty";
        return false;
    }

    std::optional<std::size_t> infer_axis;
    out_shape = target_shape_spec;
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
        const int64_t dim = out_shape[i];
        if (dim == -1) {
            if (infer_axis.has_value()) {
                out_error = "target shape contains multiple -1 dimensions";
                return false;
            }
            infer_axis = i;
            continue;
        }
        if (dim == 0) {
            if (i >= input_shape.size()) {
                out_error = "target shape contains axis copy out of range";
                return false;
            }
            out_shape[i] = input_shape[i];
            continue;
        }
        if (dim < -1) {
            out_error = "target shape contains invalid negative dimension";
            return false;
        }
    }

    const bool input_static = HasOnlyStaticDims(input_shape);
    const int64_t input_count =
        input_static ? StaticElementCount(input_shape) : -1;

    bool known_target_static = true;
    int64_t known_target_count = 1;
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
        if (infer_axis.has_value() && *infer_axis == i) {
            continue;
        }
        if (out_shape[i] < 0) {
            known_target_static = false;
            break;
        }
        known_target_count *= out_shape[i];
    }

    if (infer_axis.has_value()) {
        if (!input_static || !known_target_static || known_target_count == 0 ||
            input_count % known_target_count != 0) {
            out_error = "cannot infer -1 dimension from input and target shape";
            return false;
        }
        out_shape[*infer_axis] = input_count / known_target_count;
        return true;
    }

    if (input_static && known_target_static &&
        input_count != known_target_count) {
        out_error = "input and target shapes have different element counts";
        return false;
    }
    return true;
}

int64_t ComputeSpatialOutputSize(int64_t input_size,
                                 int64_t kernel_size,
                                 int64_t stride,
                                 int64_t pad_begin,
                                 int64_t pad_end,
                                 int64_t dilation) noexcept
{
    return (input_size + pad_begin + pad_end - dilation * (kernel_size - 1) -
            1) /
               stride +
           1;
}

} // namespace tc::frontend
