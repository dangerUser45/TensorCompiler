#include "codegen_driver.hpp"

#include "frontend_mlir.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <llvm/TargetParser/Host.h>

namespace {

std::string ShellQuote(const std::filesystem::path& path)
{
    std::string quoted = "'";
    for (const char c : path.string()) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::filesystem::path MakeTempPath(const std::string& stem,
                                   const std::string& extension)
{
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "_" + std::to_string(tick) + extension);
}

bool ReadFileToString(const std::filesystem::path& path, std::string& out_text)
{
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out_text = buffer.str();
    return true;
}

bool WriteStringToFile(const std::filesystem::path& path,
                       const std::string& text)
{
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << text;
    return out.good();
}

struct TensorStorage final
{
    enum class Kind
    {
        kInput,
        kConstant,
        kBuffer,
        kEmpty
    };

    Kind kind = Kind::kEmpty;
    tc::backend::TensorType type;
    std::string pointer_name;
    std::vector<float> constant_values;
    bool constant_is_splat = false;
};

std::string FloatLiteral(float value)
{
    const double promoted = static_cast<double>(value);
    std::uint64_t bits = 0;
    std::memcpy(&bits, &promoted, sizeof(bits));
    std::ostringstream out;
    out << "0x" << std::uppercase << std::hex << std::setw(16)
        << std::setfill('0') << bits;
    return out.str();
}

std::vector<int64_t> CoordinatesFromLinear(const tc::backend::TensorType& type,
                                           std::size_t linear_index)
{
    std::vector<int64_t> coords(type.shape.size(), 0);
    for (std::size_t axis = type.shape.size(); axis > 0; --axis) {
        const std::size_t i = axis - 1;
        const std::size_t dim = static_cast<std::size_t>(type.shape[i]);
        coords[i] = static_cast<int64_t>(linear_index % dim);
        linear_index /= dim;
    }
    return coords;
}

std::size_t LinearFromCoordinates(const tc::backend::TensorType& type,
                                  const std::vector<int64_t>& coords)
{
    std::size_t linear = 0;
    for (std::size_t i = 0; i < type.shape.size(); ++i) {
        linear *= static_cast<std::size_t>(type.shape[i]);
        linear += static_cast<std::size_t>(coords[i]);
    }
    return linear;
}

class LlvmIrEmitter final
{
public:
    bool Emit(const tc::backend::FrontendMlirModule& module,
              const std::string& target_triple,
              std::string& out_ir,
              tc::backend::BackendDiagnostic& diagnostic)
    {
        module_ = &module;
        diagnostic = tc::backend::BackendDiagnostic{};
        values_.clear();
        temp_id_ = 0;
        buffer_id_ = 0;

        if (!ValidateModule(diagnostic)) {
            return false;
        }

        const std::string effective_triple =
            target_triple.empty() ? llvm::sys::getDefaultTargetTriple()
                                  : target_triple;

        out_ << "; ModuleID = 'tensor_compiler_frontend_mlir'\n";
        out_ << "target triple = \"" << effective_triple << "\"\n\n";
        EmitFunctionHeader();

        for (const auto& op : module.ops) {
            if (!EmitOp(op, diagnostic)) {
                return false;
            }
        }

        out_ << "}\n";
        out_ir = out_.str();
        return true;
    }

private:
    bool ValidateModule(tc::backend::BackendDiagnostic& diagnostic)
    {
        if (module_->entry_name != "tc_model") {
            return Fail(diagnostic, "entry function must be @tc_model");
        }
        if (module_->inputs.empty()) {
            return Fail(diagnostic, "tc_model must have at least one input");
        }
        if (module_->output_type.element_type != "f32") {
            return Fail(diagnostic, "tc_model output must be f32 tensor");
        }
        for (const auto& input : module_->inputs) {
            if (input.type.element_type != "f32") {
                return Fail(diagnostic, "tc_model input must be f32 tensor");
            }
        }
        return true;
    }

    bool Fail(tc::backend::BackendDiagnostic& diagnostic,
              const std::string& message)
    {
        diagnostic.stage = "llvm-ir-emission";
        diagnostic.message = message;
        return false;
    }

    std::string NextTemp()
    {
        return "%v" + std::to_string(temp_id_++);
    }

    std::string NextBuffer()
    {
        return "%buf" + std::to_string(buffer_id_++);
    }

    void EmitFunctionHeader()
    {
        out_ << "define void @tc_model(";
        for (std::size_t i = 0; i < module_->inputs.size(); ++i) {
            if (i != 0) {
                out_ << ", ";
            }
            out_ << "ptr nocapture readonly %arg" << i << "_buf";

            TensorStorage storage;
            storage.kind = TensorStorage::Kind::kInput;
            storage.type = module_->inputs[i].type;
            storage.pointer_name = "%arg" + std::to_string(i) + "_buf";
            values_[module_->inputs[i].name] = std::move(storage);
        }

        out_ << ", ptr nocapture writeonly %out_buf) local_unnamed_addr {\n";
        out_ << "entry:\n";
    }

    TensorStorage* FindValue(const std::string& name)
    {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    const TensorStorage* FindValue(const std::string& name) const
    {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    TensorStorage CreateBuffer(const std::string& result_name,
                               const tc::backend::TensorType& type)
    {
        TensorStorage storage;
        storage.kind = TensorStorage::Kind::kBuffer;
        storage.type = type;
        storage.pointer_name = NextBuffer();
        const std::size_t count = type.element_count();
        out_ << "  " << storage.pointer_name << " = alloca [" << count
             << " x float], align 16\n";
        values_[result_name] = storage;
        return storage;
    }

    std::string ReadElement(const TensorStorage& storage,
                            std::size_t linear_index)
    {
        switch (storage.kind) {
            case TensorStorage::Kind::kConstant: {
                const float value =
                    storage.constant_is_splat
                        ? storage.constant_values.front()
                        : storage.constant_values.at(linear_index);
                return FloatLiteral(value);
            }
            case TensorStorage::Kind::kInput: {
                const std::string ptr = NextTemp();
                const std::string value = NextTemp();
                out_ << "  " << ptr << " = getelementptr inbounds float, ptr "
                     << storage.pointer_name << ", i64 " << linear_index
                     << '\n';
                out_ << "  " << value << " = load float, ptr " << ptr
                     << ", align 4\n";
                return value;
            }
            case TensorStorage::Kind::kBuffer: {
                const std::string ptr = NextTemp();
                const std::string value = NextTemp();
                out_ << "  " << ptr << " = getelementptr inbounds ["
                     << storage.type.element_count() << " x float], ptr "
                     << storage.pointer_name << ", i64 0, i64 " << linear_index
                     << '\n';
                out_ << "  " << value << " = load float, ptr " << ptr
                     << ", align 4\n";
                return value;
            }
            case TensorStorage::Kind::kEmpty:
                break;
        }
        return "0.000000000e+00";
    }

    void StoreElement(const TensorStorage& storage,
                      std::size_t linear_index,
                      const std::string& value)
    {
        const std::string ptr = NextTemp();
        out_ << "  " << ptr << " = getelementptr inbounds ["
             << storage.type.element_count() << " x float], ptr "
             << storage.pointer_name << ", i64 0, i64 " << linear_index << '\n';
        out_ << "  store float " << value << ", ptr " << ptr << ", align 4\n";
    }

    void StoreOutputElement(std::size_t linear_index, const std::string& value)
    {
        const std::string ptr = NextTemp();
        out_ << "  " << ptr << " = getelementptr inbounds float, ptr %out_buf"
             << ", i64 " << linear_index << '\n';
        out_ << "  store float " << value << ", ptr " << ptr << ", align 4\n";
    }

    bool EmitOp(const tc::backend::FrontendMlirOp& op,
                tc::backend::BackendDiagnostic& diagnostic)
    {
        switch (op.kind) {
            case tc::backend::FrontendMlirOpKind::kConstant:
                return EmitConstant(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kTensorEmpty:
                return EmitTensorEmpty(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kBroadcast:
                return EmitBroadcast(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kAddF:
            case tc::backend::FrontendMlirOpKind::kMulF:
            case tc::backend::FrontendMlirOpKind::kMaximumF:
                return EmitElementwise(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kMatMul:
                return EmitMatMul(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kReshape:
                return EmitReshape(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kMaxPoolNchw:
                return EmitMaxPool(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kTranspose:
                return EmitTranspose(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kConv2DNchwFchw:
                return EmitConv(op, diagnostic);
            case tc::backend::FrontendMlirOpKind::kReturn:
                return EmitReturn(op, diagnostic);
        }
        return Fail(diagnostic, "unsupported frontend MLIR operation");
    }

    bool EmitConstant(const tc::backend::FrontendMlirOp& op,
                      tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.result_type.element_type != "f32") {
            return Fail(diagnostic, "only f32 constants are supported");
        }
        TensorStorage storage;
        storage.kind = TensorStorage::Kind::kConstant;
        storage.type = op.result_type;
        storage.constant_values = op.constant_values;
        storage.constant_is_splat = op.constant_is_splat;
        if (!storage.constant_is_splat &&
            storage.constant_values.size() != storage.type.element_count()) {
            return Fail(diagnostic,
                        "constant element count does not match type");
        }
        values_[op.result] = std::move(storage);
        return true;
    }

    bool EmitTensorEmpty(const tc::backend::FrontendMlirOp& op,
                         tc::backend::BackendDiagnostic&)
    {
        TensorStorage storage;
        storage.kind = TensorStorage::Kind::kEmpty;
        storage.type = op.result_type;
        values_[op.result] = std::move(storage);
        return true;
    }

    bool EmitBroadcast(const tc::backend::FrontendMlirOp& op,
                       tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.empty()) {
            return Fail(diagnostic, "broadcast has no input");
        }
        const TensorStorage* input = FindValue(op.operands[0]);
        if (!input) {
            return Fail(diagnostic,
                        "unknown broadcast input " + op.operands[0]);
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        for (std::size_t out_linear = 0;
             out_linear < output.type.element_count();
             ++out_linear) {
            const std::vector<int64_t> out_coords =
                CoordinatesFromLinear(output.type, out_linear);
            std::vector<int64_t> in_coords(input->type.shape.size(), 0);
            for (std::size_t i = 0; i < input->type.shape.size(); ++i) {
                const int64_t out_axis =
                    i < op.dimensions.size()
                        ? op.dimensions[i]
                        : static_cast<int64_t>(output.type.shape.size() -
                                               input->type.shape.size() + i);
                if (out_axis >= 0 &&
                    static_cast<std::size_t>(out_axis) < out_coords.size() &&
                    input->type.shape[i] != 1) {
                    in_coords[i] =
                        out_coords[static_cast<std::size_t>(out_axis)];
                }
            }
            const std::size_t in_linear =
                LinearFromCoordinates(input->type, in_coords);
            StoreElement(output, out_linear, ReadElement(*input, in_linear));
        }
        return true;
    }

    bool EmitElementwise(const tc::backend::FrontendMlirOp& op,
                         tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.size() != 2) {
            return Fail(diagnostic, "elementwise op expects two operands");
        }
        const TensorStorage* lhs = FindValue(op.operands[0]);
        const TensorStorage* rhs = FindValue(op.operands[1]);
        if (!lhs || !rhs) {
            return Fail(diagnostic, "unknown elementwise operand");
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        for (std::size_t i = 0; i < output.type.element_count(); ++i) {
            const std::string lhs_value = ReadElement(*lhs, i);
            const std::string rhs_value = ReadElement(*rhs, i);
            std::string result;
            if (op.kind == tc::backend::FrontendMlirOpKind::kAddF) {
                result = NextTemp();
                out_ << "  " << result << " = fadd float " << lhs_value << ", "
                     << rhs_value << '\n';
            } else if (op.kind == tc::backend::FrontendMlirOpKind::kMulF) {
                result = NextTemp();
                out_ << "  " << result << " = fmul float " << lhs_value << ", "
                     << rhs_value << '\n';
            } else {
                const std::string cmp = NextTemp();
                result = NextTemp();
                out_ << "  " << cmp << " = fcmp ogt float " << lhs_value << ", "
                     << rhs_value << '\n';
                out_ << "  " << result << " = select i1 " << cmp << ", float "
                     << lhs_value << ", float " << rhs_value << '\n';
            }
            StoreElement(output, i, result);
        }
        return true;
    }

    bool EmitMatMul(const tc::backend::FrontendMlirOp& op,
                    tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.size() < 2) {
            return Fail(diagnostic, "matmul expects two inputs");
        }
        const TensorStorage* lhs = FindValue(op.operands[0]);
        const TensorStorage* rhs = FindValue(op.operands[1]);
        if (!lhs || !rhs || lhs->type.shape.size() != 2 ||
            rhs->type.shape.size() != 2) {
            return Fail(diagnostic, "matmul operands must be rank-2 tensors");
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        const std::size_t m = static_cast<std::size_t>(lhs->type.shape[0]);
        const std::size_t k = static_cast<std::size_t>(lhs->type.shape[1]);
        const std::size_t n = static_cast<std::size_t>(rhs->type.shape[1]);
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                std::string acc = "0.000000000e+00";
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const std::string product = NextTemp();
                    const std::string sum = NextTemp();
                    const std::string lhs_value =
                        ReadElement(*lhs, row * k + kk);
                    const std::string rhs_value =
                        ReadElement(*rhs, kk * n + col);
                    out_ << "  " << product << " = fmul float " << lhs_value
                         << ", " << rhs_value << '\n';
                    out_ << "  " << sum << " = fadd float " << acc << ", "
                         << product << '\n';
                    acc = sum;
                }
                StoreElement(output, row * n + col, acc);
            }
        }
        return true;
    }

    bool EmitReshape(const tc::backend::FrontendMlirOp& op,
                     tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.size() != 1) {
            return Fail(diagnostic, "reshape expects one input");
        }
        const TensorStorage* input = FindValue(op.operands[0]);
        if (!input) {
            return Fail(diagnostic, "unknown reshape input " + op.operands[0]);
        }
        if (input->type.element_count() != op.result_type.element_count()) {
            return Fail(diagnostic,
                        "reshape input and output element counts must match");
        }

        TensorStorage output = *input;
        output.type = op.result_type;
        values_[op.result] = std::move(output);
        return true;
    }

    bool EmitMaxPool(const tc::backend::FrontendMlirOp& op,
                     tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.empty()) {
            return Fail(diagnostic, "MaxPool expects an input");
        }
        const TensorStorage* input = FindValue(op.operands[0]);
        if (!input || input->type.shape.size() != 4 ||
            op.result_type.shape.size() != 4) {
            return Fail(diagnostic, "MaxPool expects rank-4 NCHW tensors");
        }
        if (op.kernel_shape.size() != 2 || op.strides.size() != 2 ||
            op.kernel_shape[0] <= 0 || op.kernel_shape[1] <= 0 ||
            op.strides[0] <= 0 || op.strides[1] <= 0) {
            return Fail(diagnostic,
                        "MaxPool expects positive 2D kernel and strides");
        }

        const std::size_t n_size =
            static_cast<std::size_t>(op.result_type.shape[0]);
        const std::size_t c_size =
            static_cast<std::size_t>(op.result_type.shape[1]);
        const std::size_t out_h =
            static_cast<std::size_t>(op.result_type.shape[2]);
        const std::size_t out_w =
            static_cast<std::size_t>(op.result_type.shape[3]);
        const std::size_t in_c = static_cast<std::size_t>(input->type.shape[1]);
        const std::size_t in_h = static_cast<std::size_t>(input->type.shape[2]);
        const std::size_t in_w = static_cast<std::size_t>(input->type.shape[3]);
        const std::size_t kernel_h =
            static_cast<std::size_t>(op.kernel_shape[0]);
        const std::size_t kernel_w =
            static_cast<std::size_t>(op.kernel_shape[1]);
        const std::size_t stride_h = static_cast<std::size_t>(op.strides[0]);
        const std::size_t stride_w = static_cast<std::size_t>(op.strides[1]);
        if (c_size != in_c ||
            (out_h > 0 && (out_h - 1) * stride_h + kernel_h > in_h) ||
            (out_w > 0 && (out_w - 1) * stride_w + kernel_w > in_w)) {
            return Fail(diagnostic, "MaxPool output shape is incompatible");
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        for (std::size_t n = 0; n < n_size; ++n) {
            for (std::size_t c = 0; c < c_size; ++c) {
                for (std::size_t oh = 0; oh < out_h; ++oh) {
                    for (std::size_t ow = 0; ow < out_w; ++ow) {
                        const std::size_t base_h = oh * stride_h;
                        const std::size_t base_w = ow * stride_w;
                        std::string acc;
                        for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                                const std::size_t input_linear =
                                    ((n * in_c + c) * in_h + (base_h + kh)) *
                                        in_w +
                                    (base_w + kw);
                                const std::string value =
                                    ReadElement(*input, input_linear);
                                if (acc.empty()) {
                                    acc = value;
                                    continue;
                                }
                                const std::string cmp = NextTemp();
                                const std::string next = NextTemp();
                                out_ << "  " << cmp << " = fcmp ogt float "
                                     << acc << ", " << value << '\n';
                                out_ << "  " << next << " = select i1 " << cmp
                                     << ", float " << acc << ", float " << value
                                     << '\n';
                                acc = next;
                            }
                        }
                        const std::size_t out_linear =
                            ((n * c_size + c) * out_h + oh) * out_w + ow;
                        StoreElement(output, out_linear, acc);
                    }
                }
            }
        }
        return true;
    }

    bool EmitTranspose(const tc::backend::FrontendMlirOp& op,
                       tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.empty()) {
            return Fail(diagnostic, "transpose expects an input");
        }
        const TensorStorage* input = FindValue(op.operands[0]);
        if (!input || op.dimensions.size() != input->type.shape.size()) {
            return Fail(diagnostic, "invalid transpose permutation");
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        for (std::size_t out_linear = 0;
             out_linear < output.type.element_count();
             ++out_linear) {
            const std::vector<int64_t> out_coords =
                CoordinatesFromLinear(output.type, out_linear);
            std::vector<int64_t> in_coords(input->type.shape.size(), 0);
            for (std::size_t i = 0; i < op.dimensions.size(); ++i) {
                in_coords[static_cast<std::size_t>(op.dimensions[i])] =
                    out_coords[i];
            }
            StoreElement(
                output,
                out_linear,
                ReadElement(*input,
                            LinearFromCoordinates(input->type, in_coords)));
        }
        return true;
    }

    bool EmitConv(const tc::backend::FrontendMlirOp& op,
                  tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.size() < 2) {
            return Fail(diagnostic, "conv expects input and weight");
        }
        const TensorStorage* input = FindValue(op.operands[0]);
        const TensorStorage* weight = FindValue(op.operands[1]);
        if (!input || !weight || input->type.shape.size() != 4 ||
            weight->type.shape.size() != 4 ||
            op.result_type.shape.size() != 4) {
            return Fail(diagnostic, "conv expects rank-4 NCHW/FCHW tensors");
        }

        TensorStorage output = CreateBuffer(op.result, op.result_type);
        const std::size_t n_size =
            static_cast<std::size_t>(op.result_type.shape[0]);
        const std::size_t f_size =
            static_cast<std::size_t>(op.result_type.shape[1]);
        const std::size_t out_h =
            static_cast<std::size_t>(op.result_type.shape[2]);
        const std::size_t out_w =
            static_cast<std::size_t>(op.result_type.shape[3]);
        const std::size_t c_size =
            static_cast<std::size_t>(input->type.shape[1]);
        const std::size_t in_h = static_cast<std::size_t>(input->type.shape[2]);
        const std::size_t in_w = static_cast<std::size_t>(input->type.shape[3]);
        const std::size_t kernel_h =
            static_cast<std::size_t>(weight->type.shape[2]);
        const std::size_t kernel_w =
            static_cast<std::size_t>(weight->type.shape[3]);
        const std::vector<int64_t> pads =
            op.pads.empty() ? std::vector<int64_t>{ 0, 0, 0, 0 } : op.pads;
        if (pads.size() != 4 ||
            std::any_of(pads.begin(), pads.end(), [](int64_t pad) {
                return pad < 0;
            })) {
            return Fail(diagnostic,
                        "conv pads must contain four non-negative values");
        }
        const int64_t pad_top = pads[0];
        const int64_t pad_left = pads[1];

        for (std::size_t n = 0; n < n_size; ++n) {
            for (std::size_t f = 0; f < f_size; ++f) {
                for (std::size_t oh = 0; oh < out_h; ++oh) {
                    for (std::size_t ow = 0; ow < out_w; ++ow) {
                        std::string acc = "0.000000000e+00";
                        for (std::size_t c = 0; c < c_size; ++c) {
                            for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                                    const int64_t ih =
                                        static_cast<int64_t>(oh + kh) - pad_top;
                                    const int64_t iw =
                                        static_cast<int64_t>(ow + kw) -
                                        pad_left;
                                    if (ih < 0 || iw < 0 ||
                                        ih >= static_cast<int64_t>(in_h) ||
                                        iw >= static_cast<int64_t>(in_w)) {
                                        continue;
                                    }
                                    const std::size_t input_linear =
                                        ((n * c_size + c) * in_h +
                                         static_cast<std::size_t>(ih)) *
                                            in_w +
                                        static_cast<std::size_t>(iw);
                                    const std::size_t weight_linear =
                                        ((f * c_size + c) * kernel_h + kh) *
                                            kernel_w +
                                        kw;
                                    const std::string product = NextTemp();
                                    const std::string sum = NextTemp();
                                    const std::string input_value =
                                        ReadElement(*input, input_linear);
                                    const std::string weight_value =
                                        ReadElement(*weight, weight_linear);
                                    out_ << "  " << product << " = fmul float "
                                         << input_value << ", " << weight_value
                                         << '\n';
                                    out_ << "  " << sum << " = fadd float "
                                         << acc << ", " << product << '\n';
                                    acc = sum;
                                }
                            }
                        }
                        const std::size_t out_linear =
                            ((n * f_size + f) * out_h + oh) * out_w + ow;
                        StoreElement(output, out_linear, acc);
                    }
                }
            }
        }
        return true;
    }

    bool EmitReturn(const tc::backend::FrontendMlirOp& op,
                    tc::backend::BackendDiagnostic& diagnostic)
    {
        if (op.operands.empty()) {
            return Fail(diagnostic, "return expects a tensor");
        }
        const TensorStorage* value = FindValue(op.operands[0]);
        if (!value) {
            return Fail(diagnostic, "unknown return value " + op.operands[0]);
        }

        for (std::size_t i = 0; i < module_->output_type.element_count(); ++i) {
            StoreOutputElement(i, ReadElement(*value, i));
        }
        out_ << "  ret void\n";
        return true;
    }

    const tc::backend::FrontendMlirModule* module_ = nullptr;
    std::unordered_map<std::string, TensorStorage> values_;
    std::ostringstream out_;
    std::size_t temp_id_ = 0;
    std::size_t buffer_id_ = 0;
};

} // namespace

namespace tc::backend {

bool EmitLlvmIrFromMlirText(const std::string& mlir_text,
                            const std::string& source_name,
                            const std::string& pass_pipeline,
                            const std::string& target_triple,
                            std::string& out_llvm_ir,
                            BackendDiagnostic& out_diagnostic)
{
    (void)pass_pipeline;
    out_llvm_ir.clear();
    out_diagnostic = BackendDiagnostic{};

    FrontendMlirModule module;
    if (!ParseFrontendMlirModule(
            mlir_text, source_name, module, out_diagnostic)) {
        return false;
    }

    LlvmIrEmitter emitter;
    return emitter.Emit(module, target_triple, out_llvm_ir, out_diagnostic);
}

bool EmitAsmFromMlirText(const std::string& mlir_text,
                         const std::string& source_name,
                         const std::string& pass_pipeline,
                         const std::string& target_triple,
                         std::string& out_asm,
                         BackendDiagnostic& out_diagnostic)
{
    out_asm.clear();

    std::string llvm_ir;
    if (!EmitLlvmIrFromMlirText(mlir_text,
                                source_name,
                                pass_pipeline,
                                target_triple,
                                llvm_ir,
                                out_diagnostic)) {
        return false;
    }

    const std::filesystem::path ir_path =
        MakeTempPath("tc_backend_llvm", ".ll");
    const std::filesystem::path asm_path = MakeTempPath("tc_backend_asm", ".s");
    if (!WriteStringToFile(ir_path, llvm_ir)) {
        out_diagnostic.stage = "asm-emission";
        out_diagnostic.message = "failed to write temporary LLVM IR file";
        return false;
    }

    std::string command = "llc -O0 -filetype=asm";
    if (!target_triple.empty()) {
        command += " -mtriple=" + target_triple;
    }
    command += " -o " + ShellQuote(asm_path) + " " + ShellQuote(ir_path);

    if (std::system(command.c_str()) != 0) {
        out_diagnostic.stage = "asm-emission";
        out_diagnostic.message = "llc failed while emitting assembly";
        std::error_code ec;
        std::filesystem::remove(ir_path, ec);
        std::filesystem::remove(asm_path, ec);
        return false;
    }

    if (!ReadFileToString(asm_path, out_asm)) {
        out_diagnostic.stage = "asm-emission";
        out_diagnostic.message = "failed to read generated assembly file";
        std::error_code ec;
        std::filesystem::remove(ir_path, ec);
        std::filesystem::remove(asm_path, ec);
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(ir_path, ec);
    std::filesystem::remove(asm_path, ec);
    out_diagnostic = BackendDiagnostic{};
    return true;
}

} // namespace tc::backend
