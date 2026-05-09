#include "frontend_mlir.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <numeric>
#include <regex>
#include <sstream>
#include <string_view>

namespace {

std::string Trim(std::string_view text)
{
    std::size_t begin = 0;
    while (begin < text.size() &&
           std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> SplitCommaList(std::string_view text)
{
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t sep = text.find(',', start);
        const std::size_t end =
            sep == std::string_view::npos ? text.size() : sep;
        parts.push_back(Trim(text.substr(start, end - start)));
        if (sep == std::string_view::npos) {
            break;
        }
        start = sep + 1;
    }
    return parts;
}

bool ParseIntList(std::string_view text, std::vector<int64_t>& out_values)
{
    out_values.clear();
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) {
        return true;
    }

    for (const std::string& part : SplitCommaList(trimmed)) {
        if (part.empty()) {
            return false;
        }
        char* end = nullptr;
        const long long value = std::strtoll(part.c_str(), &end, 10);
        if (end == part.c_str() || *end != '\0') {
            return false;
        }
        out_values.push_back(static_cast<int64_t>(value));
    }
    return true;
}

bool ParseFloatList(std::string_view text,
                    std::vector<float>& out_values,
                    bool& out_is_splat)
{
    out_values.clear();
    out_is_splat = false;

    std::string body = Trim(text);
    if (body.size() >= 2 && body.front() == '[' && body.back() == ']') {
        body = Trim(std::string_view(body).substr(1, body.size() - 2));
    } else {
        out_is_splat = true;
    }

    for (const std::string& part : SplitCommaList(body)) {
        if (part.empty()) {
            return false;
        }
        char* end = nullptr;
        const float value = std::strtof(part.c_str(), &end);
        if (end == part.c_str() || *end != '\0') {
            return false;
        }
        out_values.push_back(value);
    }
    return !out_values.empty();
}

bool ParseTensorType(std::string_view text, tc::backend::TensorType& out_type)
{
    constexpr std::string_view kPrefix = "tensor<";
    const std::string trimmed = Trim(text);
    if (trimmed.rfind(kPrefix, 0) != 0 || trimmed.back() != '>') {
        return false;
    }

    const std::string body =
        trimmed.substr(kPrefix.size(), trimmed.size() - kPrefix.size() - 1);
    const std::size_t last_x = body.rfind('x');
    if (last_x == std::string::npos) {
        out_type.shape.clear();
        out_type.element_type = body;
        return !out_type.element_type.empty();
    }

    out_type.element_type = body.substr(last_x + 1);
    if (out_type.element_type.empty()) {
        return false;
    }

    std::vector<int64_t> shape;
    const std::string dims = body.substr(0, last_x);
    std::size_t start = 0;
    while (start <= dims.size()) {
        const std::size_t sep = dims.find('x', start);
        const std::size_t end = sep == std::string::npos ? dims.size() : sep;
        const std::string part = dims.substr(start, end - start);
        if (part.empty()) {
            return false;
        }
        char* parse_end = nullptr;
        const long long dim = std::strtoll(part.c_str(), &parse_end, 10);
        if (parse_end == part.c_str() || *parse_end != '\0') {
            return false;
        }
        shape.push_back(static_cast<int64_t>(dim));
        if (sep == std::string::npos) {
            break;
        }
        start = sep + 1;
    }
    out_type.shape = std::move(shape);
    return true;
}

bool ParseValueDecl(std::string_view text,
                    tc::backend::MlirValueDecl& out_value)
{
    const std::string trimmed = Trim(text);
    const std::size_t colon = trimmed.find(':');
    if (colon == std::string::npos) {
        return false;
    }

    out_value.name = Trim(std::string_view(trimmed).substr(0, colon));
    return !out_value.name.empty() &&
           ParseTensorType(std::string_view(trimmed).substr(colon + 1),
                           out_value.type);
}

void SetParseError(tc::backend::BackendDiagnostic& diagnostic,
                   const std::string& source_name,
                   std::size_t line_number,
                   const std::string& message)
{
    diagnostic.stage = "mlir-parse";
    diagnostic.message = "failed to parse frontend MLIR";
    if (!source_name.empty()) {
        diagnostic.message += " from '" + source_name + "'";
    }
    diagnostic.message +=
        " at line " + std::to_string(line_number) + ": " + message;
}

bool ParseFunctionLine(const std::string& line,
                       tc::backend::FrontendMlirModule& module)
{
    constexpr std::string_view kPrefix = "func.func @";
    if (line.rfind(kPrefix, 0) != 0) {
        return false;
    }

    const std::size_t open_paren = line.find('(', kPrefix.size());
    const std::size_t close_paren = line.rfind(')');
    const std::size_t arrow = line.find("->", close_paren);
    const std::size_t open_brace = line.find('{', arrow);
    if (open_paren == std::string::npos || close_paren == std::string::npos ||
        arrow == std::string::npos || open_brace == std::string::npos ||
        close_paren < open_paren) {
        return false;
    }

    module.entry_name = Trim(std::string_view(line).substr(
        kPrefix.size(), open_paren - kPrefix.size()));
    if (module.entry_name.empty()) {
        return false;
    }

    const std::string output_type = Trim(
        std::string_view(line).substr(arrow + 2, open_brace - (arrow + 2)));
    if (!ParseTensorType(output_type, module.output_type)) {
        return false;
    }

    module.inputs.clear();
    const std::string args = Trim(std::string_view(line).substr(
        open_paren + 1, close_paren - open_paren - 1));
    if (args.empty()) {
        return true;
    }

    for (const std::string& part : SplitCommaList(args)) {
        tc::backend::MlirValueDecl value;
        if (!ParseValueDecl(part, value)) {
            return false;
        }
        module.inputs.push_back(std::move(value));
    }
    return true;
}

bool ParseOperationLine(const std::string& line,
                        tc::backend::FrontendMlirOp& op)
{
    std::smatch match;

    static const std::regex kConstantRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*arith\.constant\s+dense<(.+)>\s*:\s*(tensor<[^>]+>)$)");
    if (std::regex_match(line, match, kConstantRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kConstant;
        op.result = match[1].str();
        if (!ParseFloatList(
                match[2].str(), op.constant_values, op.constant_is_splat)) {
            return false;
        }
        return ParseTensorType(match[3].str(), op.result_type);
    }

    static const std::regex kEmptyRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*tensor\.empty\(\)\s*:\s*(tensor<[^>]+>)$)");
    if (std::regex_match(line, match, kEmptyRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kTensorEmpty;
        op.result = match[1].str();
        return ParseTensorType(match[2].str(), op.result_type);
    }

    static const std::regex kBroadcastRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*linalg\.broadcast\s+ins\((%[A-Za-z0-9_.$]+)\s*:\s*tensor<[^>]+>\)\s+outs\((%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)\)\s+dimensions\s*=\s*\[([^\]]*)\]$)");
    if (std::regex_match(line, match, kBroadcastRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kBroadcast;
        op.result = match[1].str();
        op.operands = { match[2].str(), match[3].str() };
        return ParseTensorType(match[4].str(), op.result_type) &&
               ParseIntList(match[5].str(), op.dimensions);
    }

    static const std::regex kBinaryRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*arith\.(addf|mulf|maximumf)\s+(%[A-Za-z0-9_.$]+),\s*(%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)$)");
    if (std::regex_match(line, match, kBinaryRegex)) {
        const std::string op_name = match[2].str();
        if (op_name == "addf") {
            op.kind = tc::backend::FrontendMlirOpKind::kAddF;
        } else if (op_name == "mulf") {
            op.kind = tc::backend::FrontendMlirOpKind::kMulF;
        } else {
            op.kind = tc::backend::FrontendMlirOpKind::kMaximumF;
        }
        op.result = match[1].str();
        op.operands = { match[3].str(), match[4].str() };
        return ParseTensorType(match[5].str(), op.result_type);
    }

    static const std::regex kMatMulRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*linalg\.matmul\s+ins\((%[A-Za-z0-9_.$]+),\s*(%[A-Za-z0-9_.$]+)\s*:\s*tensor<[^>]+>,\s*tensor<[^>]+>\)\s+outs\((%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)\)\s*->\s*tensor<[^>]+>$)");
    if (std::regex_match(line, match, kMatMulRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kMatMul;
        op.result = match[1].str();
        op.operands = { match[2].str(), match[3].str(), match[4].str() };
        return ParseTensorType(match[5].str(), op.result_type);
    }

    static const std::regex kTransposeRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*linalg\.transpose\s+ins\((%[A-Za-z0-9_.$]+)\s*:\s*tensor<[^>]+>\)\s+outs\((%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)\)\s+permutation\s*=\s*\[([^\]]*)\]$)");
    if (std::regex_match(line, match, kTransposeRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kTranspose;
        op.result = match[1].str();
        op.operands = { match[2].str(), match[3].str() };
        return ParseTensorType(match[4].str(), op.result_type) &&
               ParseIntList(match[5].str(), op.dimensions);
    }

    static const std::regex kConvRegex(
        R"(^(%[A-Za-z0-9_.$]+)\s*=\s*linalg\.conv_2d_nchw_fchw\s+ins\((%[A-Za-z0-9_.$]+),\s*(%[A-Za-z0-9_.$]+)\s*:\s*tensor<[^>]+>,\s*tensor<[^>]+>\)\s+outs\((%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)\)\s*->\s*tensor<[^>]+>$)");
    if (std::regex_match(line, match, kConvRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kConv2DNchwFchw;
        op.result = match[1].str();
        op.operands = { match[2].str(), match[3].str(), match[4].str() };
        return ParseTensorType(match[5].str(), op.result_type);
    }

    static const std::regex kReturnRegex(
        R"(^return\s+(%[A-Za-z0-9_.$]+)\s*:\s*(tensor<[^>]+>)$)");
    if (std::regex_match(line, match, kReturnRegex)) {
        op.kind = tc::backend::FrontendMlirOpKind::kReturn;
        op.operands = { match[1].str() };
        return ParseTensorType(match[2].str(), op.result_type);
    }

    return false;
}

} // namespace

namespace tc::backend {

std::size_t TensorType::element_count() const noexcept
{
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(),
                           shape.end(),
                           std::size_t{ 1 },
                           [](std::size_t acc, int64_t dim) {
                               return dim <= 0
                                          ? std::size_t{ 0 }
                                          : acc * static_cast<std::size_t>(dim);
                           });
}

std::string FormatTensorType(const TensorType& type)
{
    std::ostringstream out;
    out << "tensor<";
    for (std::size_t i = 0; i < type.shape.size(); ++i) {
        if (i != 0) {
            out << 'x';
        }
        out << type.shape[i];
    }
    if (!type.shape.empty()) {
        out << 'x';
    }
    out << type.element_type << '>';
    return out.str();
}

bool ParseFrontendMlirModule(const std::string& source,
                             const std::string& source_name,
                             FrontendMlirModule& out_module,
                             BackendDiagnostic& out_diagnostic)
{
    out_module = FrontendMlirModule{};
    out_diagnostic = BackendDiagnostic{};

    std::istringstream in(source);
    std::string line;
    std::size_t line_number = 0;
    bool saw_function = false;
    bool saw_return = false;

    while (std::getline(in, line)) {
        ++line_number;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed == "module {" || trimmed == "}" ||
            trimmed.rfind("//", 0) == 0) {
            continue;
        }

        if (trimmed.rfind("func.func", 0) == 0) {
            if (saw_function || !ParseFunctionLine(trimmed, out_module)) {
                SetParseError(out_diagnostic,
                              source_name,
                              line_number,
                              "invalid function signature");
                return false;
            }
            saw_function = true;
            continue;
        }

        if (!saw_function) {
            SetParseError(out_diagnostic,
                          source_name,
                          line_number,
                          "operation before function");
            return false;
        }

        FrontendMlirOp op;
        if (!ParseOperationLine(trimmed, op)) {
            SetParseError(out_diagnostic,
                          source_name,
                          line_number,
                          "unsupported or malformed operation: " + trimmed);
            return false;
        }
        if (op.kind == FrontendMlirOpKind::kReturn) {
            saw_return = true;
        }
        out_module.ops.push_back(std::move(op));
    }

    if (!saw_function) {
        SetParseError(out_diagnostic, source_name, 0, "missing @tc_model");
        return false;
    }
    if (out_module.entry_name != "tc_model") {
        SetParseError(
            out_diagnostic, source_name, 0, "entry function must be @tc_model");
        return false;
    }
    if (!saw_return) {
        SetParseError(out_diagnostic, source_name, 0, "missing return");
        return false;
    }
    return true;
}

} // namespace tc::backend
