#include "type_info.hpp"

#include <array>
#include <cstddef>

namespace tc::frontend {

const char* DataIDToString(DataID id) noexcept
{
    constexpr std::array<const char*, 14> kNames = {
        "UNDEFINED",      "INT8",          "INT16",          "INT32",
        "INT64",          "UNSIGNED_INT8", "UNSIGNED_INT16", "UNSIGNED_INT32",
        "UNSIGNED_INT64", "FLOAT",         "DOUBLE",         "COMPLEX64",
        "COMPLEX128",     "STRING",
    };

    const auto index = static_cast<std::size_t>(id);
    if (index < kNames.size()) {
        return kNames[index];
    }
    return "UNDEFINED";
}

} // namespace tc::frontend
