#pragma once

#include <filesystem>

namespace tc::frontend {

struct DumpPaths {
  std::filesystem::path dot;
  std::filesystem::path img;
  std::filesystem::path orig_dump;
};

inline std::filesystem::path default_dump_dir() {
  // получение build директории
  const char* p = std::getenv("DUMP_DIR"); 
  if (p != nullptr && *p != '\0')
      return std::filesystem::path(p);

  std::error_code ec;
  auto current_build_dir = std::filesystem::current_path(ec);
  if (!ec) return current_build_dir / "dump";
  return std::filesystem::temp_directory_path() / "dump";
}

inline DumpPaths
make_dump_paths(std::filesystem::path base = default_dump_dir()) {
  std::filesystem::create_directories(base);
  return {
    .dot = base / ("out.dot"),
    .img = base / ("img.svg"),
    .orig_dump = base / ("orig_dump.txt")
  };
}

} // namespace tc::frontend
