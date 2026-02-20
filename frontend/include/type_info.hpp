#pragma once

#include <complex>
#include <cstdint>
#include <string>

namespace tc::frontend::graph {

enum class DataT {
  UNDEFINED,

  INT8,
  INT16,
  INT32,
  INT64,
  
  UNSIGNED_INT8,
  UNSIGNED_INT16,
  UNSIGNED_INT32,
  UNSIGNED_INT64,
  
  FLOAT,
  DOUBLE,

  COMPLEX64,
  COMPLEX128,

  STRING
};

template<class T> struct TypeInfo;

template<> struct TypeInfo<int8_t>  { static constexpr DataT type = DataT::INT8; };
template<> struct TypeInfo<int16_t> { static constexpr DataT type = DataT::INT16; };
template<> struct TypeInfo<int32_t> { static constexpr DataT type = DataT::INT32; };
template<> struct TypeInfo<int64_t> { static constexpr DataT type = DataT::INT64; };

template<> struct TypeInfo<uint64_t> { static constexpr DataT type = DataT::UNSIGNED_INT64; };
template<> struct TypeInfo<uint32_t> { static constexpr DataT type = DataT::UNSIGNED_INT32; };
template<> struct TypeInfo<uint16_t> { static constexpr DataT type = DataT::UNSIGNED_INT16; };
template<> struct TypeInfo<uint8_t>  { static constexpr DataT type = DataT::UNSIGNED_INT8; };

template<> struct TypeInfo<float>  { static constexpr DataT type = DataT::FLOAT; };
template<> struct TypeInfo<double> { static constexpr DataT type = DataT::DOUBLE; };

template<> struct TypeInfo<std::complex<float>>  { static constexpr DataT type = DataT::COMPLEX64; };
template<> struct TypeInfo<std::complex<double>> { static constexpr DataT type = DataT::COMPLEX128; };

template<> struct TypeInfo<std::string> { static constexpr DataT type = DataT::STRING; };
} //namespace tc::frontend::graph
