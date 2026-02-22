#pragma once

#include <cstdint>
#include <complex>
#include <string>

namespace tc::frontend {

enum class DataID {
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

struct DataT {
  DataID id = DataID::UNDEFINED;
  std::string data_type_str = "UNDEFINED";
};

template<class T> struct TypeInfo;

template<> struct TypeInfo<int8_t>  { inline static const DataT type = {DataID::INT8, "INT8"}; };
template<> struct TypeInfo<int16_t> { inline static const DataT type = {DataID::INT16, "INT16"}; };
template<> struct TypeInfo<int32_t> { inline static const DataT type = {DataID::INT32, "INT32"}; };
template<> struct TypeInfo<int64_t> { inline static const DataT type = {DataID::INT64, "INT64"}; };

template<> struct TypeInfo<uint64_t> { inline static const DataT type = {DataID::UNSIGNED_INT64, "UNSIGNED_INT64"}; };
template<> struct TypeInfo<uint32_t> { inline static const DataT type = {DataID::UNSIGNED_INT32, "UNSIGNED_INT32"}; };
template<> struct TypeInfo<uint16_t> { inline static const DataT type = {DataID::UNSIGNED_INT16, "UNSIGNED_INT16"}; };
template<> struct TypeInfo<uint8_t>  { inline static const DataT type = {DataID::UNSIGNED_INT8, "UNSIGNED_INT8"}; };

template<> struct TypeInfo<float>  { inline static const DataT type = {DataID::FLOAT, "FLOAT"}; };
template<> struct TypeInfo<double> { inline static const DataT type = {DataID::DOUBLE, "DOUBLE"}; };

template<> struct TypeInfo<std::complex<float>>  { inline static const DataT type = {DataID::COMPLEX64, "COMPLEX64"}; };
template<> struct TypeInfo<std::complex<double>> { inline static const DataT type = {DataID::COMPLEX128, "COMPLEX128"}; };

template<> struct TypeInfo<std::string> { inline static const DataT type = DataT{DataID::STRING, "STRING"}; };
} //namespace tc::frontend
