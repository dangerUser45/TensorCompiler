#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "type_info.hpp"

namespace tc::frontend {

using TensorData = std::variant<
    std::monostate,

    std::vector<int8_t>,
    std::vector<int16_t>,
    std::vector<int32_t>,
    std::vector<int64_t>,

    std::vector<uint8_t>,
    std::vector<uint32_t>,
    std::vector<uint16_t>,
    std::vector<uint64_t>,

    std::vector<float>,
    std::vector<double>,

    std::vector<std::complex<float>>,
    std::vector<std::complex<double>>,

    std::vector<std::string>
>;

class TensorInfo
{
public:
    TensorInfo() = default;

    virtual ~TensorInfo() = default;

    TensorInfo(const TensorInfo&) = delete; // -- чтобы не было срезки
    TensorInfo& operator=(const TensorInfo&) = delete; //
    TensorInfo(TensorInfo&&) = delete;
    TensorInfo& operator=(TensorInfo&&) = delete;

    const std::string& get_name() const noexcept
    {
        return name_;
    }
    const std::vector<int64_t>& get_shape() const noexcept
    {
        return shape_;
    }

    void set_name(std::string name)
    {
        name_ = std::move(name);
    }
    void set_shape(std::vector<int64_t> shape)
    {
        shape_ = std::move(shape);
    }

    DataT get_data_type() const noexcept
    {
        return data_type_;
    }

protected:
    DataT data_type_{};

private:
    std::string name_;
    std::vector<int64_t> shape_;
};

struct Initializers final : public TensorInfo
{
public:
    template<typename T>
    const std::vector<T>& get_values() const
    {
        if (data_type_.id == DataID::UNDEFINED)
            throw std::logic_error(
                "Type hasn't been defined yet. Use \"set_values()\"");
        return std::get<std::vector<T>>(values_);
    }

    template<typename T>
    void set_values(std::vector<T> vec)
    {
        DataT type = TypeInfo<T>::type;

        // Если тип уже зафиксирован, запрещаем менять его на другой.
        if (data_type_.id != DataID::UNDEFINED && data_type_.id != type.id) {
            throw std::logic_error("set_values: dtype change is forbidden");
        }
        values_ = std::move(vec);
        data_type_ = type;
    }

private:
    TensorData values_ = std::monostate{};
};

class Attribute final
{
public:
    template<typename T>
    const std::vector<T>& get_values() const
    {
        if (data_type_.id == DataID::UNDEFINED)
            throw std::logic_error(
                "Type hasn't been defined yet. Use \"set_values()\"");
        return std::get<std::vector<T>>(values_);
    }

    const std::string& get_name() const noexcept
    {
        return name_;
    }
    DataT get_data_type() const noexcept
    {
        return data_type_;
    }

    template<typename T>
    void set_values(std::vector<T> vec)
    {
        DataT type = TypeInfo<T>::type;

        // Если тип уже зафиксирован, запрещаем менять его на другой.
        if (data_type_.id != DataID::UNDEFINED && data_type_.id != type.id) {
            throw std::logic_error("set_values: dtype change is forbidden");
        }
        values_ = std::move(vec);
        data_type_ = type;
    }

    void set_name(std::string name)
    {
        name_ = std::move(name);
    }

private:
    std::string name_;
    DataT data_type_{};
    TensorData values_ = std::monostate{};
};

class Node final
{
public:
    using AttrVecT = std::vector<std::unique_ptr<Attribute>>;

    // NOTE пока NODE - final, accept не нужен
    //  virtual void accept(Visitor& v) = 0;

    const std::string& get_name_node() const noexcept
    {
        return name_node_;
    }
    const std::string& get_name_op() const noexcept
    {
        return name_op_;
    }
    const std::vector<std::string>& get_inputs() const noexcept
    {
        return inputs_;
    }
    const std::vector<std::string>& get_outputs() const noexcept
    {
        return outputs_;
    }
    const AttrVecT& get_attrs() const noexcept
    {
        return attr_;
    }

    void set_name_node(std::string name_node)
    {
        name_node_ = std::move(name_node);
    }
    void set_name_op(std::string name_op)
    {
        name_op_ = std::move(name_op);
    }
    void set_inputs(std::vector<std::string> inputs)
    {
        inputs_ = std::move(inputs);
    }
    void set_outputs(std::vector<std::string> outputs)
    {
        outputs_ = std::move(outputs);
    }
    void set_attr(AttrVecT&& attr)
    {
        attr_ = std::move(attr);
    }

private:
    std::string name_node_;
    std::string name_op_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;

    AttrVecT attr_;
};

class Graph final
{
public:
    Graph() = default;
    ~Graph() = default;

    Graph(const Graph&) = delete;                 // copy forbidden
    Graph& operator=(const Graph&) = delete;      // because use unique_ptr
    Graph(Graph&&) noexcept = default;            // move is allowed
    Graph& operator=(Graph&&) noexcept = default; //

    using NodeVecT = std::vector<std::unique_ptr<Node>>;
    using InitVecT = std::vector<std::unique_ptr<Initializers>>;
    using TensVecT = std::vector<std::unique_ptr<TensorInfo>>;

    const std::string& get_name() const noexcept
    {
        return name_;
    }
    const NodeVecT& get_nodes() const noexcept
    {
        return nodes_;
    }
    const InitVecT& get_inits() const noexcept
    {
        return inits_;
    }
    const TensVecT& get_input_tensors() const noexcept
    {
        return input_tensor_vec_;
    }
    const TensVecT& get_output_tensors() const noexcept
    {
        return output_tensor_vec_;
    }

    void set_name(std::string new_name)
    {
        name_ = std::move(new_name);
    }
    void set_nodes(NodeVecT&& nodes)
    {
        nodes_ = std::move(nodes);
    }
    void set_inits(InitVecT&& inits)
    {
        inits_ = std::move(inits);
    }
    void set_input_tensors(TensVecT&& input_tensor)
    {
        input_tensor_vec_ = std::move(input_tensor);
    }
    void set_output_tensors(TensVecT&& output_tensor)
    {
        output_tensor_vec_ = std::move(output_tensor);
    }

private:
    std::string name_;

    NodeVecT nodes_{};
    InitVecT inits_{};

    TensVecT input_tensor_vec_{};
    TensVecT output_tensor_vec_{};
};
} // namespace tc::frontend
