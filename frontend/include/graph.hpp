#pragma once

#include <vector>
#include <string>
#include <memory>
#include <typeinfo>
namespace tc::frontend::graph {

struct dims_map { int dims, elems_per_dim; };
class Tensor{ 
  public:
    virtual ~Tensor() {}

    const std::string& get_name() const noexcept {return name_;}
    const std::vector<dims_map>& get_dims_data() const noexcept {return this->dims_data_;}

    void set_name(std::string name) noexcept {name_ = name;}
    void set_dims_data(std::vector<dims_map> dims_data) noexcept {dims_data_ = dims_data;}

  protected:
  
  private:
  // enum TensorDataType type_;  //TODO make different type of tensor 
  std::string name_;
  std::vector<dims_map> dims_data_;
};

template <typename dataT>
struct Initializers final : Tensor  {
  public:
    const std::string& get_name() const noexcept {return name_;}
    const std::vector<dims_map>& get_dims_map() const noexcept {return dims_data_;}
    const std::vector<dataT>& get_values() const noexcept {return values_;}
    const std::string& get_type_name() const noexcept {return type_name_;}

    void set_name (std::string name) noexcept { name_ = name; }
    void set_dims_data (std::vector<dims_map> dims_data) noexcept { dims_data_ = dims_data; }
    void set_values (std::vector<dataT> values) noexcept { values_ = values; }
    void set_type_name (std::string type_name) noexcept { type_name_ = type_name; }

  private:
    std::string name_;
    std::vector<dims_map> dims_data_;
    std::vector<dataT> values_;

    std::string type_name_ = typeid(dataT).name();
};

template <typename dataAttrT>
class Attribute final {
  public:
    const std::string& get_name() const noexcept { name_; }
    const std::vector<dataAttrT>& get_values() { values_; }

    void set_name(std::string name) noexcept {name_ = name;}
    void set_values(std::vector<dataAttrT> values) noexcept {values_ = values;}

  private:
    std::string name_;
    std::vector<dataAttrT> values_;
};

template <typename dataAttrT>
class Node final {
  public:
    using AttrVecT = std::vector<std::unique_ptr<Attribute<dataAttrT>>>;
    
    const std::string& get_name_node() const noexcept { return name_node_; }
    const std::string& get_name_op() const noexcept { return name_op_; }
    const std::vector<std::string>& get_inputs() const noexcept { return inputs_; }
    const std::vector<std::string>& get_outputs() const noexcept { return outputs_; }
    const AttrVecT& get_attrs() const noexcept { return attr_; }
    
    void set_name_node(std::string name_node) noexcept { name_node_ = name_node; }
    void set_name_op(std::string name_op) noexcept { name_op_ = name_op; }
    void set_inputs(std::vector<std::string> inputs) noexcept { inputs_ = inputs; }
    void set_outputs(std::vector<std::string> outputs) noexcept { outputs_ = outputs; }
    void set_attr(AttrVecT attr) noexcept { attr_ = attr; }

  private:
    std::string name_node_;
    std::string name_op_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    
    AttrVecT attr_;
};

template <typename dataAttrT, typename dataT>
class Graph final {
public:
  Graph() = default;
  ~Graph() = default;

  Graph(const Graph&) = delete;                  // copy forbidden 
  Graph& operator=(const Graph&) = delete;       // because use unique_ptr
  Graph(Graph&&) noexcept = default;             // move is allowed
  Graph& operator=(Graph&&) noexcept = default;  //
  
  using NodeVecT = std::vector<std::unique_ptr<Node<dataAttrT>>>;
  using InitVecT = std::vector<std::unique_ptr<Initializers<dataT>>>;
  
  const std::string& get_name () noexcept { return name_; }
  const NodeVecT& get_nodes() noexcept { return nodes_; }
  const NodeVecT& get_inits() noexcept { return inits_; }
  const Tensor& get_input_tensor() noexcept { return input_tensor_; }
  const Tensor& get_output_tensor() noexcept { return output_tensor_; }

  void set_name(std::string new_name) noexcept {name_ = new_name;}
  void set_nodes(NodeVecT nodes) noexcept { nodes_ = nodes; }
  void set_inits(InitVecT inits) noexcept { inits_ = inits; }
  void set_input_tensor(Tensor input_tensor) noexcept { input_tensor_ = input_tensor; }
  void set_output_tensor(Tensor output_tensor) noexcept { output_tensor_ = output_tensor; }
  
private:
  std::string name_;

  NodeVecT nodes_ {};
  InitVecT inits_ {};

  Tensor input_tensor_ {};
  Tensor output_tensor_ {};
};
} //namespace tc::frontend::graph
