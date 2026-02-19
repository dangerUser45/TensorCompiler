#pragma once

#include <vector>
#include <string>
#include <memory>
#include <typeinfo>
#include <utility>

namespace tc::frontend::graph {

struct dims_map { int dims, elems_per_dim; };
class Tensor{ 
  public:
    virtual ~Tensor() = default;

    const std::string& get_name() const noexcept { return name_; }
    const std::vector<dims_map>& get_dims_data() const noexcept { return this->dims_data_; }

    void set_name(std::string name) { name_ = std::move(name); }
    void set_dims_data(std::vector<dims_map> dims_data) { dims_data_ = std::move(dims_data); }
  
  private:
  // enum TensorDataType type_;  //NOTE may make different type of tensor 
  std::string name_;
  std::vector<dims_map> dims_data_;
};

template <typename dataT>
struct Initializers final : Tensor {
  public:
    const std::vector<dataT>& get_values() const noexcept {return values_;}
    const std::string& get_type_name() const noexcept { return type_name_; } 

    void set_values (std::vector<dataT> values) { values_ = std::move(values); }

  private:
    std::vector<dataT> values_;
    std::string type_name_ = typeid(dataT).name();
};

template <typename dataAttrT>
class Attribute final {
  public:
    const std::string& get_name() const noexcept { return name_; }
    const std::vector<dataAttrT>& get_values() const noexcept { return values_; }

    void set_name(std::string name) { name_ = std::move(name); }
    void set_values(std::vector<dataAttrT> values) { values_ = std::move(values); }

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
    
    void set_name_node(std::string name_node) { name_node_ = std::move(name_node); }
    void set_name_op(std::string name_op) { name_op_ = std::move(name_op); }
    void set_inputs(std::vector<std::string> inputs) { inputs_ = std::move(inputs); }
    void set_outputs(std::vector<std::string> outputs) { outputs_ = std::move(outputs); }
    void set_attr(AttrVecT&& attr) { attr_ = std::move(attr); }

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
  using TensVecT = std::vector<std::unique_ptr<Tensor>>; 
  
  const std::string& get_name () const noexcept { return name_; }
  const NodeVecT& get_nodes() const noexcept { return nodes_; }
  const InitVecT& get_inits() const noexcept { return inits_; }
  const TensVecT& get_input_tensors() const noexcept { return input_tensor_vec_; }
  const TensVecT& get_output_tensors() const noexcept { return output_tensor_vec_; }

  void set_name(std::string new_name)  {name_ = std::move(new_name);}
  void set_nodes(NodeVecT&& nodes)  { nodes_ = std::move(nodes); }
  void set_inits(InitVecT&& inits)  { inits_ = std::move(inits); }
  void set_input_tensors(TensVecT&& input_tensor)  { input_tensor_vec_ = std::move(input_tensor); }
  void set_output_tensors(TensVecT&& output_tensor) { output_tensor_vec_ = std::move(output_tensor); }
  
private:
  std::string name_;

  NodeVecT nodes_ {};
  InitVecT inits_ {};

  TensVecT input_tensor_vec_ {};
  TensVecT output_tensor_vec_ {};
};
} //namespace tc::frontend::graph
