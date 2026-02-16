#pragma once

#include <vector>
#include <string>
#include <typeinfo>

/*
  Class Graph:
1) std::string name
2) list<class Node> Nodes 
3) class (Tensor?) input
4) class (Tensor?) output
5) list<struct Initializers> Initializers 

  class Node:
1) std::string node_name (ex: Conv1, Conv2 etc.)

2) std::string name of operations (Conv for example)
3) list<std::string> input
4) list<std::string> output
5) list<class(or struct) attribute> attributes

  class Tensor
1) std::string name
2) 

  struct Initializers
1) std::string name
2) enum data_type
3) std::floating float_data
4) list<int> dims

  struct Attributes
1) std::string name
2) std::vector<int> ints
3) (std::string?) type
*/


namespace tensorcompiler {

enum TensorDataType {
  tensor,
};  

struct dims_map { int dims, elems_per_dim; };

class Tensor{
public:

protected:

private:
  std::string name_;
  enum TensorDataType type_;
  std::vector<dims_map> dims_data_;
};


template <typename dataT>
struct Initializers {
  std::string name_;
  std::vector<dims_map> dims_data_;
  std::vector<dataT> raw_data_;

  // const char* watch_type {stringify};
  std::string type_name_ = typeid(dataT).name();
};

template <typename dataAttrT>
class Attribute {
  public:
  
  protected:

  private:
    std::string name_;
    std::vector<dataAttrT> data_;
};

template <typename dataAttrT>
class Node {
  public:

  protected:

  private:
    std::string name_node_;
    std::string name_op_;

    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    
    std::vector<Attribute<dataAttrT>&> attr_;
};

template <typename dataAttrT, typename dataT>
class Graph {
public: 
  
protected:
  
private:
  std::string name_;

  std::vector<Node<dataAttrT>& > nodes_;
  std::vector<Initializers<dataT>&> inits_;


};


} //namespace tensorcompiler
