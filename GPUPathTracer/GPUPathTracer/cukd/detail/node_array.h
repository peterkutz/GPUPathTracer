// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_NODE_ARRAY_H
#define CUKD_NODE_ARRAY_H

#include "dev_structs.h"
#include "../utils.h"

namespace cukd {

class NodeArray {
    public:
        NodeArray() : n_nodes_(0), n_elements_(0) {};

        void resize_nodes(int nodes);
        void resize_elements(int elements);

        int n_nodes() const { return n_nodes_; };
        int n_elements() const { return element_idx.size(); };

        void clear();

        device::NodeArray dev_array();

    protected:
        int n_nodes_;
        int n_elements_;
        DevVector<int> split_axis;
        DevVector<float> split_position;
        DevVector<int> left_nodes, right_nodes;
        DevVector<int> depth;
        DevVector<int> node_size;
        DevVector<int> element_idx;
        DevVector<int> node_element_first_idx;
};

inline
void
NodeArray::clear() {
    n_nodes_ = 0;
    n_elements_ = 0;
    split_axis.clear();
    split_position.clear();
    left_nodes.clear();
    right_nodes.clear();
    depth.clear();
    node_size.clear();
    element_idx.clear();
    node_element_first_idx.clear();
};

inline
void
NodeArray::resize_nodes(int nodes) {
    n_nodes_ = nodes;
    split_axis.resize(nodes);
    split_position.resize(nodes);
    left_nodes.resize(nodes);
    right_nodes.resize(nodes);
    depth.resize(nodes);
    node_size.resize(nodes);
    node_element_first_idx.resize(nodes);
};

inline
void
NodeArray::resize_elements(int elements) {
    element_idx.resize(elements);
    n_elements_ = elements;
};

inline
device::NodeArray
NodeArray::dev_array() {
    device::NodeArray devarray;
    devarray.split_axis             = split_axis.pointer();
    devarray.split_position         = split_position.pointer();
    devarray.left_nodes             = left_nodes.pointer();
    devarray.right_nodes            = right_nodes.pointer();
    devarray.depth                  = depth.pointer();
    devarray.node_size              = node_size.pointer();
    devarray.element_idx            = element_idx.pointer();
    devarray.node_element_first_idx = node_element_first_idx.pointer();
    return devarray;
}

}  // namespace cukd

#endif  // CUKD_NODE_ARRAY_H
