#include <cstdio>
#include <cassert>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <fstream>


namespace topology {

namespace {

auto vertex2edge(int vertex_count, int edge_count, const int *edges) -> std::tuple<std::vector<int>, std::vector<int>>
{
    std::vector<std::vector<int>> v2e_vec(vertex_count);

    for (int i = 0; i < edge_count; ++i) {
        const auto ip0 = edges[i * 2 + 0];
        const auto ip1 = edges[i * 2 + 1];
        v2e_vec[ip0].push_back(i);
        v2e_vec[ip1].push_back(i);
    }

    std::vector<int> v2e, v2e_slice;
    v2e.reserve(edge_count * 2);
    v2e_slice.reserve(vertex_count * 2);

    for (int i = 0; i < vertex_count; ++i) {
        v2e_slice.push_back(static_cast<int>(v2e.size()));
        for (const auto e : v2e_vec[i])
            v2e.push_back(e);
        v2e_slice.push_back(static_cast<int>(v2e.size()));
    }

    assert(v2e.size() == edge_count * 2);
    assert(v2e_slice.size() == vertex_count * 2);

    return std::make_tuple(v2e, v2e_slice);
}

void extend_segment(int v, const std::vector<int> &v2e, const std::vector<int> &v2e_slice,
                    const int *edges, std::unordered_set<int> &uncovered_e, std::vector<int> &segment)
{
    int v_pre;

    do {
        const auto begin = v2e_slice[v * 2 + 0];
        const auto end   = v2e_slice[v * 2 + 1];
        v_pre = v;

        for (int i = begin; i < end; ++i) {
            const auto e = v2e[i];

            if (uncovered_e.erase(e) > 0) {
                const auto ip0 = edges[e * 2 + 0];
                const auto ip1 = edges[e * 2 + 1];
                const auto ip = v != ip0 ? ip0 : ip1;

                if (ip != v) {
                    v = ip;
                    segment.push_back(v);
                    break;
                }
            }
        }
    } while (v_pre != v);
}

void maximal_segments(int vertex_count, int edge_count, const int *edges,
                      int *vertex_indices, int *segment_indices)
{
    const auto [v2e, v2e_slice] = vertex2edge(vertex_count, edge_count, edges);

    std::unordered_set<int> uncovered_e;
    {
        std::vector<int> e(edge_count);
        std::iota(e.begin(), e.end(), 0);
        uncovered_e.insert(e.cbegin(), e.cend());
    }

    std::vector<int> prefix;
    std::vector<int> suffix;
    prefix.reserve(edge_count + 1);
    suffix.reserve(edge_count + 1);

    auto vertex_size = 0;
    auto segment_size = 0;

    while (!uncovered_e.empty()) {
        const auto seed_e = *uncovered_e.cbegin();
        uncovered_e.erase(seed_e);

        prefix.clear();
        suffix.clear();

        prefix.push_back(edges[seed_e * 2 + 0]);
        prefix.push_back(edges[seed_e * 2 + 1]);

        extend_segment(edges[seed_e * 2 + 1], v2e, v2e_slice, edges, uncovered_e, prefix);
        extend_segment(edges[seed_e * 2 + 0], v2e, v2e_slice, edges, uncovered_e, suffix);

        segment_indices[segment_size] = vertex_size;
        segment_size += 1;

        std::copy(prefix.crbegin(), prefix.crend(), vertex_indices + vertex_size);
        vertex_size += static_cast<int>(prefix.size());

        std::copy(suffix.cbegin(), suffix.cend(), vertex_indices + vertex_size);
        vertex_size += static_cast<int>(suffix.size());

        assert(vertex_size <= edge_count * 2);
        assert(segment_size <= edge_count);
    }
}

} // namespace

} // namespace topology



#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


namespace nb = nanobind;
using namespace nb::literals;

template<typename T> using TensorX = nb::ndarray<nb::numpy, T, nb::shape<-1>, nb::c_contig>;
template<typename T, long dim1> using TensorXN = nb::ndarray<nb::numpy, T, nb::shape<-1, dim1>, nb::c_contig>;

#define CHECK_TENSOR_SIZE(tensor, dim, size) if (tensor.shape(dim) != size) throw std::runtime_error(#tensor " has an invalid tensor size")

namespace {

void maximal_segments(int vertex_count, TensorXN<int, 2> edges, TensorX<int> vertex_indices, TensorX<int> segment_indices)
{
    const auto edge_count = static_cast<int>(edges.shape(0));

    CHECK_TENSOR_SIZE(vertex_indices, 0, edge_count * 2);
    CHECK_TENSOR_SIZE(segment_indices, 0, edge_count);

    topology::maximal_segments(vertex_count, edge_count, edges.data(), vertex_indices.data(), segment_indices.data());
}

} // namespace


NB_MODULE(core, m)
{
    m.def("maximal_segments", &::maximal_segments);
}


