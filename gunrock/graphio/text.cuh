// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * binary.cuh
 *
 * @brief Input / output of text graph format
 */

#pragma once

#include <fstream>
#include <iterator>

namespace gunrock {
namespace graphio {

/*
 * @brief Write human-readable CSR arrays into 3 files.
 * Can be easily used for python interface.
 *
 * @param[in] file_name Original graph file path and name.
 * @param[in] v Number of vertices in input graph.
 * @param[in] e Number of edges in input graph.
 * @param[in] row_offsets Row-offsets array store row pointers.
 * @param[in] col_indices Column-indices array store destinations.
 * @param[in] edge_values Per edge weight values associated.
 */
template <typename GraphT>
void WriteText(
    char *file_name,
    GraphT &graph,
    bool quiet = false)
{
    typedef GraphT::VertexId VertexT;
    typedef GraphT::SizeT    SizeT;
    typedef GraphT::Value    ValueT;

    if (!quiet)
        printf("  Writing graph into text format...");
    char rows_str[256], cols_str[256], edges_str[256], nodes_str[256];

    sprintf(rows_str, "%s.rows", file_name);
    sprintf(cols_str, "%s.cols", file_name);
    sprintf(edges_str, "%s.vals", file_name);
    sprintf(nodes_str, "%s.nodes", file_name);

    std::ofstream rows_output(rows_str);
    if (rows_output.is_open())
    {
        std::copy(graph.row_offsets, graph.row_offsets + graph.nodes + 1,
            std::ostream_iterator<SizeT>(rows_output, "\n"));
        rows_output.close();
    }

    std::ofstream cols_output(cols_str);
    if (cols_output.is_open())
    {
        std::copy(graph.col_indices, graph.col_indices + graph.edges,
            std::ostream_iterator<VertexT>(cols_output, "\n"));
        cols_output.close();
    }

    if (graph.edge_values != NULL)
    {
        std::ofstream edges_output(edges_str);
        if (edges_output.is_open())
        {
            std::copy(graph.edge_values, graph.edge_values + edges,
                std::ostream_iterator<ValueT>(edges_output, "\n"));
            edges_output.close();
        }
    }

    if (graph.node_values != NULL)
    {
        std::ofstream nodes_output(nodes_str);
        if (nodes_output.is_open())
        {
            std::copy(graph.node_values, graph.node_values + graph.nodes,
                std::ostream_iterator<ValueT>(nodes_output, "\n"));
            nodes_output.close();
        }
    }

    if (!quiet)
        printf("Done.\n");
}
} // namespace graphio
} // namespace gunrock
