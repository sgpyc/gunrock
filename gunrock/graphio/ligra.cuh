// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * ligra.cuh
 *
 * @brief Input / output of Ligra graph format
 */

#pragma once
#include <fstream>

namespace gunrock {
namespace graphio {

/*
 * @brief Write Ligra input CSR arrays into .adj file.
 * Can be easily used for python interface.
 *
 * @param[in] file_name Original graph file path and name.
 * @param[in] v Number of vertices in input graph.
 * @param[in] e Number of edges in input graph.
 * @param[in] row Row-offsets array store row pointers.
 * @param[in] col Column-indices array store destinations.
 * @param[in] edge_values Per edge weight values associated.
 * @param[in] quiet Don't print out anything.
 */
template <typename GraphT>
void WriteLigra(
    char  *file_name,
    GraphT &graph,
    bool quiet = false)
{
    typedef GraphT::VertexId VertexT;
    typedef GraphT::SizeT    SizeT;
    typedef GraphT::Value    ValueT;

    char adj_name[256];
    sprintf(adj_name, "%s.adj", file_name);
    if (!quiet)
    {
        printf("writing to ligra .adj file...");
    }

    std::ofstream fout3(adj_name);
    if (fout3.is_open())
    {
        fout3 << graph.nodes << " " << graph.nodes << " " << graph.edges << std::endl;
        for (VertexT v = 0; v < graph.nodes; ++v)
            fout3 << graph.row_offsets[v] << std::endl;
        for (SizeT e = 0; e < graph.edges; ++e)
            fout3 << graph.column_indices[e] << std::endl;
        if (graph.edge_values != NULL)
        {
            for (SizeT e = 0; e < graph.edges; ++e)
                fout3 << graph.edge_values[e] << std::endl;
        }
        if (graph.node_values != NULL)
        {
            for (VertexT v = 0; v < graph.nodes; v++)
                fout3 << graph.node_alues[v] << std::endl;
        }
        fout3.close();
    }
    if (!quiet)
        printf("Done.\n");
}

} // namespace graphio
} // namespace gunrock
