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
 * @brief Input / output of binary compressed graph format
 */

#pragma once
#include <time.h>
#include <fstream>
#include <gunrock/util/pinned_memory.cuh>

namespace gunrock {
namespace graphio {
namespace binary  {

/**
 *
 * @brief Store graph information into a file.
 *
 * @param[in] file_name Original graph file path and name.
 * @param[in] v Number of vertices in input graph.
 * @param[in] e Number of edges in input graph.
 * @param[in] row Row-offsets array store row pointers.
 * @param[in] col Column-indices array store destinations.
 * @param[in] edge_values Per edge weight values associated.
 *
 */
template <typename GraphT>
void Write(
    const char  *file_name,
    GraphT &graph,
    bool quiet = false)
{
    typedef typename GraphT::VertexT  VertexT;
    typedef typename GraphT::SizeT    SizeT;
    typedef typename GraphT::ValueT   ValueT;

    std::ofstream fout(file_name);
    if (fout.is_open())
    {
        fout.write(reinterpret_cast<const char*>(&graph.nodes), sizeof(SizeT));
        fout.write(reinterpret_cast<const char*>(&graph.edges), sizeof(SizeT));
        fout.write(reinterpret_cast<const char*>(graph.row_offsets), (graph.nodes + 1) * sizeof(SizeT));
        fout.write(reinterpret_cast<const char*>(graph.column_indices), graph.edges * sizeof(VertexT));
        if (graph.edge_values != NULL)
        {
            fout.write(reinterpret_cast<const char*>(graph.edge_values),
                graph.edges * sizeof(ValueT));
        }

        if (graph.node_values != NULL)
        {
            fout.write(reinterpret_cast<const char*>(graph.node_values),
                graph.nodes * sizeof(ValueT));
        }
        fout.close();
    }
} // WriteBinary

template <typename GraphT>
void WriteLabel(
    char *file_name,
    GraphT &graph,
    bool quiet = false)
{
    if(!quiet)
        printf("  Writing the labels of %lld vertices to binary format...",
            (long long)graph.nodes);

    std::ofstream fout(file_name);
    if(fout.is_open())
    {
        fout.write(reinterpret_cast<const char*>(graph.node_values), graph.nodes * sizeof(GraphT::Value));
        fout.close();
    }
    if (!quiet)
        printf("Done.\n");
} // WriteBinaryLabel

/**
 * @brief Read from stored row_offsets, column_indices arrays.
 *
 * @tparam LOAD_EDGE_VALUES Whether or not to load edge values.
 *
 * @param[in] f_in Input file name.
 * @param[in] quiet Don't print out anything.
 */
template <
    bool LOAD_EDGE_VALUES = false,
    bool LOAD_NODE_VALUES = false,
    typename GraphT>
cudaError_t Read(const char *f_in, GraphT &graph, bool quiet = false)
{
    typedef typename GraphT::VertexT  VertexT;
    typedef typename GraphT::SizeT    SizeT;
    typedef typename GraphT::ValueT   ValueT;
    cudaError_t retval = cudaSuccess;

    if (!quiet)
    {
        printf("  Reading directly from stored binary CSR arrays...");
    }
    time_t mark1 = time(NULL);

    std::ifstream input(f_in);
    SizeT v, e;
    input.read(reinterpret_cast<char*>(&v), sizeof(SizeT));
    input.read(reinterpret_cast<char*>(&e), sizeof(SizeT));

    if (retval = graph. template FromScratch<LOAD_EDGE_VALUES, LOAD_NODE_VALUES>(v, e))
        return retval;

    input.read(reinterpret_cast<char*>(graph.row_offsets), (v + 1)*sizeof(SizeT));
    input.read(reinterpret_cast<char*>(graph.column_indices), e * sizeof(VertexT));
    if (LOAD_EDGE_VALUES)
    {
        input.read(reinterpret_cast<char*>(graph.edge_values), e * sizeof(ValueT));
    }

    if (LOAD_NODE_VALUES)
    {
        input.read(reinterpret_cast<char*>(graph.node_values), v * sizeof(ValueT));
    }

    time_t mark2 = time(NULL);
    if (!quiet)
    {
        printf("Done (%ds).\n", (int) (mark2 - mark1));
    }

    graph.GetOutNodes();
    return retval;
} // ReadBinary

/**
 * @brief Read from stored labels.
 *
 * @param[in] f_label Input label file name.
 * @param[in] quiet Don't print out anything.
 */
template <typename GraphT>
cudaError_t ReadLabel(
    char *f_label,
    GraphT &graph,
    bool quiet = false)
{
    cudaError_t retval = cudaSuccess;

    if (!quiet)
    {
        printf("  Reading directly from stored binary label arrays ...\n");
    }
    time_t mark1 = time(NULL);

    if (retval = util::MallocPinned(graph.node_values, graph.nodes, graph.pinned))
        return retval;

    std::ifstream input_label(f_label);
    input_label.read(reinterpret_cast<char*>(graph.node_values), graph.nodes * sizeof(GraphT::Value));

    for (typename GraphT::VertexT v=0; v<graph.nodes; v++)
        printf("%lld ", (long long)graph.node_values[v]); printf("\n");

    time_t mark2 = time(NULL);
    if (!quiet)
    {
        printf("Done reading (%ds).\n", (int) (mark2 - mark1));
    }
    return retval;
}

template <typename InfoT>
struct StoreInfo
{
    static void StoreI(InfoT *info, std::string filename)
    {
        info->info["dataset"] = filename;
    }
};

template <>
struct StoreInfo<util::NullType>
{
    static void StoreI(util::NullType *info, std::string filename)
    {

    }
};

template <
    bool HAS_EDGE_VALUES,
    bool HAS_NODE_VALUES,
    typename CsrT,
    typename InfoT>
cudaError_t Load(
    util::CommandLineArgs &args,
    CsrT &csr,
    InfoT *info)
{
    cudaError_t retval = cudaSuccess;
    const char *filename = args.GetCmdLineArgument<std::string>("graph-file").c_str();
    bool quiet = args.CheckCmdLineFlag("quiet");

    StoreInfo<InfoT>::StoreI(info, std::string(filename));
    if (retval = Read<HAS_EDGE_VALUES, HAS_NODE_VALUES, CsrT>(
        filename, csr, quiet))
        return retval;

    return retval;
}

}// namespace binary
}// namespace graphio
}// namespace gunrock
