// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * coo.cuh
 *
 * @brief Coordinate Format (a.k.a. triplet format) Graph Data Structure
 */

#pragma once

#include <type_traits>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/pinned_memory.cuh>

namespace gunrock {

/**
 * @brief COO sparse format edge. (A COO graph is just a
 * list/array/vector of these.)
 *
 * @tparam VertexT Vertex identifiler type.
 * @tparam ValueT Attribute value type.
 *
 */
template <typename _VertexT, typename _ValueT>
struct CooEdge {
    typedef _VertexT VertexT;
    typedef _ValueT  ValueT;
    typedef CooEdge<VertexT, ValueT> EdgeT;

    VertexT row;
    VertexT col;
    ValueT val;

    CooEdge() {}
    CooEdge(VertexT row, VertexT col, ValueT val) :
        row(row), col(col), val(val) {}

    template <typename T>
    __device__ __host__ __forceinline__
    void SetVal(T value) {
        val = value;
    }

    template <typename T>
    __device__ __host__ __forceinline__
    T GetVal() {
        return val;
    }

    bool operator==(const EdgeT &rhs)
    {
        if (row != rhs.row) return false;
        if (col != rhs.col) return false;
        return true;
    }

    bool operator!=(const EdgeT &rhs)
    {
        return (!(*this == rhs));
    }
};


/*
 * @brief Coo data structure.
 *
 * @tparam VertexT Vertex identifier type.
 */
template<typename VertexT>
struct CooEdge<VertexT, util::NullType>
{
    typedef CooEdge<VertexT, util::NullType> EdgeT;

    VertexT row;
    VertexT col;

    CooEdge() {}

    template <typename T>
    CooEdge(VertexT row, VertexT col, T val) :
        row(row), col(col) {}

    template <typename T>
    __device__ __host__ __forceinline__
    void SetVal(T value) {}

    template <typename T>
    __device__ __host__ __forceinline__
    T GetVal() {return 0;}

    bool operator==(const EdgeT &rhs)
    {
        if (row != rhs.row) return false;
        if (col != rhs.col) return false;
        return true;
    }

    bool operator!=(const EdgeT &rhs)
    {
        return (!(*this == rhs));
    }
};


/**
 * @brief Comparator for sorting COO sparse format edges first by row
 *
 * @tparam CooEdge COO Datatype
 *
 * @param[in] elem1 First element to compare
 * @param[in] elem2 Second element to compare
 * @returns true if first element comes before second element in (r,c)
 * order, otherwise false
 *
 * @see ColumnFirstTupleCompare
 */
template <typename EdgeT>
bool RowFirstTupleCompare (
    EdgeT elem1,
    EdgeT elem2) {
    if (elem1.row < elem2.row) {
        // Sort edges by source node
        return true;
    } else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
        // Sort edgelists as well for coherence
        return true;
    }

    return false;
}

/**
 * @brief Comparator for sorting COO sparse format edges first by column
 *
 * @tparam Coo COO Datatype
 *
 * @param[in] elem1 First element to compare
 * @param[in] elem2 Second element to compare
 * @returns true if first element comes before second element in (c,r)
 * order, otherwise false
 *
 * @see RowFirstTupleCompare
 */
template<typename EdgeT>
bool ColumnFirstTupleCompare (
    EdgeT elem1,
    EdgeT elem2) {
    if (elem1.col < elem2.col) {
        // Sort edges by source node
        return true;
    } else if ((elem1.col == elem2.col) && (elem1.row < elem2.row)) {
        // Sort edgelists as well for coherence
        return true;
    }

    return false;
}


/**
 * @brief Coo data structure to present
 * the graph as aedge list.
 *
 * @tparam VertexT Vertex identifier.
 * @tparam ValueT Associated value type.
 * @tparam SizeT Graph size type.
 */

template <
    typename _VertexT,
    typename _SizeT,
    typename _ValueT,
    bool _HAS_EDGE_VALUES = false>
struct Coo
{
    typedef _VertexT  VertexT;
    typedef _SizeT    SizeT;
    typedef _ValueT   ValueT;

    static const bool HAS_EDGE_VALUES = _HAS_EDGE_VALUES;

    SizeT nodes;            // Number of nodes in the graph
    SizeT edges;            // Number of edges in the graph
    bool  pinned;           // Whether to use pinned host memory

    // select edge type based on whether has edge value
    typedef typename std::conditional<HAS_EDGE_VALUES,
        CooEdge<VertexT, ValueT>, CooEdge<VertexT, util::NullType> >::type EdgeT;
    EdgeT *coo_edges;   // the edge list
    ValueT *node_values; // node values

    /**
     * @brief Coo Constructor
     *
     * @param[in] pinned Use pinned memory for Coo data structure
     * (default: do not use pinned memory)
     */
    Coo(bool _pinned = false) :
        nodes (0),
        edges (0),
        coo_edges ((EdgeT*)NULL),
        node_values((ValueT*)NULL),
        pinned(_pinned)
    {}

    /**
     * @brief Allocate memory for COO graph.
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     * @param[in] whether to have node associative values
     */
    cudaError_t FromScratch(SizeT nodes, SizeT edges,
        bool has_node_values = false)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = Release()) return retval;

        if (retval = util::MallocPinned(coo_edges, edges, pinned))
            return retval;

        if (has_node_values)
            if (retval = util::MallocPinned(node_values, nodes, pinned))
                return retval;

        this -> nodes = nodes;
        this -> edges = edges;

        return retval;
    }

    template <typename CooT>
    cudaError_t FromCoo(CooT &source, bool quiet = false)
    {
        cudaError_t retval = cudaSuccess;
        if (!quiet)
        {
            printf("  Copying %lld vertices, %lld directed edges "
                "from COO to COO format...",
                (long long)source.nodes, (long long)source.edges);
            fflush(stdout);
        }
        time_t mark1 = time(NULL);

        if (retval = Release()) return retval;
        if (retval = FromScratch(source.nodes, source.edges)) return retval;
        if (source.node_values != NULL)
            if (retval = util::MemcpyPinned(node_values,
                source.node_values, source.nodes, pinned))
                return retval;

        #pragma omp parallel for
        for (SizeT e = 0; e<edges; e++)
        {
            coo_edges[e].row = source.coo_edges[e].row;
            coo_edges[e].col = source.coo_edges[e].col;
            coo_edges[e].SetVal(source.coo_edges[e]. template GetVal<ValueT>());
        }

        nodes = source.nodes;
        edges = source.edges;

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done (%ds).\n", (int)(mark2 - mark1));
        }
        return retval;
    }

    template <typename CsrT>
    cudaError_t FromCsr(CsrT &source, bool quiet = false)
    {
        cudaError_t retval = cudaSuccess;
        if (!quiet)
        {
            printf("  Converting %lld vertices, %lld directed edges "
                "from CSR to COO format...",
                (long long)source.nodes, (long long)source.edges);
            fflush(stdout);
        }
        time_t mark1 = time(NULL);

        if (retval = Release()) return retval;
        if (retval = FromScratch(source.nodes, source.edges)) return retval;
        if (source.node_values != NULL)
            if (retval = util::MemcpyPinned(node_values,
                source.node_values, source.nodes, pinned))
                return retval;

        #pragma omp parallel for
        for (VertexT u = 0; u<nodes; u++)
        {
            for (SizeT e = source.row_offsets[u]; e < source.row_offsets[u+1]; e++)
            {
                coo_edges[e].col = source.column_indices[e];
                coo_edges[e].row = u;
                coo_edges[e].SetVal((source.edge_values == NULL) ?
                    0 : source.edge_values[e]);
            }
        }

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done (%ds).\n", (int)(mark2 - mark1));
        }
        return retval;
    }

    template <typename CscT>
    cudaError_t FromCsc(CscT &source, bool quiet = false)
    {
        cudaError_t retval = cudaSuccess;
        if (!quiet)
        {
            printf("  Converting %lld vertices, %lld directed edges "
                "from CSC to COO format...",
                (long long)source.nodes, (long long)source.edges);
            fflush(stdout);
        }
        time_t mark1 = time(NULL);

        if (retval = Release()) return retval;
        if (retval = FromScratch(source.nodes, source.edges)) return retval;
        if (source.node_values != NULL)
            if (retval = util::MemcpyPinned<ValueT, typename CscT::ValueT, SizeT>(node_values,
                source.node_values, source.nodes, pinned))
                return retval;

        #pragma omp parallel for
        for (VertexT v = 0; v<nodes; v++)
        {
            for (SizeT e = source.column_offsets[v]; e < source.column_offsets[v+1]; e++)
            {
                coo_edges[e].col = v;
                coo_edges[e].row = source.row_indices[e];
                coo_edges[e].SetVal((source.edge_values == NULL) ?
                    0 : source.edge_values[e]);
            }
        }

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done (%ds).\n", (int)(mark2 - mark1));
        }
        return retval;
    }

    /**
     * @brief Release allocated memory
     */
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = util::FreePinned(coo_edges, pinned))
            return retval;

        if (retval = util::FreePinned(node_values, pinned))
            return retval;

        nodes = 0;
        edges = 0;

        return retval;
    }


    /**
     * @brief Coo destructor
     */
    ~Coo()
    {
        Release();
    }
};

} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
