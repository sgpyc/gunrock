// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csc.cuh
 *
 * @brief CSC (Compressed Sparse Column) Graph Data Structure
 */

#pragma once

namespace gunrock {

/**
 * @brief CSR data structure which uses Compressed Sparse Row
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier.
 * @tparam ValueTAssociated value type.
 * @tparam SizeT Graph size type.
 */
template <typename _VertexT, typename _SizeT, typename _ValueT>
struct Csc
{
    typedef _VertexT  VertexT;
    typedef _SizeT    SizeT;
    typedef _ValueT   ValueT;

    typedef Csc<VertexT, SizeT, ValueT> CscT;
    SizeT nodes;            // Number of nodes in the graph
    SizeT edges;            // Number of edges in the graph
    SizeT out_nodes;        // Number of nodes which have outgoing edges
    SizeT average_degree;   // Average vertex degrees

    VertexT *row_indices;    // Row indices corresponding to all the
    // non-zero values in the sparse matrix
    SizeT    *column_offsets; // List of indices where each column of the
    // sparse matrix starts
    ValueT   *edge_values;    // List of values attached to edges in the graph
    ValueT   *node_values;    // List of values attached to nodes in the graph

    ValueT average_edge_value;
    ValueT average_node_value;
    bool  pinned;  // Whether to use pinned memory

    /**
     * @brief CSC Constructor
     *
     * @param[in] pinned Use pinned memory for CSC data structure
     * (default: do not use pinned memory)
     */
    Csc(bool _pinned = false) :
        nodes              (0),
        edges              (0),
        average_degree     (0),
        average_edge_value (0),
        average_node_value (0),
        out_nodes          (-1),
        column_offsets     ((SizeT  *)NULL),
        row_indices        ((VertexT*)NULL),
        edge_values        ((ValueT *)NULL),
        node_values        ((ValueT *)NULL),
        pinned             (_pinned)
    {
    }

    template <bool HAS_EDGE_VALUES = false, bool HAS_NODE_VALUES = false>
    cudaError_t FromCsc(CscT &source)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = Release()) return retval;

        nodes              = source.nodes;
        edges              = source.edges;
        average_degree     = source.average_degree;
        average_edge_value = source.average_edge_value;
        average_node_value = source.average_node_value;
        out_nodes          = source.out_nodes;
        pinned             = source.pinned;

        if (retval = util::MemcpyPinned(
            column_offsets   , source.column_offsets, nodes+1, pinned))
            return retval;

        if (retval = util::MemcpyPinned(
            row_indices, source.row_indices, edges, pinned))
            return retval;

        if (HAS_EDGE_VALUES)
        if (retval = util::MemcpyPinned(
            edge_values   , source.edge_values   , edges, pinned))
            return retval;

        if (HAS_NODE_VALUES)
        if (retval = util::MemcpyPinned(
            node_values   , source.node_values   , nodes, pinned))
            return retval;

        return retval;
    }

    template <bool HAS_EDGE_VALUES, bool HAS_NODE_VALUES, typename CsrT>
    cudaError_t FromCsr(CsrT &source)
        //Csr<VertexT, SizeT, ValueT> &source)
    {
        cudaError_t retval = cudaSuccess;

        Coo<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> coo;
        average_degree     = source.average_degree;
        average_edge_value = source.average_edge_value;
        average_node_value = source.average_node_value;
        out_nodes          = source.out_nodes;
        pinned             = source.pinned;

        if (retval = coo.FromCsr(source))
            return retval;

        if (retval = FromCoo(coo))
            return retval;

        if (HAS_NODE_VALUES && source.node_values == NULL)
        {
            if (retval = util::MallocPinned(node_values, source.nodes, pinned))
                return retval;
            for (VertexT v = 0; v<source.nodes; v++)
                node_values[v] = 0;
        }

        return retval;
    }

    /**
     * @brief Build CSC graph from COO graph, sorted or unsorted
     *
     * @param[in] ordered_rows Are the rows sorted? If not, sort them.
     * @param[in] quiet Don't print out anything.
     *
     * Default: Assume rows are not sorted.
     */
    template <bool HAS_EDGE_VALUES, bool HAS_NODE_VALUES, typename CooT>
    cudaError_t FromCoo(
        CooT  &coo,
        bool  ordered_rows = false,
        bool  quiet = false)
    {
        typedef typename CooT::EdgeT EdgeT;
        cudaError_t retval = cudaSuccess;

        if (!quiet)
        {
            printf("  Converting %lld vertices, %lld directed edges (%s) "
                "from COO to CSC format...",
                (long long)coo.nodes, (long long)coo.edges,
                ordered_rows ? "ordered" : "unordered");
            fflush(stdout);
        }

        time_t mark1 = time(NULL);

        if (retval = FromScratch<HAS_EDGE_VALUES, HAS_NODE_VALUES>(
                coo.nodes, coo.edges))
            return retval;

        // Sort COO by row
        if (!ordered_rows)
        {
            util::omp_sort(coo.coo_edges, coo.edges, ColumnFirstTupleCompare<EdgeT>);
        }

        SizeT edge_offsets[129];
        SizeT edge_counts [129];
        #pragma omp parallel
        {
            int num_threads  = omp_get_num_threads();
            int thread_num   = omp_get_thread_num();
            SizeT edge_start = (long long)(coo.edges) * thread_num / num_threads;
            SizeT edge_end   = (long long)(coo.edges) * (thread_num + 1) / num_threads;
            SizeT node_start = (long long)(coo.nodes) * thread_num / num_threads;
            SizeT node_end   = (long long)(coo.nodes) * (thread_num + 1) / num_threads;
            EdgeT *new_coo   = (EdgeT*) malloc (sizeof(EdgeT) * (edge_end - edge_start));
            SizeT edge       = edge_start;
            SizeT new_edge   = 0;
            for (edge = edge_start; edge < edge_end; edge++)
            {
                if ((coo.coo_edges[edge].col != coo.coo_edges[edge].row) &&
                    ((edge == 0) ||
                     (coo.coo_edges[edge] != coo.coo_edges[edge-1])))
                {
                    new_coo[new_edge] = coo.coo_edges[edge];
                    new_edge ++;
                }
            }
            edge_counts[thread_num] = new_edge;
            for (VertexT node = node_start; node < node_end; node++)
                column_offsets[node] = -1;

            #pragma omp barrier
            #pragma omp single
            {
                edge_offsets[0] = 0;
                for (int i = 0; i < num_threads; i++)
                    edge_offsets[i + 1] = edge_offsets[i] + edge_counts[i];
                column_offsets[0] = 0;
            }

            SizeT edge_offset = edge_offsets[thread_num];
            VertexT first_col = new_edge > 0 ? new_coo[0].col : -1;
            SizeT pointer = -1;
            for (edge = 0; edge < new_edge; edge++)
            {
                SizeT edge_  = edge + edge_offset;
                VertexT col = new_coo[edge].col;
                column_offsets[col + 1] = edge_ + 1;
                if (col == first_col) pointer = edge_ + 1;
                // Fill in rows up to and including the current row
                //for (VertexT row = prev_row + 1; row <= current_row; row++) {
                //    row_offsets[row] = edge;
                //}
                //prev_row = current_row;

                row_indices[edge + edge_offset] = new_coo[edge].row;
                if (HAS_EDGE_VALUES)
                {
                    //new_coo[edge].Val(edge_values[edge]);
                    edge_values[edge + edge_offset] = new_coo[edge]. template GetVal<ValueT>();
                }
            }
            #pragma omp barrier
            if (edge_start > 0 && coo.coo_edges[edge_start].col == coo.coo_edges[edge_start - 1].col) // same column as previous thread
                if (edge_end == coo.edges || coo.coo_edges[edge_end].col != coo.coo_edges[edge_start].col) // first column ends at this thread
                {
                    column_offsets[first_col + 1] = pointer;
                }
            #pragma omp barrier
            // Fill out any trailing edgeless nodes (and the end-of-list element)
            //for (VertexT row = prev_row + 1; row <= nodes; row++) {
            //    row_offsets[row] = real_edge;
            //}
            if (column_offsets[node_start] == -1)
            {
                VertexT i = node_start;
                while (column_offsets[i] == -1) i--;
                column_offsets[node_start] = column_offsets[i];
            }
            for (VertexT node = node_start + 1; node < node_end; node++)
                if (column_offsets[node] == -1)
                {
                    column_offsets[node] = column_offsets[node - 1];
                }
            if (thread_num == 0) edges = edge_offsets[num_threads];

            free(new_coo); new_coo = NULL;
        }

        column_offsets[nodes] = edges;

        time_t mark2 = time(NULL);
        if (!quiet)
        {
            printf("Done (%ds).\n", (int)(mark2 - mark1));
        }

        if (HAS_NODE_VALUES)
        {
            if (coo.node_values == NULL)
            {
                for (SizeT node = 0; node < nodes; node++)
                {
                    node_values[node] = 0;
                }
            } else {
                memcpy(node_values, coo.node_values, sizeof(ValueT) * nodes);
            }
        }

        return retval;
    }

    /**
     * @brief Allocate memory for CSC graph.
     *
     * @tparam LOAD_EDGE_VALUES
     * @tparam LOAD_NODE_VALUES
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     */
    template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    cudaError_t FromScratch(
        SizeT nodes,
        SizeT edges)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = Release()) return retval;

        this->nodes = nodes;
        this->edges = edges;

        if (retval = util::MallocPinned(column_offsets, nodes +1, pinned))
            return retval;

        if (retval = util::MallocPinned(row_indices, edges, pinned))
            return retval;

        if (LOAD_NODE_VALUES)
            if (retval = util::MallocPinned(node_values, nodes, pinned))
                return retval;

        if (LOAD_EDGE_VALUES)
            if (retval = util::MallocPinned(edge_values, edges, pinned))
                return retval;

        return retval;
    }

    /**
     * @brief Deallocates CSC graph
     */
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        if (retval = util::FreePinned(column_offsets, pinned))
            return retval;

        if (retval = util::FreePinned(row_indices, pinned))
            return retval;

        if (retval = util::FreePinned(node_values, pinned))
            return retval;

        if (retval = util::FreePinned(edge_values, pinned))
            return retval;

        nodes = 0;
        edges = 0;
        return retval;
    }

    /**
     * @brief CSC destructor
     */
    ~Csc()
    {
        Release();
    }
}; // Csc

} // namespace gunrock
