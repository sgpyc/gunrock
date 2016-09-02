// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csr.cuh
 *
 * @brief CSR (Compressed Sparse Row) Graph Data Structure
 */

#pragma once

#include <time.h>
#include <stdio.h>
//#include <string>
//#include <vector>
#include <fstream>
#include <iostream>
//#include <algorithm>
#include <omp.h>

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/pinned_memory.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/coo.cuh>

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
struct Csr
{
    typedef _VertexT  VertexT;
    typedef _SizeT    SizeT;
    typedef _ValueT   ValueT;

    typedef Csr<VertexT, SizeT, ValueT> CsrT;
    SizeT nodes;            // Number of nodes in the graph
    SizeT edges;            // Number of edges in the graph
    SizeT out_nodes;        // Number of nodes which have outgoing edges
    SizeT average_degree;   // Average vertex degrees

    VertexT *column_indices; // Column indices corresponding to all the
    // non-zero values in the sparse matrix
    SizeT    *row_offsets;    // List of indices where each row of the
    // sparse matrix starts
    ValueT    *edge_values;    // List of values attached to edges in the graph
    ValueT    *node_values;    // List of values attached to nodes in the graph

    ValueT average_edge_value;
    ValueT average_node_value;
    bool  pinned;  // Whether to use pinned memory

    /**
     * @brief CSR Constructor
     *
     * @param[in] pinned Use pinned memory for CSR data structure
     * (default: do not use pinned memory)
     */
    Csr(bool _pinned = false) :
        nodes              (0),
        edges              (0),
        average_degree     (0),
        average_edge_value (0),
        average_node_value (0),
        out_nodes          (-1),
        row_offsets        ((SizeT  *)NULL),
        column_indices     ((VertexT*)NULL),
        edge_values        ((ValueT *)NULL),
        node_values        ((ValueT *)NULL),
        pinned             (_pinned)
    {
    }

    template <bool HAS_EDGE_VALUES = false, bool HAS_NODE_VALUES = false>
    cudaError_t FromCsr(CsrT &source)
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
            row_offsets   , source.row_offsets, nodes+1, pinned))
            return retval;

        if (retval = util::MemcpyPinned(
            column_indices, source.column_indices, edges, pinned))
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

    template <bool HAS_EDGE_VALUES, bool HAS_NODE_VALUES, typename CscT>
    cudaError_t FromCsc(CscT &source)
        //Csr<VertexT, SizeT, Value> &source)
    {
        cudaError_t retval = cudaSuccess;

        Coo<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> coo;
        average_degree     = source.average_degree;
        average_edge_value = source.average_edge_value;
        average_node_value = source.average_node_value;
        out_nodes          = source.out_nodes;
        pinned             = source.pinned;

        if (retval = coo.FromCsc(source))
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
     * @brief Allocate memory for CSR graph.
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

        if (retval = util::MallocPinned(row_offsets, nodes +1, pinned))
            return retval;

        if (retval = util::MallocPinned(column_indices, edges, pinned))
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
     * @brief Build CSR graph from COO graph, sorted or unsorted
     *
     * @param[in] output_file Output file to dump the graph topology info
     * @param[in] coo Pointer to COO-format graph
     * @param[in] coo_nodes Number of nodes in COO-format graph
     * @param[in] coo_edges Number of edges in COO-format graph
     * @param[in] ordered_rows Are the rows sorted? If not, sort them.
     * @param[in] undirected Is the graph directed or not?
     * @param[in] reversed Is the graph reversed or not?
     * @param[in] quiet Don't print out anything.
     *
     * Default: Assume rows are not sorted.
     */
    template <bool HAS_EDGE_VALUES, bool HAS_NODE_VALUES, typename CooT>
    cudaError_t FromCoo(
        //char  *output_file,
        //Tuple *coo,
        //SizeT coo_nodes,
        //SizeT coo_edges,
        CooT  &coo,
        //bool  load_edge_values = false,
        bool  ordered_rows = false,
        //bool  undirected = false,
        //bool  reversed = false,
        bool  quiet = false)
    {
        typedef typename CooT::EdgeT EdgeT;
        cudaError_t retval = cudaSuccess;

        if (!quiet)
        {
            printf("  Converting %lld vertices, %lld directed edges (%s) "
                "from COO to CSR format...",
                (long long)coo.nodes, (long long)coo.edges,
                ordered_rows ? "ordered" : "unordered");
            fflush(stdout);
        }
        time_t mark1 = time(NULL);

        if (retval = Release())
            return retval;
        if (retval = FromScratch<HAS_EDGE_VALUES, HAS_NODE_VALUES>(
                coo.nodes, coo.edges))
            return retval;

        // Sort COO by row
        if (!ordered_rows)
        {
            util::omp_sort(coo.coo_edges, coo.edges, RowFirstTupleCompare<EdgeT>);
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
                row_offsets[node] = -1;

            #pragma omp barrier
            #pragma omp single
            {
                edge_offsets[0] = 0;
                for (int i = 0; i < num_threads; i++)
                    edge_offsets[i + 1] = edge_offsets[i] + edge_counts[i];
                row_offsets[0] = 0;
            }

            SizeT edge_offset = edge_offsets[thread_num];
            VertexT first_row = new_edge > 0 ? new_coo[0].row : -1;
            SizeT pointer = -1;
            for (edge = 0; edge < new_edge; edge++)
            {
                SizeT edge_  = edge + edge_offset;
                VertexT row = new_coo[edge].row;
                row_offsets[row + 1] = edge_ + 1;
                if (row == first_row) pointer = edge_ + 1;
                // Fill in rows up to and including the current row
                //for (VertexT row = prev_row + 1; row <= current_row; row++) {
                //    row_offsets[row] = edge;
                //}
                //prev_row = current_row;

                column_indices[edge + edge_offset] = new_coo[edge].col;
                if (HAS_EDGE_VALUES)
                {
                    //new_coo[edge].Val(edge_values[edge]);
                    edge_values[edge + edge_offset] = new_coo[edge]. template GetVal<ValueT>();
                }
            }
            #pragma omp barrier
            //if (first_row != last_row)
            if (edge_start > 0 && coo.coo_edges[edge_start].row == coo.coo_edges[edge_start - 1].row) // same row as previous thread
                if (edge_end == coo.edges || coo.coo_edges[edge_end].row != coo.coo_edges[edge_start].row) // first row ends at this thread
                {
                    row_offsets[first_row + 1] = pointer;
                }
            #pragma omp barrier
            // Fill out any trailing edgeless nodes (and the end-of-list element)
            //for (VertexT row = prev_row + 1; row <= nodes; row++) {
            //    row_offsets[row] = real_edge;
            //}
            if (row_offsets[node_start] == -1)
            {
                VertexT i = node_start;
                while (row_offsets[i] == -1) i--;
                row_offsets[node_start] = row_offsets[i];
            }
            for (VertexT node = node_start + 1; node < node_end; node++)
                if (row_offsets[node] == -1)
                {
                    row_offsets[node] = row_offsets[node - 1];
                }
            if (thread_num == 0) edges = edge_offsets[num_threads];

            free(new_coo); new_coo = NULL;
        }

        row_offsets[nodes] = edges;

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

        // Write offsets, indices, node, edges etc. into file
        /*if (LOAD_EDGE_VALUES)
        {
            WriteBinary(output_file, nodes, edges,
                        row_offsets, column_indices, edge_values);
            //WriteCSR(output_file, nodes, edges,
            //         row_offsets, column_indices, edge_values);
            //WriteToLigraFile(output_file, nodes, edges,
            //                 row_offsets, column_indices, edge_values);
        }
        else
        {
            WriteBinary(output_file, nodes, edges,
                        row_offsets, column_indices);
        }*/

        // Compute out_nodes
        GetOutNodes();

        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Print log-scale degree histogram of the graph.
     */
    void PrintHistogram()
    {
        fflush(stdout);

        // Initialize
        SizeT log_counts[32];
        for (int i = 0; i < 32; i++)
        {
            log_counts[i] = 0;
        }

        // Scan
        SizeT max_log_length = -1;
        for (VertexT i = 0; i < nodes; i++)
        {

            SizeT length = row_offsets[i + 1] - row_offsets[i];

            int log_length = -1;
            while (length > 0)
            {
                length >>= 1;
                log_length++;
            }
            if (log_length > max_log_length)
            {
                max_log_length = log_length;
            }

            log_counts[log_length + 1]++;
        }
        printf("\nDegree Histogram (%lld vertices, %lld edges):\n",
               (long long) nodes, (long long) edges);
        printf("    Degree   0: %lld (%.2f%%)\n",
               (long long) log_counts[0],
               (float) log_counts[0] * 100.0 / nodes);
        for (int i = 0; i < max_log_length + 1; i++)
        {
            printf("    Degree 2^%i: %lld (%.2f%%)\n",
                i, (long long)log_counts[i + 1],
                (float) log_counts[i + 1] * 100.0 / nodes);
        }
        printf("\n");
        fflush(stdout);
    }


    /**
     * @brief Display CSR graph to console
     *
     * @param[in] with_edge_value Whether display graph with edge values.
     */
    void DisplayGraph(bool with_edge_value = false)
    {
        SizeT displayed_node_num = (nodes > 40) ? 40 : nodes;
        printf("First %d nodes's neighbor list of the input graph:\n",
               displayed_node_num);
        for (SizeT node = 0; node < displayed_node_num; node++)
        {
            util::PrintValue(node);
            printf(":");
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    edge++)
            {
                if (edge - row_offsets[node] > 40) break;
                printf("[");
                util::PrintValue(column_indices[edge]);
                if (with_edge_value && edge_values != NULL)
                {
                    printf(",");
                    util::PrintValue(edge_values[edge]);
                }
                printf("], ");
            }
            printf("\n");
        }
    }

    /**
     * @brief Display CSR graph to console
     */
    void DisplayGraph(const char name[], SizeT limit = 40)
    {
        SizeT displayed_node_num = (nodes > limit) ? limit : nodes;
        printf("%s : #nodes = ", name); util::PrintValue(nodes);
        printf(", #edges = "); util::PrintValue(edges);
        printf("\n");

        for (SizeT i = 0; i < displayed_node_num; i++)
        {
            util::PrintValue(i);
            printf(",");
            util::PrintValue(row_offsets[i]);
            if (node_values != NULL)
            {
                printf(",");
                util::PrintValue(node_values[i]);
            }
            printf(" (");
            for (SizeT j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            {
                if (j != row_offsets[i]) printf(" , ");
                util::PrintValue(column_indices[j]);
                if (edge_values != NULL)
                {
                    printf(",");
                    util::PrintValue(edge_values[j]);
                }
            }
            printf(")\n");
        }

        printf("\n");
    }

    /**
     * @brief Check values.
     */
    bool CheckValue()
    {
        for (SizeT node = 0; node < nodes; ++node)
        {
            for (SizeT edge = row_offsets[node];
                    edge < row_offsets[node + 1];
                    ++edge)
            {
                int src_node = node;
                int dst_node = column_indices[edge];
                int edge_value = edge_values[edge];
                for (SizeT r_edge = row_offsets[dst_node];
                        r_edge < row_offsets[dst_node + 1];
                        ++r_edge)
                {
                    if (column_indices[r_edge] == src_node)
                    {
                        if (edge_values[r_edge] != edge_value)
                            return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Find node with largest neighbor list
     * @param[in] max_degree Maximum degree in the graph.
     *
     * \return int the source node with highest degree
     */
    int GetNodeWithHighestDegree(int& max_degree)
    {
        int degree = 0;
        int src = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > degree)
            {
                degree = row_offsets[node + 1] - row_offsets[node];
                src = node;
            }
        }
        max_degree = degree;
        return src;
    }

    /**
     * @brief Display the neighbor list of a given node.
     *
     * @param[in] node Vertex ID to display.
     */
    void DisplayNeighborList(VertexT node)
    {
        if (node < 0 || node >= nodes) return;
        for (SizeT edge = row_offsets[node];
                edge < row_offsets[node + 1];
                edge++)
        {
            util::PrintValue(column_indices[edge]);
            printf(", ");
        }
        printf("\n");
    }

    /**
     * @brief Get the average degree of all the nodes in graph
     */
    SizeT GetAverageDegree()
    {
        if (average_degree == 0)
        {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node)
            {
                count += 1;
                mean += (row_offsets[node + 1] - row_offsets[node] - mean) / count;
            }
            average_degree = static_cast<SizeT>(mean);
        }
        return average_degree;
    }

    /**
     * @brief Get the degrees of all the nodes in graph
     *
     * @param[in] node_degrees node degrees to fill in
     */
    void GetNodeDegree(SizeT *node_degrees)
    {
	    for(SizeT node=0; node < nodes; ++node)
	    {
		    node_degrees[node] = row_offsets[node+1]-row_offsets[node];
	    }
    }

    /**
     * @brief Get the average node value in graph
     */
    ValueT GetAverageNodeValue()
    {
        if (abs(average_node_value - 0) < 0.001 && node_values != NULL)
        {
            double mean = 0, count = 0;
            for (SizeT node = 0; node < nodes; ++node)
            {
                if (node_values[node] < UINT_MAX)
                {
                    count += 1;
                    mean += (node_values[node] - mean) / count;
                }
            }
            average_node_value = static_cast<ValueT>(mean);
        }
        return average_node_value;
    }

    /**
     * @brief Get the average edge value in graph
     */
    ValueT GetAverageEdgeValue()
    {
        if (abs(average_edge_value - 0) < 0.001 && edge_values != NULL)
        {
            double mean = 0, count = 0;
            for (SizeT edge = 0; edge < edges; ++edge)
            {
                if (edge_values[edge] < UINT_MAX)
                {
                    count += 1;
                    mean += (edge_values[edge] - mean) / count;
                }
            }
            average_edge_value = static_cast<ValueT>(mean);
        }
        return average_edge_value;
    }

    SizeT GetOutNodes()
    {
        // compute out_nodes
        SizeT out_node = 0;
        for (SizeT node = 0; node < nodes; node++)
        {
            if (row_offsets[node + 1] - row_offsets[node] > 0)
            {
                ++out_node;
            }
        }
        out_nodes = out_node;
        return out_node;
    }

    /**@}*/

    /**
     * @brief Deallocates CSR graph
     */
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        if (retval = util::FreePinned(row_offsets, pinned))
            return retval;

        if (retval = util::FreePinned(column_indices, pinned))
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
     * @brief CSR destructor
     */
    ~Csr()
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
