// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * market.cuh
 *
 * @brief MARKET Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <libgen.h>
#include <iostream>

#include <gunrock/graphio/utils.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace graphio {
namespace market  {

template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadLabelStream(
    FILE *f_in,
    char *output_file,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool quiet = false)
{
    SizeT lines_read = -1;
    SizeT nodes = 0;

    char line[1024];
    Value *labels = NULL;

    time_t mark0 = time(NULL);
    if(!quiet) printf("Parsing node labels...\n");

    fflush(stdout);

    while (true)
    {

        if (fscanf(f_in, "%[^\n]\n", line) <= 0)
        {
            break;
        }

        if (line[0] == '%')
        {

            // Comment

        }
        else if (lines_read == -1)
        {

            // Problem description
            long long ll_nodes_x, ll_nodes_y;
            if (sscanf(line, "%lld %lld",
                       &ll_nodes_x, &ll_nodes_y) != 2)
            {
                fprintf(stderr, "Error parsing node labels:"
                        " invalid problem description.\n");
                return -1;
            }

            if (ll_nodes_x != ll_nodes_y)
            {
                fprintf(stderr,
                        "Error parsing node labels: not square (%lld, %lld)\n",
                        ll_nodes_x, ll_nodes_y);
                return -1;
            }

            nodes = ll_nodes_x;

            if (!quiet)
            {
                printf(" (%lld nodes)... ",
                       (unsigned long long) ll_nodes_x);
                fflush(stdout);
            }

            // Allocate node labels
            unsigned long long allo_size = sizeof(Value);
            allo_size = allo_size * nodes;
            labels = (Value*)malloc(allo_size);
            if (labels == NULL)
            {
                fprintf(stderr, "Error parsing node labels:"
                    "labels allocation failed, sizeof(Value) = %lu,"
                    " nodes = %lld, allo_size = %lld\n",
                    sizeof(Value), (long long)nodes, (long long)allo_size);
                return -1;
            }

            lines_read++;

        }
        else
        {

            // node label description (v -> l)
            if (!labels)
            {
                fprintf(stderr, "Error parsing node labels: invalid format\n");
                return -1;
            }
            if (lines_read >= nodes)
            {
                fprintf(stderr,
                        "Error parsing node labels:"
                        "encountered more than %lld nodes\n",
                        (long long)nodes);
                if (labels) free(labels);
                return -1;
            }

            long long ll_node, ll_label;
            if (sscanf(line, "%lld %lld", &ll_node, &ll_label) != 2)
                {
                    fprintf(stderr,
                            "Error parsing node labels: badly formed\n");
                    if (labels) free(labels);
                    return -1;
                }

	    labels[lines_read] = ll_label;

            lines_read++;

        }
    }

    if (labels == NULL)
    {
        fprintf(stderr, "No input labels found\n");
        return -1;
    }

    if (lines_read != nodes)
    {
        fprintf(stderr,
                "Error parsing node labels: only %lld/%lld nodes read\n",
                (long long)lines_read, (long long)nodes);
        if (labels) free(labels);
        return -1;
    }

    time_t mark1 = time(NULL);
    if (!quiet)
    {
        printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
        fflush(stdout);
    }

    // Convert labels into binary
    csr_graph.template FromLabels<LOAD_VALUES>(output_file, labels, nodes, quiet);

    free(labels);
    fflush(stdout);

    return 0;
}

/**
 * @brief Reads a MARKET graph from an input-stream into a CSR sparse format
 *
 * Here is an example of the matrix market format
 * +----------------------------------------------+
 * |%%MatrixMarket matrix coordinate real general | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment lines
 * |%                                             | <--+
 * |  M N L                                       | <--- rows, columns, entries
 * |  I1 J1 A(I1, J1)                             | <--+
 * |  I2 J2 A(I2, J2)                             |    |
 * |  I3 J3 A(I3, J3)                             |    |-- L lines
 * |     . . .                                    |    |
 * |  IL JL A(IL, JL)                             | <--+
 * +----------------------------------------------+
 *
 * Indices are 1-based i.2. A(1,1) is the first element.
 *
 */

/**
 * @brief (Special for SM) Read csr arrays directly instead of transfer from coo format
 * @param[in] f_in          Input graph file name.
 * @param[in] f_label       Input label file name.
 * @param[in] csr_graph     Csr graph object to store the graph data.
 * @param[in] undirected    Is the graph undirected or not?
 */
template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int ReadCsrArrays_SM(char *f_in, char *f_label, Csr<VertexId, SizeT, Value> &csr_graph,
                  bool undirected, bool quiet)
{
    csr_graph.template FromCsr_SM<LOAD_VALUES>(f_in, f_label, quiet);
    return 0;
}

/**
 * \defgroup Public Interface
 * @{
 */


/**
 * @brief Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from STDIN.
 * @param[in] output_file Output file name for binary i/o.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed Is the graph reversed or not?
 * @param[in] quiet If true, print no output
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
template <
    bool HAS_EDGE_VALUES,
    bool HAS_NODE_VALUES,
    typename CooT>
cudaError_t Read(
    const char *mm_filename,
    CooT &coo,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    typedef typename CooT::VertexT  VertexT;
    typedef typename CooT::SizeT    SizeT;
    typedef typename CooT::ValueT   ValueT;
    typedef typename CooT::EdgeT    EdgeT;

    cudaError_t retval = cudaSuccess;
    FILE *f_in = NULL;
    if (mm_filename == NULL)
    {
        // Read from stdin
        if (!quiet)
        {
            printf("Reading from stdin...");
        }
        f_in = stdin;
    } else {
        // Read from file
        f_in = fopen(mm_filename, "r");
        if (f_in)
        {
            if (!quiet)
            {
                printf("Reading from file %s ...", mm_filename);
            }
        } else {
            return util::GRError("Unable to open file", __FILE__, __LINE__);
        }
    }

    SizeT edges_read = -1;
    SizeT nodes = 0;
    SizeT edges = 0;
    bool  skew  = false; //whether edge values are the inverse for symmetric matrices
    bool  array = false; //whether the mtx file is in dense array format

    time_t mark0 = time(NULL);
    if (!quiet)
    {
        printf("  Parsing MARKET COO format...");
    }
    fflush(stdout);

    char line[1024];
    //bool ordered_rows = true;
    if (retval = coo.Release()) return retval;

    while (true)
    {
        if (fscanf(f_in, "%[^\n]\n", line) <= 0)
        {
            break;
        }

        if (line[0] == '%')
        {
            // Comment
            if (strlen(line) >= 2 && line[1] == '%')
            {
                // Banner
                if (!undirected)
                    undirected = (strstr(line, "symmetric") != NULL);
                skew       = (strstr(line, "skew"     ) != NULL);
                array      = (strstr(line, "array"    ) != NULL);
                printf("undirected = %s, skew = %s, array = %s\n",
                    undirected ? "true" : "false",
                    skew ? "true" : "false",
                    array ? "true" : "false");
            }
        }

        else if (edges_read == -1)
        {
            // Problem description
            long long ll_nodes_x, ll_nodes_y, ll_edges;
            int items_scanned = sscanf(line, "%lld %lld %lld",
                       &ll_nodes_x, &ll_nodes_y, &ll_edges);

            if (array && items_scanned == 2)
            {
                ll_edges = ll_nodes_x * ll_nodes_y;
            }

            else if (!array && items_scanned == 3)
            {
                if (ll_nodes_x != ll_nodes_y)
                {
                    char err_msg[512];
                    sprintf(err_msg,
                        "Error parsing MARKET graph: not square (%lld, %lld)\n",
                        ll_nodes_x, ll_nodes_y);
                    return util::GRError(err_msg, __FILE__, __LINE__);
                }
                if (undirected) ll_edges *=2;

            } else {
                return util::GRError("Error parsing MARKET graph: invalid problem description.", __FILE__, __LINE__);
            }

            nodes = ll_nodes_x;
            edges = ll_edges;

            if (!quiet)
            {
                printf(" (%lld nodes, %lld directed edges)... ",
                       (unsigned long long) ll_nodes_x,
                       (unsigned long long) ll_edges);
                fflush(stdout);
            }

            // Allocate coo graph
            if (retval = coo.FromScratch(nodes, edges, HAS_NODE_VALUES))
                return retval;

            edges_read++;

        } else {
            // Edge description (v -> w)
            if (coo.coo_edges == (EdgeT*)NULL)
            {
                return util::GRError("Error parsing MARKET graph: invalid format.", __FILE__, __LINE__);
            }

            if (edges_read >= edges)
            {
                char err_msg[512];
                sprintf(err_msg,
                    "Error parsing MARKET graph:"
                    "encountered more than %lld edges\n",
                    (long long)edges);
                return util::GRError(err_msg, __FILE__, __LINE__);
            }

            long long ll_row, ll_col, ll_value;
            // Value ll_value;  // used for parse float / double
            int num_input;
            if (HAS_EDGE_VALUES)
            {
                num_input = sscanf(line, "%lld %lld %lld",
                                   &ll_row, &ll_col, &ll_value);
                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                }

                else if (array || num_input < 2)
                {
                    return util::GRError("Error parsing MARKET graph: badly formed edge", __FILE__, __LINE__);
                }

                else if (num_input == 2)
                {
                    ll_value = rand() % 64;
                }

            } else {
                num_input = sscanf(line, "%lld %lld", &ll_row, &ll_col);

                if (array && (num_input == 1))
                {
                    ll_value = ll_row;
                    ll_col   = edges_read / nodes;
                    ll_row   = edges_read - nodes * ll_col;
                }

                else if (array || (num_input != 2))
                {
                    return util::GRError("Error parsing MARKET graph: badly formed edge", __FILE__, __LINE__);
                }
            }

            if (HAS_EDGE_VALUES)
            {
                coo.coo_edges[edges_read].SetVal(ll_value);
            }
            if (reversed && !undirected)
            {
                coo.coo_edges[edges_read].col = ll_row - 1;   // zero-based array
                coo.coo_edges[edges_read].row = ll_col - 1;   // zero-based array
                //ordered_rows = false;
            } else {
                coo.coo_edges[edges_read].row = ll_row - 1;   // zero-based array
                coo.coo_edges[edges_read].col = ll_col - 1;   // zero-based array
                //ordered_rows = false;
            }

            edges_read++;

            if (undirected)
            {
                // Go ahead and insert reverse edge
                coo.coo_edges[edges_read].row = ll_col - 1;       // zero-based array
                coo.coo_edges[edges_read].col = ll_row - 1;       // zero-based array

                if (HAS_EDGE_VALUES)
                {
                    coo.coo_edges[edges_read].SetVal(ll_value * (skew ? -1 : 1));
                }

                //ordered_rows = false;
                edges_read++;
            }
        }
    }

    if (coo.coo_edges == (EdgeT*)NULL)
    {
        return util::GRError("No graph found", __FILE__, __LINE__);
    }

    if (edges_read != edges)
    {
        char err_msg[512];
        sprintf(err_msg,
            "Error parsing MARKET graph: only %lld/%lld edges read",
            (long long)edges_read, (long long)edges);
        return util::GRError(err_msg, __FILE__, __LINE__);
    }

    time_t mark1 = time(NULL);
    if (!quiet)
    {
        printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
        fflush(stdout);
    }

    if (mm_filename != NULL) fclose(f_in);
    return retval;
}


/**
 * @brief (Special for SM) Loads a MARKET-formatted CSR graph from the specified file.
 *
 * @param[in] mm_filename Graph file name, if empty, it is loaded from STDIN.
 * @param[in] label_filename Label file name, if empty, it is loaded from STDIN.
 * @param[in] output_file Output file name for binary i/o.
 * @param[in] output_label Output label file name for binary i/o.
 * @param[in] csr_graph Reference to CSR graph object. @see Csr
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed   Whether or not the graph is inversed.
 * @param[in] quiet If true, print no output
 *
 * \return If there is any File I/O error along the way. 0 for no error.
 */
/*template<bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph_SM(
    //char *mm_filename,
    char *label_filename,
    //char *output_file,
    char *output_label,
    Csr<VertexId, SizeT, Value> &csr_graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    FILE *_file = fopen(output_file, "r");
    FILE *_label = fopen(output_label, "r");
    if (_file && _label)
    {
        fclose(_file);
        fclose(_label);
            if (ReadCsrArrays_SM<LOAD_VALUES>(
                    output_file, output_label, csr_graph, undirected, quiet) != 0)
                return -1;
    }
    else
    {
        if (mm_filename == NULL && label_filename == NULL)
        {
            // Read from stdin
            if (!quiet)
            {
                printf("Reading from stdin:\n");
            }
            if (ReadMarketStream<false>(
                        stdin, output_file, csr_graph, undirected, reversed) != 0)
            {
                return -1;
            }
        }
        else
        {
            // Read from file
            FILE *f_in = fopen(mm_filename, "r");
            if (f_in)
            {
                if (!quiet)
                {
                    printf("Reading from %s:\n", mm_filename);
                }
                if (ReadMarketStream<false>(
                            f_in, output_file, csr_graph,
                            undirected, reversed, quiet) != 0)
                {
                    fclose(f_in);
                    return -1;
                }
                fclose(f_in);
            }
            else
            {
                perror("Unable to open graph file");
                return -1;
            }

	    // Read from label
            FILE *label_in = fopen(label_filename, "r");
	    if(label_in)
	    {
		if(!quiet) printf("Reading form %s:\n", label_filename);
 		if(ReadLabelStream<LOAD_VALUES>(label_in, output_label, csr_graph, quiet) != 0)
		{
		    fclose(label_in);
		    return -1;
		}
		fclose(label_in);
	    }
	    else
	    {
		perror("Unable to open label file");
		return -1;
	    }

        }
    }
    return 0;
}*/

/**
 * @brief read in graph function read in graph according to its type.
 *
 * @tparam LOAD_VALUES
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] file_in    Input MARKET graph file.
 * @param[in] file_label Input label file.
 * @param[in] graph      CSR graph object to store the graph data.
 * @param[in] undirected Is the graph undirected or not?
 * @param[in] reversed   Whether or not the graph is inversed.
 * @param[in] quiet     Don't print out anything to stdout
 *
 * \return int Whether error occurs (0 correct, 1 error)
 */
/*template <bool LOAD_VALUES, typename VertexId, typename SizeT, typename Value>
int BuildMarketGraph_SM(
    char *file_in,
    char *file_label,
    Csr<VertexId, SizeT, Value> &graph,
    bool undirected,
    bool reversed,
    bool quiet = false)
{
    // seperate the graph path and the file name
    char *temp1 = strdup(file_in);
    char *temp2 = strdup(file_in);
    char *file_path = dirname (temp1);
    char *file_name = basename(temp2);
    char *temp3, *temp4, *label_path, *label_name;
  if(LOAD_VALUES){
    // seperate the label path and the file name
    temp3 = strdup(file_label);
    temp4 = strdup(file_label);
    label_path = dirname (temp3);
    label_name = basename(temp4);
  }
    if (undirected)
    {
        char ud[256];  // undirected graph
        char lb[256]; // label
        sprintf(ud, "%s/.%s.ud.%d.%s%s%sbin", file_path, file_name, 0,
            ((sizeof(VertexId) == 8) ? "64bVe." : ""),
            ((sizeof(Value   ) == 8) ? "64bVa." : ""),
            ((sizeof(SizeT   ) == 8) ? "64bSi." : ""));

      if(LOAD_VALUES){
        sprintf(lb, "%s/.%s.lb.%s%sbin", label_path, label_name,
            ((sizeof(VertexId) == 8) ? "64bVe." : ""),
            ((sizeof(Value   ) == 8) ? "64bVa." : ""));
      }
       //for(int i=0; ud[i]; i++) printf("%c",ud[i]); printf("\n");
       //for(int i=0; lb[i]; i++) printf("%c",lb[i]); printf("\n");
        if (BuildMarketGraph_SM<LOAD_VALUES>(file_in, file_label, ud, lb, graph,
                    true, reversed, quiet) != 0)
            return 1;
    }
    else
    {
        fprintf(stderr, "Unspecified Graph Type.\n");
        return 1;
    }
    return 0;
}*/

template <
    bool HAS_EDGE_VALUES,
    bool HAS_NODE_VALUES,
    typename CooT>
void GetCompressedFileName(
    util::CommandLineArgs &args,
    //char* file_in,
    char* compressed_filename)
{
    bool undirected = args.CheckCmdLineFlag("undirected");
    bool reversed   = args.CheckCmdLineFlag("reversed");
    char *file_in   = strdup(args.GetCmdLineArgument<std::string>("graph-file").c_str());
    // seperate the graph path and the file name
    printf("file_in = %s\n", file_in);
    char *temp1 = strdup(file_in);
    char *temp2 = strdup(file_in);
    char *file_path = dirname (temp1);
    char *file_name = basename(temp2);
    sprintf(compressed_filename, "%s/.%s.%s%d.%s%s%sbin",
        file_path, file_name,
        (undirected)? "ud." : ((reversed) ? "rv." : "di."),
        (HAS_EDGE_VALUES) ? 1 : 0,
        ((sizeof(typename CooT::VertexT) == 8) ? "64bVe." : ""),
        ((sizeof(typename CooT::ValueT ) == 8) ? "64bVa." : ""),
        ((sizeof(typename CooT::SizeT  ) == 8) ? "64bSi." : ""));
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
    typename CooT,
    typename InfoT>
cudaError_t Load(
    util::CommandLineArgs &args,
    CooT &coo,
    InfoT *info)
{
    cudaError_t retval = cudaSuccess;
    bool quiet      = args.CheckCmdLineFlag("quiet");
    bool undirected = args.CheckCmdLineFlag("undirected");
    bool inversed   = args.CheckCmdLineFlag("inversed");
    char *market_filename = strdup(args.GetCmdLineArgument<std::string>("graph-file").c_str());

    /*if (!quiet)
    {
        printf("Loading Matrix-market coordinate-formatted graph ...");
    }*/

    std::ifstream fp(market_filename);
    if (market_filename == NULL || !fp.is_open())
    {
        char error_msg[512];
        sprintf(error_msg, "Input graph file %s does not exist.", market_filename);
        return util::GRError(error_msg, __FILE__, __LINE__);
    }

    /*boost::filesystem::path market_filename_path(market_filename);
    file_stem = market_filename_path.stem().string();
    info["dataset"] = file_stem;*/
    StoreInfo<InfoT>::StoreI(info, std::string(market_filename));
    if (retval = Read<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>(
        market_filename, coo, undirected, inversed, quiet))
    {
        return retval;
    }

    //if (!quiet)
    //    printf("Done.\n");
    return retval;
}

/**@}*/

} // namespace market
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
