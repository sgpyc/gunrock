// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graphio.cuh
 *
 * @brief GraphIo generation headers
 */

#pragma once

#include <string>
#include <gunrock/coo.cuh>
#include <gunrock/csr.cuh>
#include <gunrock/csc.cuh>
#include <gunrock/util/test_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/binary.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/grmat.cuh>
#include <gunrock/graphio/rgg.cuh>
#ifdef BOOST_VERSION
    #include <gunrock/graphio/small_world.cuh>
#endif
namespace gunrock {
namespace graphio {

/**
 * @brief Utility function to load input graph.
 *
 * @tparam EDGE_VALUE
 * @tparam INVERSE_GRAPH
 *
 * @param[in] args Command line arguments.
 * @param[in] csr_ref Reference to the CSR graph.
 *
 * \return int whether successfully loaded the graph (0 success, 1 error).
 */
template <
    bool HAS_EDGE_VALUES,
    bool HAS_NODE_VALUES,
    typename CooT,
    typename CsrT,
    typename CscT,
    typename InfoT>
cudaError_t LoadGraph(
    util::CommandLineArgs &args,
    CooT *coo,
    CsrT *csr,
    CscT *csc,
    InfoT *info)
{
    cudaError_t retval = cudaSuccess;
    bool require_coo = (coo == (CooT*)NULL)? false : true;
    bool require_csr = (csr == (CsrT*)NULL)? false : true;
    bool require_csc = (csc == (CscT*)NULL)? false : true;
    bool coo_ready = false;
    bool csr_ready = false;
    bool csc_ready = false;

    std::string graph_type;
    args.GetCmdLineArgument("graph-type", graph_type);
    printf("graph-type = %s\n", graph_type.c_str());
    bool pinned = args.CheckCmdLineFlag("pinned-graph");
    bool quiet  = args.CheckCmdLineFlag("quiet");
    char compressed_filename[512];
    bool write_binary = false;

    if (graph_type == "market")
    {
        market::GetCompressedFileName<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>
            (args, compressed_filename);
        FILE *try_file = fopen(compressed_filename, "r");
        if (try_file)
        { // compressed file available
            fclose(try_file);
            if (csr == (CsrT*)NULL) csr = new CsrT(pinned);
            retval = binary::Read
                <HAS_EDGE_VALUES, HAS_NODE_VALUES, CsrT>
                (compressed_filename, csr[0], quiet);
            csr_ready = true;
        } else {
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            printf("reading %s\n",compressed_filename);
            retval = market::Load <HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
                (args, coo[0], info);
            write_binary = true;
            coo_ready = true;
        }
    }

    else if (graph_type == "binary")
    {
        if (csr == (CsrT*)NULL) csr = new CsrT(pinned);
        retval = binary::Load <HAS_EDGE_VALUES, HAS_NODE_VALUES, CsrT, InfoT>
            (args, csr[0], info);
        csr_ready = true;
    }

    else if (graph_type == "rmat")
    {
        if (coo == (CooT*)NULL) coo = new CooT(pinned);
        retval = rmat::Generate   <HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
            (args, coo[0], info);
        coo_ready = true;
    }

    else if (graph_type == "grmat")
    {
        if (coo == (CooT*)NULL) coo = new CooT(pinned);
        retval = grmat::Generate  <HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
            (args, coo[0], info);
        coo_ready = true;
    }

    /*else if (graph_type == "metarmat")
    {
        if (coo == (CooT*)NULL) coo = new CooT(pinned);
        retval = metarmat::Generate<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
            (args, coo[0], info);
        coo_ready = true;
    }*/

    else if (graph_type == "rgg")
    {
        if (coo == (CooT*)NULL) coo = new CooT(pinned);
        retval = rgg::Generate <HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
            (args, coo[0], info);
        coo_ready = true;
    }

    /*else if (graph_type == "smallworld")
    {
        #ifdef defined(BOOST_VERSION)
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            retval = smallworld::Generate<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT, InfoT>
                (args, coo[0], info);
            coo_ready = true;
        #else
            retval = util::GRError("Small world generator requires boost library", __FILE__, __LINE__);
        #endif
    }*/

    else retval = util::GRError(cudaErrorInvalidConfiguration,
        "Uknown graph type", __FILE__, __LINE__);

    if (retval) return retval;

    if ((require_csr || write_binary) && !csr_ready)
    {
        if (csc_ready && !coo_ready)
        {
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            if (retval = coo -> FromCsc(csc[0]))
                return retval;
            coo_ready = true;
        }

        if (coo_ready)
        {
            if (csr == (CsrT*) NULL) csr = new CsrT(pinned);
            if (retval = csr ->template FromCoo<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>
                (*coo, false, quiet))
                return retval;
            csr_ready = true;
        }
    }

    if (require_csc && !csc_ready)
    {
        if (csr_ready && !coo_ready)
        {
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            if (retval = coo -> FromCsr(csr[0]))
                return retval;
            coo_ready = true;
        }
        if (coo_ready)
        {
            if (csc == (CscT*)NULL) csc = new CscT(pinned);
            if (retval = csc ->template FromCoo<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>
                (*coo, false, quiet))
                return retval;
            csc_ready = true;
        }
    }

    if (require_coo && !coo_ready)
    {
        if (csr_ready)
        {
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            if (retval = coo -> FromCsr(csr[0]))
                return retval;
            coo_ready = true;
        } else if (csc_ready)
        {
            if (coo == (CooT*)NULL) coo = new CooT(pinned);
            if (retval = coo -> FromCsc(csc[0]))
                return retval;
            coo_ready = true;
        }
    }

    if (write_binary && csr_ready)
    {
        market::GetCompressedFileName<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>
            (args, compressed_filename);
        binary::Write<CsrT>(compressed_filename, csr[0], quiet);
    }

    if ((!require_coo) && (coo != (CooT*)NULL))
    {
        delete coo; coo = (CooT*)NULL;
    }
    if ((!require_csr) && (csr != (CsrT*)NULL))
    {
        delete csr; csr = (CsrT*)NULL;
    }
    if ((!require_csc) && (csc != (CscT*)NULL))
    {
        delete csc; csc = (CscT*)NULL;
    }
    return retval;



    /*if (graph_type == "market")  // Matrix-market graph
    {

    }
    else if (graph_type == "rmat" || graph_type == "grmat" || graph_type == "metarmat")  // R-MAT graph
    {







        // generate R-MAT graph
        else // must be metarmat
        {
            if (graphio::grmat::BuildMetaRmatGraph<EDGE_VALUE>(
                rmat_nodes,
                rmat_edges,
                csr_ref,
                info["undirected"].get_bool(),
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                rmat_vmultipiler,
                rmat_vmin,
                rmat_seed,
                args.CheckCmdLineFlag("quiet"),
                temp_devices.size(),
                gpu_idx) != 0)
            {
                return 1;
            }
        }
    }

    else if (graph_type == "smallworld")
    {
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Generating Small World Graph ...\n");
        }

        SizeT  sw_nodes = 1 << 10;
        SizeT  sw_scale = 10;
        double sw_p     = 0.0;
        SizeT  sw_k     = 6;
        int    sw_seed  = -1;
        double sw_vmultipiler = 1.00;
        double sw_vmin        = 1.00;

        args.GetCmdLineArgument("sw_scale", sw_scale);
        sw_nodes = 1 << sw_scale;
        args.GetCmdLineArgument("sw_nodes", sw_nodes);
        args.GetCmdLineArgument("sw_k"    , sw_k    );
        args.GetCmdLineArgument("sw_p"    , sw_p    );
        args.GetCmdLineArgument("sw_seed" , sw_seed );
        args.GetCmdLineArgument("sw_vmultipiler", sw_vmultipiler);
        args.GetCmdLineArgument("sw_vmin"       , sw_vmin);

        info["sw_seed"       ] = sw_seed       ;
        info["sw_scale"      ] = (int64_t)sw_scale      ;
        info["sw_nodes"      ] = (int64_t)sw_nodes      ;
        info["sw_p"          ] = sw_p          ;
        info["sw_k"          ] = (int64_t)sw_k          ;
        info["sw_vmultipiler"] = sw_vmultipiler;
        info["sw_vmin"       ] = sw_vmin       ;

        util::CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::small_world::BuildSWGraph<EDGE_VALUE>(
            sw_nodes,
            csr_ref,
            sw_k,
            sw_p,
            info["undirected"].get_bool(),
            sw_vmultipiler,
            sw_vmin,
            sw_seed,
            args.CheckCmdLineFlag("quiet")) != cudaSuccess)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Small World Graph generated in %.3lf ms, "
                "k = %lld, p = %.3lf\n",
                elapsed, (long long)sw_k, sw_p);
        }
    }
    else
    {
        fprintf(stderr, "Unspecified graph type.\n");
        exit(EXIT_FAILURE);
    }

    if (!args.CheckCmdLineFlag("quiet"))
    {
        csr_ref.GetAverageDegree();
        csr_ref.PrintHistogram();
        if (info["algorithm"].get_str().compare("SSSP") == 0)
        {
            csr_ref.GetAverageEdgeValue();
            int max_degree;
            csr_ref.GetNodeWithHighestDegree(max_degree);
            printf("Maximum degree: %d\n", max_degree);
        }
    }
    return 0;*/
}

/**
 * @brief SM Utility function to load input graph.
 *
 * @tparam NODE_VALUE
 *
 * @param[in] args Command line arguments.
 * @param[in] csr_ref Reference to the CSR graph.
 *
 * \return int whether successfully loaded the graph (0 success, 1 error).
 */
/*template<bool NODE_VALUE, typename GraphT>
int LoadGraph_SM(
    util::CommandLineArgs &args,
    GraphT &csr_ref,
    std::string type)
{
    std::string graph_type = args.GetCmdLineArgvGraphType();
    if (graph_type == "market")  // Matrix-market graph
    {
        if (!args.CheckCmdLineFlag("quiet"))
        {
            printf("Loading Matrix-market coordinate-formatted graph ...\n");
        }
        char *market_filename = NULL;
        char *label_filename = NULL;

        if(type=="query"){
            market_filename = args.GetCmdLineArgvQueryDataset();
            if(NODE_VALUE)
                label_filename = args.GetCmdLineArgvQueryLabel();
        }
        else
        {
            market_filename = args.GetCmdLineArgvDataDataset();
            if(NODE_VALUE)
                label_filename = args.GetCmdLineArgvDataLabel();
        }

        if (market_filename == NULL)
        {
            printf("Log.");
            fprintf(stderr, "Input graph does not exist.\n");
            return 1;
        }

        if (NODE_VALUE && label_filename == NULL)
        {
            printf("Log.");
            fprintf(stderr, "Input graph labels does not exist.\n");
            return 1;
        }*/

        /*boost::filesystem::path market_filename_path(market_filename);
        file_stem = market_filename_path.stem().string();
        info["dataset"] = file_stem;*/
        /*if (graphio::BuildMarketGraph_SM<NODE_VALUE>(
            market_filename,
            label_filename,
            csr_ref,
            args.CheckCmdLineFlag("undirected"),
            false,
            args.CheckCmdLineFlag("quiet")) != 0)
        {
            return 1;
        }
    }
    else
    {
        fprintf(stderr, "Unspecified graph type.\n");
        return 1;
    }

    if (!args.CheckCmdLineFlag("quiet"))
    {
        csr_ref.GetAverageDegree();
        csr_ref.PrintHistogram();
        if (info["algorithm"].get_str().compare("SSSP") == 0)
        {
            csr_ref.GetAverageEdgeValue();
            int max_degree;
            csr_ref.GetNodeWithHighestDegree(max_degree);
            printf("Maximum degree: %d\n", max_degree);
        }
    }
    return 0;
}*/

/**
 * @brief SM Initialization process for Info.
 *
 * @param[in] algorithm_name Algorithm name.
 * @param[in] args Command line arguments.
 * @param[in] csr_query_ref Reference to the CSR structure.
 * @param[in] csr_data_ref Reference to the CSR structure.
 */
/*template <typename GraphT>
void Init_SM(
    util::CommandLineArgs &args,
    GraphT &csr_query_ref,
    GraphT &csr_data_ref)
{
    if (info["node_value"].get_bool())
    {
        LoadGraph_SM<true>(args, csr_query_ref, "query");
        LoadGraph_SM<true>(args, csr_data_ref, "data");
    } else {
        LoadGraph_SM<false>(args, csr_query_ref, "query");
        LoadGraph_SM<false>(args, csr_data_ref, "data");
    }
    csr_query_ptr = &csr_query_ref;
    csr_data_ptr = &csr_data_ref;

    InitBase("SM", args);
}*/
} // namespace graphio
} // namespace gunrock
