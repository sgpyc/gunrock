// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * rmat.cuh
 *
 * @brief R-MAT Graph Construction Routines
 */

#pragma once

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <random>
#include <time.h>

#include <gunrock/graphio/utils.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace graphio {
namespace rmat {

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
//inline double Sprng (struct drand48_data *rand_data)
inline double Sprng (
    Engine *engine,
    Distribution *distribution)
{
    return (*distribution)(*engine);
}

/**
 * @brief Utility function.
 *
 * @param[in] rand_data
 */
//inline bool Flip (struct drand48_data *rand_data)
inline bool Flip (
    Engine *engine,
    Distribution *distribution)
{
    return Sprng(engine, distribution) >= 0.5;
}

/**
 * @brief Utility function to choose partitions.
 *
 * @param[in] u
 * @param[in] v
 * @param[in] step
 * @param[in] a
 * @param[in] b
 * @param[in] c
 * @param[in] d
 * @param[in] rand_data
 */
template <typename VertexT>
inline void ChoosePartition (
    VertexT *u, VertexT *v, VertexT step,
    double a, double b, double c, double d,
    Engine *engine, Distribution *distribution)
{
    double p;
    p = Sprng(engine, distribution);

    if (p < a)
    {
        // do nothing
    }
    else if ((a < p) && (p < a + b))
    {
        *v = *v + step;
    }
    else if ((a + b < p) && (p < a + b + c))
    {
        *u = *u + step;
    }
    else if ((a + b + c < p) && (p < a + b + c + d))
    {
        *u = *u + step;
        *v = *v + step;
    }
}

/**
 * @brief Utility function to very parameters.
 *
 * @param[in] a
 * @param[in] b
 * @param[in] c
 * @param[in] d
 * @param[in] rand_data
 */
inline void VaryParams(
    double *a, double *b, double *c, double *d,
    Engine *engine, Distribution *distribution)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(engine, distribution))
    {
        *a += *a * v * Sprng(engine, distribution);
    }
    else
    {
        *a -= *a * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *b += *b * v * Sprng(engine, distribution);
    }
    else
    {
        *b -= *b * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *c += *c * v * Sprng(engine, distribution);
    }
    else
    {
        *c -= *c * v * Sprng(engine, distribution);
    }
    if (Flip(engine, distribution))
    {
        *d += *d * v * Sprng(engine, distribution);
    }
    else
    {
        *d -= *d * v * Sprng(engine, distribution);
    }

    S = *a + *b + *c + *d;

    *a = *a / S;
    *b = *b / S;
    *c = *c / S;
    *d = *d / S;
}

/**
 * @brief Builds a R-MAT CSR graph.
 *
 * @tparam WITH_VALUES Whether or not associate with per edge weight values.
 * @tparam VertexT Vertex identifier.
 * @tparam ValueT ValueT type.
 * @tparam SizeT Graph size type.
 *
 * @param[in] nodes
 * @param[in] edges
 * @param[in] graph
 * @param[in] undirected
 * @param[in] a0
 * @param[in] b0
 * @param[in] c0
 * @param[in] d0
 * @param[in] vmin
 * @param[in] vmultipiler
 * @param[in] seed
 */
template <bool HAS_EDGE_VALUES, bool HAS_NODE_VALUES, typename CooT>
cudaError_t Build(
    typename CooT::SizeT nodes,
    typename CooT::SizeT edges,
    CooT &coo,
    bool undirected,
    double a0           = 0.57,
    double b0           = 0.19,
    double c0           = 0.19,
    double d0           = 0.05,
    double vmin         = 1.00,
    double vmultipiler  = 1.00,
    int    seed         = -1,
    bool   quiet        = false)
{
    typedef typename CooT::VertexT  VertexT;
    typedef typename CooT::SizeT    SizeT;
    typedef typename CooT::ValueT   ValueT;
    typedef typename CooT::EdgeT    EdgeT;
    cudaError_t retval = cudaSuccess;

    if ((nodes < 0) || (edges < 0))
    {
        char error_msg[512];
        sprintf(error_msg, "Invalid graph size: nodes=%lld, edges=%lld",
            (long long)nodes, (long long)edges);
        return util::GRError(error_msg, __FILE__, __LINE__);
    }

    // construct COO format graph
    SizeT directed_edges = (undirected) ? edges * 2 : edges;
    if (retval = coo.FromScratch(nodes, directed_edges, HAS_NODE_VALUES))
        return retval;

    if (seed == -1) seed = time(NULL);
    if (!quiet)
    {
        printf("Generating R-MAT graph "
            "{a, b, c, d} = {%.3f, %.3f, %.3f, %.3f}, seed = %lld ...",
            a0, b0, c0, d0, (long long) seed);
        fflush(stdout);
    }
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    #pragma omp parallel
    {
        int thread_num  = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        SizeT i_start   = (long long )(edges) * thread_num / num_threads;
        SizeT i_end     = (long long )(edges) * (thread_num + 1) / num_threads;
        unsigned int seed_ = seed + 616 * thread_num;
        Engine engine(seed_);
        Distribution distribution(0.0, 1.0);

        for (SizeT i = i_start; i < i_end; i++)
        {
            EdgeT *coo_p = coo.coo_edges + i;
            double a = a0;
            double b = b0;
            double c = c0;
            double d = d0;

            VertexT u    = 1;
            VertexT v    = 1;
            VertexT step = nodes / 2;

            while (step >= 1)
            {
                ChoosePartition (&u, &v, step, a, b, c, d, &engine, &distribution);
                step /= 2;
                VaryParams (&a, &b, &c, &d, &engine, &distribution);
            }

            // create edge
            coo_p->row = u - 1; // zero-based
            coo_p->col = v - 1; // zero-based
            coo_p->SetVal(Sprng(&engine, &distribution) * vmultipiler + vmin);

            if (undirected)
            {
                EdgeT *cooi_p = coo_p + edges;
                // reverse edge
                cooi_p->row = coo_p->col;
                cooi_p->col = coo_p->row;
                cooi_p->SetVal(Sprng(&engine, &distribution) * vmultipiler + vmin);
            }
        }

        if (HAS_NODE_VALUES)
        {
            i_start   = (long long )(nodes) * thread_num / num_threads;
            i_end     = (long long )(nodes) * (thread_num + 1) / num_threads;
            for (SizeT i = i_start; i < i_end; i++)
                coo.node_values[i] = Sprng(&engine, &distribution) * vmultipiler + vmin;
        }
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    if (!quiet)
    {
        printf("Done (%.3f ms).\n", elapsed);
    }
    return retval;
}

template <typename InfoT>
struct StoreInfo
{
    template <typename SizeT>
    static void StoreI(
        InfoT *info,
        SizeT   nodes,
        SizeT   edges,
        SizeT   scale,
        SizeT   edgefactor,
        double  a,
        double  b,
        double  c,
        double  d,
        double  vmin,
        double  vmultipiler,
        int     seed)
    {
        // put everything into mObject info
        info->info["rmat_a"]           = a;
        info->info["rmat_b"]           = b;
        info->info["rmat_c"]           = c;
        info->info["rmat_d"]           = d;
        info->info["rmat_seed"]        = seed;
        info->info["rmat_scale"]       = (int64_t)scale;
        info->info["rmat_nodes"]       = (int64_t)nodes;
        info->info["rmat_edges"]       = (int64_t)edges;
        info->info["rmat_edgefactor"]  = (int64_t)edgefactor;
        info->info["rmat_vmin"]        = vmin;
        info->info["rmat_vmultipiler"] = vmultipiler;
    }
};

template <>
struct StoreInfo<util::NullType>
{
    template <typename SizeT>
    static void StoreI(
        util::NullType *info,
        SizeT   nodes,
        SizeT   edges,
        SizeT   scale,
        SizeT   edgefactor,
        double  a,
        double  b,
        double  c,
        double  d,
        double  vmin,
        double  vmultipiler,
        int     seed)
    {
    }
};

template <
    bool HAS_EDGE_VALUES,
    bool HAS_NODE_VALUES,
    typename CooT,
    typename InfoT>
cudaError_t Generate(
    util::CommandLineArgs &args,
    CooT &coo,
    InfoT *info)
{
    typedef typename CooT::VertexT  VertexT;
    typedef typename CooT::SizeT    SizeT;
    typedef typename CooT::ValueT   ValueT;
    typedef typename CooT::EdgeT    EdgeT;
    cudaError_t retval = cudaSuccess;

    // parse R-MAT parameters
    SizeT   nodes       = 1 << 10;
    SizeT   edges       = 1 << 10;
    SizeT   scale       = 10;
    SizeT   edgefactor  = 48;
    double  a           = 0.57;
    double  b           = 0.19;
    double  c           = 0.19;
    double  d           = 1 - (a + b + c);
    double  vmin        = 1;
    double  vmultipiler = 64;
    int     seed        = -1;
    bool    undirected  = args.CheckCmdLineFlag("undirected");
    bool    quiet       = args.CheckCmdLineFlag("quiet");

    args.GetCmdLineArgument("rmat_scale",       scale);
    nodes = 1 << scale;
    args.GetCmdLineArgument("rmat_nodes",       nodes);
    args.GetCmdLineArgument("rmat_edgefactor",  edgefactor);
    edges = nodes * edgefactor;
    args.GetCmdLineArgument("rmat_edges",       edges);
    args.GetCmdLineArgument("rmat_a",           a);
    args.GetCmdLineArgument("rmat_b",           b);
    args.GetCmdLineArgument("rmat_c",           c);
    d = 1 - (a + b + c);
    args.GetCmdLineArgument("rmat_d",           d);
    args.GetCmdLineArgument("rmat_seed",        seed);
    args.GetCmdLineArgument("rmat_vmin",        vmin);
    args.GetCmdLineArgument("rmat_vmultipiler", vmultipiler);

    StoreInfo<InfoT>::StoreI(info,
        nodes, edges, scale, edgefactor,
        a, b, c, d, vmin, vmultipiler, seed);

    if (retval = Build<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>
        (nodes, edges, coo, undirected,
        a, b, c, d, vmin, vmultipiler, seed, quiet))
        return retval;

    return retval;
}

} // namespace rmat
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
