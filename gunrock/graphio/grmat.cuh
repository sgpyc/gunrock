// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.

/**
 * @file
 * grmat.cuh
 *
 * @brief gpu based R-MAT Graph Construction Routines
 */

#pragma once

#include <curand_kernel.h>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace graphio {
namespace grmat {

__device__ __forceinline__ double Sprng(curandState &rand_state)
{
    return curand_uniform(&rand_state);
}

__device__ __forceinline__ bool Flip(curandState &rand_state)
{
    return Sprng(rand_state) >= 0.5;
}

template <typename VertexT>
__device__ __forceinline__ void ChoosePartition (
    VertexT &u, VertexT &v, VertexT step,
    double a, double b, double c, double d,
    curandState &rand_state)
{
    double p = Sprng(rand_state);

    if (p < a)
    {
        // do nothing
    } else if ((a < p) && (p < a + b))
    {
        v += step;
    }
    else if ((a + b < p) && (p < a + b + c))
    {
        u += step;
    }
    else if ((a + b + c < p) && (p < a + b + c + d))
    {
        u += step;
        v += step;
    }
}

__device__ __forceinline__ void VaryParams(
    double a, double b, double c, double d,
    curandState &rand_state)
{
    double v, S;

    // Allow a max. of 5% variation
    v = 0.05;

    if (Flip(rand_state))
    {
        a += a * v * Sprng(rand_state);
    } else {
        a -= a * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        b += b * v * Sprng(rand_state);
    } else {
        b -= b * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        c += c * v * Sprng(rand_state);
    } else {
        c -= c * v * Sprng(rand_state);
    }

    if (Flip(rand_state))
    {
        d += d * v * Sprng(rand_state);
    } else {
        d -= d * v * Sprng(rand_state);
    }

    S = a + b + c + d;

    a = a / S;
    b = b / S;
    c = c / S;
    d = d / S;
}

template <typename VertexT, typename SizeT, typename ValueT, typename EdgeT>
__global__ void Rmat_Kernel(
    SizeT          num_nodes,
    SizeT          edge_count,
    EdgeT         *d_edges,
    bool           undirected,
    ValueT         vmin,
    ValueT         vmultipiler,
    double         a0,
    double         b0,
    double         c0,
    double         d0,
    curandState   *d_rand_states)
{
    SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    curandState &rand_state = d_rand_states[i];

    while (i < edge_count)
    {
        double a = a0, b = b0, c = c0, d = d0;
        VertexT u = 1, v = 1, step = num_nodes >> 1;
        EdgeT *edge = d_edges + i;

        while (step >= 1)
        {
            ChoosePartition(u, v, step, a, b, c, d, rand_state);
            step >>= 1;
            VaryParams(a, b, c, d, rand_state);
        }
        edge -> row = u - 1;
        edge -> col = v - 1;
        edge -> SetVal(Sprng(rand_state) * vmultipiler + vmin);

        if (undirected)
        {
            edge = d_edges + (i+ edge_count);
            edge -> row = v - 1;
            edge -> col = u - 1;
            edge -> SetVal(Sprng(rand_state) * vmultipiler + vmin);
        }
        i += STRIDE;
    }
}

template <typename VertexT, typename SizeT, typename ValueT>
__global__ void GenerateNodeValues(
    SizeT node_count,
    ValueT *node_values,
    ValueT  vmin,
    ValueT  vmultipiler,
    curandState *d_rand_states)
{
    SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    curandState &rand_state = d_rand_states[i];

    while (i < node_count)
    {
        node_values [i] = Sprng(rand_state) * vmultipiler + vmin;
        i += STRIDE;
    }
}

template <typename SizeT>
__global__ void Rand_Init(
    unsigned int seed,
    curandState *d_states)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, d_states + id);
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
    typename CooT::SizeT num_nodes,
    typename CooT::SizeT num_edges,
    CooT  &coo,
    bool   undirected   = false,
    double a0           = 0.57,
    double b0           = 0.19,
    double c0           = 0.19,
    double d0           = 0.05,
    double vmin         = 1.00,
    double vmultipiler  = 1.00,
    int    seed         = -1,
    bool   quiet        = false,
    int    num_gpus     = 1,
    int   *gpu_idx      = NULL)
{
    typedef typename CooT::VertexT  VertexT;
    typedef typename CooT::SizeT    SizeT;
    typedef typename CooT::ValueT   ValueT;
    typedef typename CooT::EdgeT    EdgeT;
    cudaError_t retval = cudaSuccess;
    cudaError_t *retvals = new cudaError_t[num_gpus];

    if ((num_nodes < 0) || (num_edges < 0))
    {
        char error_msg[512];
        sprintf(error_msg, "Invalid graph size: nodes=%lld, edges=%lld",
            (long long)num_nodes, (long long)num_edges);
        return util::GRError(error_msg, __FILE__, __LINE__);
    }

    SizeT directed_edges = (undirected) ? num_edges * 2 : num_edges;
    if (retval = coo.FromScratch(num_nodes, directed_edges, HAS_NODE_VALUES))
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

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, EdgeT      > *edges
        = new util::Array1D<SizeT, EdgeT      >[num_gpus];
    util::Array1D<SizeT, ValueT     > *node_values
        = (HAS_NODE_VALUES) ? (new util::Array1D<SizeT, ValueT>[num_gpus]) : NULL;
    util::Array1D<SizeT, curandState> *rand_states
        = new util::Array1D<SizeT, curandState>[num_gpus];

    #pragma omp parallel for
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaError_t &retval_ = retvals[gpu];
        do {
            if (gpu_idx != NULL)
            {
                if (retval_ = util::SetDevice(gpu_idx[gpu]))
                    break;
            }

            if (retval_ = util::GRError(cudaStreamCreate(streams + gpu),
                "cudaStreamCreate failed", __FILE__, __LINE__))
                break;
            SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
            SizeT end_edge   = num_edges * 1.0 / num_gpus * (gpu+1);
            SizeT edge_count = end_edge - start_edge;
            SizeT start_node = num_nodes * 1.0 / num_gpus * gpu;
            SizeT end_node   = num_nodes * 1.0 / num_gpus * (gpu+1);
            SizeT node_count = end_node - start_node;
            if (undirected) edge_count *=2;

            if (retval_ = edges[gpu].Allocate(edge_count, util::DEVICE))
                break;
            if (retval_ = edges[gpu].SetPointer(coo.coo_edges + start_edge * (undirected ? 2 : 1), edge_count, util::HOST))
                break;
            if (HAS_NODE_VALUES)
            {
                if (retval_ = node_values[gpu].Allocate(node_count, util::DEVICE))
                    break;
                if (retval_ = node_values[gpu].SetPointer(coo.node_values + start_node, node_count, util::HOST))
                    break;
            }

            int block_size = (sizeof(VertexT) == 4) ? 1024 : 512;
            int grid_size = edge_count / block_size + 1;
            if (grid_size > 480) grid_size = 480;
            unsigned int seed_ = seed + 616 * gpu;
            if (retval_ = rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
                break;
            Rand_Init
                <SizeT>
                <<<grid_size, block_size, 0, streams[gpu]>>>
                (seed_, rand_states[gpu].GetPointer(util::DEVICE));

            Rmat_Kernel
                <VertexT, SizeT, ValueT, EdgeT>
                <<<grid_size, block_size, 0, streams[gpu]>>>
                (num_nodes, (undirected ? edge_count/2 : edge_count),
                edges[gpu].GetPointer(util::DEVICE),
                undirected, vmultipiler, vmin,
                a0, b0, c0, d0,
                rand_states[gpu].GetPointer(util::DEVICE));
            if (retval_ = edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0, streams[gpu]))
                break;

            if (HAS_NODE_VALUES)
            {
                block_size = 512;
                grid_size = node_count / block_size + 1;
                if (grid_size > 480) grid_size = 480;
                GenerateNodeValues
                    <VertexT, SizeT, ValueT>
                    <<<grid_size, block_size, 0, streams[gpu]>>>
                    (node_count, node_values[gpu].GetPointer(util::DEVICE),
                    vmin, vmultipiler, rand_states[gpu].GetPointer(util::DEVICE));
                if (retval_ = node_values[gpu].Move(util::DEVICE, util::HOST, node_count, 0, streams[gpu]))
                    break;
            }
        } while (false);
    }
    for (int gpu = 0; gpu < num_gpus; gpu++)
        if (retvals[gpu]) return retvals[gpu];

    #pragma omp parallel for
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaError_t &retval_ = retvals[gpu];
        do
        {
            if (gpu_idx != NULL)
            {
                if (retval_ = util::SetDevice(gpu_idx[gpu]))
                    break;
            }
            if (retval_ = util::GRError(cudaStreamSynchronize(streams[gpu]),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                break;
            if (retval_ = util::GRError(cudaStreamDestroy(streams[gpu]),
                "cudaStreamDestroy failed", __FILE__, __LINE__))
                break;
            if (retval_ = edges[gpu].Release())
                break;
            if (HAS_NODE_VALUES)
                if (retval_ = node_values[gpu].Release())
                    break;
            if (retval_ = rand_states[gpu].Release())
                break;
        } while (false);
    }
    for (int gpu = 0; gpu < num_gpus; gpu++)
        if (retvals[gpu]) return retvals[gpu];

    delete[] rand_states; rand_states = NULL;
    delete[] edges;       edges       = NULL;
    delete[] node_values; node_values = NULL;
    delete[] streams;     streams     = NULL;
    delete[] retvals;     retvals     = NULL;

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    if (!quiet)
    {
        printf("Done (%.3f ms).\n", elapsed);
    }
    return retval;
}

/**
 * @brief Builds a meta R-MAT CSR graph that connects n R-MAT graphs by a single root node.
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
 * @param[in] vmultipiler
 * @param[in] vmin
 * @param[in] seed
 */
/*template <bool WITH_VALUES, typename VertexT, typename SizeT, typename ValueT>
cudaError_t BuildMetaRmatGraph(
    SizeT num_nodes, SizeT num_edges,
    Csr<VertexT, SizeT, ValueT> &graph,
    bool undirected,
    double a0 = 0.55,
    double b0 = 0.2,
    double c0 = 0.2,
    double d0 = 0.05,
    double vmultipiler = 1.00,
    double vmin = 1.00,
    int    seed = -1,
    bool   quiet = false,
    int    num_gpus = 1,
    int   *gpu_idx = NULL)
{
    // Do not build any meta root node if num_gpus == 1
    typedef CooEdge<VertexT, ValueT> EdgeTupleType;
    cudaError_t retval = cudaSuccess;

    if ((num_nodes < 0) || (num_edges < 0))
    {
        fprintf(stderr, "Invalid graph size: nodes=%lld, edges=%lld",
                (long long)num_nodes, (long long)num_edges);
        return util::GRError("Invalid graph size");
    }

    SizeT directed_edges = (undirected) ? num_edges * 2 : num_edges;
    EdgeTupleType *coo = (EdgeTupleType*) malloc (
            sizeof(EdgeTupleType) * (directed_edges*num_gpus+((num_gpus>1)?num_gpus:0)));

    if (num_gpus > 1) {
        for (int i = 0; i < num_gpus; ++i)
        {
            coo[i].row = 0;
            coo[i].col = i*num_nodes+1;
            coo[i].val = 1; // to simplify the implementation, give 1 as weight for now.
        }
    }

    if (seed == -1) seed = time(NULL);
    if (!quiet)
    {
        printf("rmat_seed = %lld\n", (long long)seed);
    }

    cudaStream_t *streams = new cudaStream_t[num_gpus];
    util::Array1D<SizeT, EdgeTupleType> *edges = new util::Array1D<SizeT, EdgeTupleType>[num_gpus];
    util::Array1D<SizeT, curandState> *rand_states = new util::Array1D<SizeT, curandState>[num_gpus];

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        if (gpu_idx != NULL)
        {
            if (retval = util::SetDevice(gpu_idx[gpu]))
                return retval;
        }

        if (retval = util::GRError(cudaStreamCreate(streams + gpu),
                    "cudaStreamCreate failed", __FILE__, __LINE__))
            return retval;
        SizeT start_edge = num_edges * 1.0 / num_gpus * gpu;
        SizeT end_edge = num_edges * 1.0 / num_gpus * (gpu+1);
        SizeT edge_count = end_edge - start_edge;
        if (undirected) edge_count *=2;
        unsigned int seed_ = seed + 616 * gpu;
        if (retval = edges[gpu].Allocate(edge_count, util::DEVICE))
            return retval;

        int block_size = 1024;
        int grid_size = edge_count / block_size + 1;
        if (grid_size > 480) grid_size = 480;
        if (retval = rand_states[gpu].Allocate(block_size * grid_size, util::DEVICE))
            return retval;
        Rand_Init
            <SizeT>
            <<<grid_size, block_size, 0, streams[gpu]>>>
            (seed_, rand_states[gpu].GetPointer(util::DEVICE));

        Rmat_Kernel
            <WITH_VALUES, VertexT, SizeT, ValueT>
            <<<grid_size, block_size, 0, streams[gpu]>>>
            (num_nodes, (undirected ? edge_count/2 : edge_count),
             edges[gpu].GetPointer(util::DEVICE),
             undirected, vmultipiler, vmin,
             a0, b0, c0, d0,
             rand_states[gpu].GetPointer(util::DEVICE));

        // for source node: add num_nodes
        // for dest node: add num_nodes
        VertexT pre_offset = (num_gpus > 1)?1:0;
        for (int copy_idx = 0; copy_idx < num_gpus; ++copy_idx) {
            VertexT offset = (copy_idx) ? num_nodes : pre_offset;
            EdgeTupleType *edges_pointer = edges[gpu].GetPointer(util::DEVICE);
            util::MemsetAddEdgeValKernel<<<256, 1024>>>(edges_pointer, offset, edge_count);
            if (retval = edges[gpu].SetPointer(coo + ((num_gpus>1)?num_gpus:0) + directed_edges * copy_idx + start_edge * (undirected ? 2 : 1), edge_count, util::HOST))
                return retval;
            if (retval = edges[gpu].Move(util::DEVICE, util::HOST, edge_count, 0, streams[gpu]))
                return retval;
        }
    }

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        if (gpu_idx != NULL)
        {
            if (retval = util::SetDevice(gpu_idx[gpu]))
                return retval;
        }
        if (retval = util::GRError(cudaStreamSynchronize(streams[gpu]),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
            return retval;
        if (retval = util::GRError(cudaStreamDestroy(streams[gpu]),
                    "cudaStreamDestroy failed", __FILE__, __LINE__))
            return retval;
        if (retval = edges[gpu].Release())
            return retval;
        if (retval = rand_states[gpu].Release())
            return retval;
    }

    delete[] rand_states; rand_states = NULL;
    delete[] edges;   edges   = NULL;
    delete[] streams; streams = NULL;

    // convert COO to CSR
    char *out_file = NULL;  // TODO: currently does not support write CSR file
    graph.template FromCoo<WITH_VALUES, EdgeTupleType>(
            out_file, coo, num_nodes*num_gpus, directed_edges*num_gpus+((num_gpus>1)?num_gpus:0), false, undirected, false, quiet);

    free(coo);

    return retval;
}*/

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
    int     num_gpus    = 1;
    int*    gpu_idx     = NULL;

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
    if (retval = args.GetDeviceList(gpu_idx, num_gpus))
        return retval;

    StoreInfo<InfoT>::StoreI(info,
        nodes, edges, scale, edgefactor,
        a, b, c, d, vmin, vmultipiler, seed);

    if (retval = Build<HAS_EDGE_VALUES, HAS_NODE_VALUES, CooT>(
        nodes, edges, coo, undirected,
        a, b, c, d, vmin, vmultipiler, seed,
        quiet, num_gpus, gpu_idx))
        return retval;

    if (gpu_idx != NULL)
    {
        delete[] gpu_idx; gpu_idx = NULL;
    }
    return retval;
}

} // namespace grmat
} // namespace graphio
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
