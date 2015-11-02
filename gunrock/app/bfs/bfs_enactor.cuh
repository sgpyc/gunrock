// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace app {
namespace bfs {

    template <typename Problem, bool INSTRUMENT, bool DEBUG, bool SIZE_CHECK> class Enactor;
        
    template <typename VertexId, typename SizeT, typename Value,
        typename ExpandIncomingHandle>
    __global__ void Expand_Incoming_BFS (
        ExpandIncomingHandle* d_handle)
        //int gpu_num,
        //int thread_num)
    {
        __shared__ ExpandIncomingHandle s_handle;
        const SizeT STRIDE = gridDim.x * blockDim.x;
        SizeT x = threadIdx.x;
        VertexId key = 0, label = 0;

        while (x<sizeof(ExpandIncomingHandle))
        {
            ((char*)&s_handle)[x] = ((char*)d_handle)[x];
            x += blockDim.x;
        }
        __syncthreads();

        x = blockIdx.x * blockDim.x + threadIdx.x;
        while ( x < s_handle.num_elements)
        {
            key   = s_handle.keys_in      [x];
            label = s_handle.vertex_ins[0][x];

            if (key < 0 || key >= s_handle.num_nodes || label < 0)
            {
                printf("%d\t %s: x, key, label = %d, %d, %d\n", 
                    s_handle.gpu_num, __func__, x, key, label);
                s_handle.keys_out[x] = -1;
                x += STRIDE;
                continue;
            }
            
            Value old_label = s_handle.vertex_orgs[0][key];
            if (atomicCAS(s_handle.vertex_orgs[0] + key, -1, label)!= -1)
            {
                if (atomicMin(s_handle.vertex_orgs[0] + key, label) <= label)
                {
                    s_handle.keys_out[x]=-1;
                    x+=STRIDE;
                    continue;
                }
            }
            if (TO_TRACK && to_track(s_handle.gpu_num, key))
            {
                printf("%d\t %s: label[%d] (%d) -> (%d)\n", 
                    s_handle.gpu_num, __func__, key, old_label, label);
            }

            s_handle.keys_out[x]=key;
            if (s_handle.num_vertex_associates == 2) 
                s_handle.vertex_orgs[1][key]=s_handle.vertex_ins[1][x];
            x+=STRIDE;
        }
    }

template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct BFSIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
{
    typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
        BaseIteration;
    typedef typename Enactor::SizeT      SizeT     ;    
    typedef typename Enactor::Value      Value     ;    
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef typename Enactor::GraphSlice GraphSlice;
    typedef BFSFunctor<VertexId, SizeT, Value, Problem> BfsFunctor;
    typedef typename Enactor::ExpandIncomingHandle ExpandIncomingHandle;
    typedef typename Enactor::FrontierT  FrontierT;
    typedef typename Enactor::FrontierA  FrontierA;
    typedef typename Enactor::WorkProgress WorkProgress;

    BFSIteration() {}

    cudaError_t Init(
        int num_gpus,
        int num_streams)

    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseIteration::Init(num_gpus,num_streams,
            true, false, false, true, Enactor::Problem::MARK_PREDECESSORS))
            return retval;
        return retval;
    }

    cudaError_t SubQueue_Core()
    {
        cudaError_t   retval         = cudaSuccess;
        //int           thread_num     = this->thread_num;
        FrontierT    *frontier_queue = this->frontier_queue;
        util::Array1D<SizeT, SizeT> 
                     *scanned_edge   = this->scanned_edge;
        FrontierA    *frontier_attribute = this->frontier_attribute;
        EnactorStats *enactor_stats  = this->enactor_stats;
        DataSlice    *h_data_slice   = this->data_slice->GetPointer(util::HOST);
        DataSlice    *d_data_slice   = this->data_slice->GetPointer(util::DEVICE);
        GraphSlice   *graph_slice    = this->graph_slice;
        WorkProgress *work_progress  = this->work_progress;
        ContextPtr    context        = this->context;
        cudaStream_t  stream         = this->stream;
        
        frontier_attribute->queue_reset = true;
        //if (enactor_stats -> iteration == 267)
        {
            //frontier_queue -> keys[frontier_attribute -> selector].EnsureSize(
            //    frontier_attribute -> queue_length);
            //frontier_queue -> keys[frontier_attribute -> selector].Move_In(
            //    util::DEVICE, util::DEVICE, this->d_keys_in,
            //    frontier_attribute -> queue_length, 0, 0, stream);
            //util::cpu_mt::PrintGPUArray("key0", this->d_keys_in, frontier_attribute -> queue_length, 
            //    this-> gpu_num, enactor_stats -> iteration, this -> stream_num, stream);
            //util::cpu_mt::PrintGPUArray("val0", h_data_slice -> labels.GetPointer(util::DEVICE), 
            //    graph_slice -> nodes, this -> gpu_num, 
            //    enactor_stats->iteration, this->stream_num, stream);
        }

        util::MemsetKernel<<<256, 256, 0, stream>>>(
            frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE),
            (VertexId)-2,
            frontier_queue -> keys[frontier_attribute -> selector^1].GetSize());
        Check_Exist<<<enactor_stats -> filter_grid_size,
            FilterKernelPolicy::THREADS, 0, stream>>>(
            frontier_attribute -> queue_length,
            this->gpu_num, 4, enactor_stats -> iteration,
            frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));

        // Edge Map
        //this->ShowDebugInfo("Advance begin", enactor_stats->iteration);
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, BfsFunctor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edge -> GetPointer(util::DEVICE),
            /*enactor_stats -> iteration != 267 ?*/ this->d_keys_in,
            //frontier_queue-> keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue-> keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            frontier_queue-> values[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes,
            graph_slice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);
        //this -> ShowDebugInfo("Advance end", enactor_stats->iteration);

        // Only need to reset queue for once
        frontier_attribute -> queue_reset = false;
        frontier_attribute -> queue_index++;
        frontier_attribute -> selector ^= 1;
        enactor_stats      -> Accumulate(
            work_progress  ->template GetQueueLengthPointer<unsigned int,SizeT>(
            frontier_attribute->queue_index), stream);
        
        /*if ( enactor_stats -> iteration == 267)
        {
            work_progress -> GetQueueLength(frontier_attribute -> queue_index, frontier_attribute -> queue_length, false, stream, true);
            if (retval = cudaStreamSynchronize(stream)) return retval;
            //sprintf(this -> mssg, 
            //    "keys1.length = %d", frontier_attribute->queue_length);
            //this -> ShowDebugInfo(this -> mssg, enactor_stats -> iteration);
            util::cpu_mt::PrintGPUArray("keys1",
                frontier_queue -> keys[frontier_attribute->selector].GetPointer(util::DEVICE), 
                frontier_attribute->queue_length, this->gpu_num, 
                enactor_stats -> iteration, this-> stream_num, stream);
        }*/
        //return retval;

        Check_Value<<<1,1,0,stream>>>(
            work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                frontier_attribute->queue_index),
            this->gpu_num, 5, enactor_stats -> iteration);

        Check_Exist_<<<enactor_stats -> filter_grid_size,
            FilterKernelPolicy::THREADS, 0, stream>>>(
            work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                frontier_attribute->queue_index),
            this->gpu_num, 3, enactor_stats -> iteration,
            frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
        
        util::MemsetCASKernel<<<256, 256, 0, stream>>>(
            frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
            -2, -1,
            work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                frontier_attribute->queue_index));
 
        // Filter
        //this-> ShowDebugInfo("Filter begin", enactor_stats->iteration);
        gunrock::oprtr::filter::Kernel
            <FilterKernelPolicy, Problem, BfsFunctor>
            <<< enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats -> iteration+1,
            frontier_attribute -> queue_reset,
            frontier_attribute -> queue_index,
            frontier_attribute -> queue_length,
            frontier_queue     -> keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue     -> values[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue     -> keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            h_data_slice-> visited_mask.GetPointer(util::DEVICE),
            work_progress[0],
            frontier_queue     -> keys  [frontier_attribute->selector  ].GetSize(),
            frontier_queue     -> keys  [frontier_attribute->selector^1].GetSize(),
            enactor_stats-> filter_kernel_stats);
        //if (Enactor::DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
        //this-> ShowDebugInfo("Filter end", enactor_stats->iteration);
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;

        //Check_Exist_<<<enactor_stats -> filter_grid_size,
        //    FilterKernelPolicy::THREADS, 0, stream>>>(
        //    work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
        //        frontier_attribute->queue_index),
        //    this->gpu_num, 4, enactor_stats -> iteration,
        //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
 

        /*if ( enactor_stats -> iteration == 267)
        {
            work_progress -> GetQueueLength(frontier_attribute -> queue_index, frontier_attribute -> queue_length, false, stream, true);
            if (retval = cudaStreamSynchronize(stream)) return retval;
            //sprintf(this -> mssg, "keys2.length = %d", 
            //    frontier_attribute->queue_length);
            //this -> ShowDebugInfo(this -> mssg, enactor_stats -> iteration);
            util::cpu_mt::PrintGPUArray("keys2", 
                frontier_queue -> keys[frontier_attribute->selector].GetPointer(util::DEVICE), 
                frontier_attribute -> queue_length, this->gpu_num, 
                enactor_stats -> iteration, this-> stream_num, stream);
        }*/
        return retval;
    }

    //template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    cudaError_t Expand_Incoming()
    /*          int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
              VertexId*       keys_in,
        util::Array1D<SizeT, VertexId>*       keys_out,
        const size_t          array_size,
              char*           array,
              DataSlice*      data_slice)*/
    {
        cudaError_t retval = cudaSuccess;
        //bool over_sized = false;
        //Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId>(
        //    "queue1", num_elements, keys_out, over_sized, -1, -1, -1);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys_in", this->h_e_handle -> keys_in,
        //    this->num_elements, this->gpu_num, 0, this->stream_num, this->stream);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("vals_in", this->h_e_handle -> vertex_ins[0],
        //    this->num_elements, this->gpu_num, 0, this->stream_num, this->stream);
        printf("%d\t \t %d\t Expand_Incoming start, num_elements = %d\n", 
            this->gpu_num, this->stream_num, this->num_elements);fflush(stdout);
        Expand_Incoming_BFS 
            <VertexId, SizeT, Value, ExpandIncomingHandle> 
            <<< this->grid_size  , this->block_size, 0, this->stream>>> 
            (this-> d_e_handle);//, this->gpu_num, this->stream_num);
        return retval;
    }

    cudaError_t Compute_OutputLength()
        /*FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        util::Array1D<SizeT, SizeT>       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)*/
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;
        if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
            "scanned_edges", this->frontier_attribute->queue_length, 
            this-> scanned_edge, over_sized, -1, -1, -1, false)) 
            return retval;
        //printf("frontier_attribute = %p, d_offsets = %p, d_indices = %p, "
        //    "d_keys_in = %p, scanned_edge = %p, max_in = %d, max_out = %d, stream = %p\n",
        //    this-> frontier_attribute, this->d_offsets, this->d_indices, 
        //    this->d_keys_in, this->scanned_edge ->GetPointer(util::DEVICE),
        //    this -> max_in, this-> max_out, this->stream);
        //fflush(stdout);

        retval = gunrock::oprtr::advance::ComputeOutputLength
            <AdvanceKernelPolicy, Problem, BfsFunctor>(
            this-> frontier_attribute,
            this-> d_offsets,
            this-> d_indices,
            this-> d_keys_in,
            this-> scanned_edge ->GetPointer(util::DEVICE),
            this-> max_in,
            this-> max_out,
            this-> context[0],
            this-> stream,
            this-> advance_type,
            this-> express);
        return retval;
    }

    cudaError_t Check_Queue_Size()
    {    
        cudaError_t   retval             = cudaSuccess;
        bool          over_sized         = false;
        int           gpu_num            = this -> gpu_num;
        int           stream_num         = this -> stream_num;
        SizeT         request_length     = this -> request_length;
        FrontierT    *frontier_queue     = this -> frontier_queue;
        FrontierA    *frontier_attribute = this -> frontier_attribute;
        EnactorStats *enactor_stats      = this -> enactor_stats;
        GraphSlice   *graph_slice        = this -> graph_slice;
        int           selector           = frontier_attribute->selector;
        long long     iteration          = enactor_stats -> iteration;

        //if (Enactor::DEBUG)
        {
            sprintf(this -> mssg, "queue_size = %d, request_length = %d",
                frontier_queue -> keys[selector^1].GetSize(),
                request_length);
            this -> ShowDebugInfo(this -> mssg, iteration, stream_num);
        }

        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", request_length, 
            &frontier_queue->keys  [selector^1], over_sized, 
            gpu_num, iteration, stream_num, false)) 
            return retval;
        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", graph_slice->nodes+2, 
            &frontier_queue->keys  [selector  ], over_sized, 
            gpu_num, iteration, stream_num, true )) 
            return retval;
        if (Problem::USE_DOUBLE_BUFFER)
        {    
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", request_length, 
                &frontier_queue->values[selector^1], over_sized, 
                gpu_num, iteration, stream_num, false)) 
                return retval;
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", graph_slice->nodes+2, 
                &frontier_queue->values[selector  ], over_sized, 
                gpu_num, iteration, this->stream_num, true )) 
                return retval;
        }    
        return retval;
    }    

};

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <
    typename _Problem, 
    bool     _INSTRUMENT, 
    bool     _DEBUG, 
    bool     _SIZE_CHECK>
class BFSEnactor : public EnactorBase<
    typename _Problem::VertexId,
    typename _Problem::SizeT, 
    typename _Problem::Value,
    _DEBUG, 
    _SIZE_CHECK,
    (_Problem::MARK_PREDECESSORS && !_Problem::ENABLE_IDEMPOTENCE)? 2 : 1, // NUM_VERTEX_ASSOCIATES
    0> // NUM_VALUE__ASSOCIATES
{   
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
    typedef EnactorBase<VertexId, SizeT, Value, DEBUG, SIZE_CHECK,
        (Problem::MARK_PREDECESSORS && !_Problem::ENABLE_IDEMPOTENCE) ? 2: 1,
        0> BaseEnactor;
    typedef BFSEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK>
        Enactor;

    int traversal_mode;
    // Methods

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        BaseEnactor(VERTEX_FRONTIERS, num_gpus, gpu_idx)//,
    {
        printf("BFSEnactor() begin.\n");fflush(stdout);
        printf("BFSEnactor() end.\n"); fflush(stdout);
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        Release();
    }

    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t ReleaseBFS()
    {
        cudaError_t retval = cudaSuccess;
        printf("BFSEnactor::Release begin.\n");fflush(stdout);
        if (retval = BaseEnactor::template Release
            <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>())
            return retval;
        printf("BFSEanctorRelease end.\n");fflush(stdout);
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        SizeT     &search_depth,
        double    &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        EnactorSlice<Enactor> *enactor_slices 
            = (EnactorSlice<Enactor>*) this->enactor_slices;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
            if (this->num_gpus!=1)
                if (util::SetDevice(this->gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            for (int stream=0; stream< this->num_subq__streams + this->num_fullq_stream ; stream++)
            {
                EnactorStats *enactor_stats_ = (stream < this->num_subq__streams) ?
                    enactor_slices[gpu].subq__enactor_statses + stream :
                    enactor_slices[gpu].fullq_enactor_stats   + stream - this->num_subq__streams;
                total_queued += enactor_stats_ -> total_queued[0];
                enactor_stats_ -> total_queued.Move(util::DEVICE, util::HOST);
                total_queued += enactor_stats_ -> total_queued[0];
                if (enactor_stats_ -> iteration > search_depth) 
                    search_depth = enactor_stats_ -> iteration;
                total_lifetimes += enactor_stats_ -> total_lifetimes;
                total_runtimes  += enactor_stats_ -> total_runtimes;
            }
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitBFS(
        Problem *problem,
        int      max_grid_size = 0,
        int      num_input_streams = -1,
        int      num_outpu_streams = -1,
        int      num_subq__streams = -1,
        int      num_split_streams = -1)
    { 
        typedef BFSIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
            IterationT;  
        cudaError_t retval = cudaSuccess;

        printf("BFSEnactor::Init begin.\n");fflush(stdout);
        if (num_input_streams < 0) num_input_streams = this->num_gpus - 1;
        if (num_outpu_streams < 0) num_outpu_streams = this->num_gpus - 1;
        if (num_subq__streams < 0) num_subq__streams = this->num_gpus;
        if (num_split_streams < 0) num_split_streams = this->num_gpus;
        if (this->num_gpus < 2) 
        {
            num_input_streams = 0;
            num_outpu_streams = 0;
            num_split_streams = 0;
        }
        // Lazy initialization
        if (retval = BaseEnactor::template Init<
            AdvanceKernelPolicy, FilterKernelPolicy, Enactor> (
            problem,
            this,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY, 
            FilterKernelPolicy::CTA_OCCUPANCY,
            256,
            num_input_streams,
            num_outpu_streams,
            num_subq__streams,
            0,
            num_split_streams)) return retval;        
        this->problem = (void*)problem;
 
        IterationT *iteration_loops = NULL;
        for (int gpu_num = 0; gpu_num < this->num_gpus; gpu_num++)
        {
            EnactorSlice<Enactor>* enactor_slice 
                = ((EnactorSlice<Enactor>*) this->enactor_slices) + gpu_num;
            if (retval = util::SetDevice(this->gpu_idx[gpu_num])) return retval;
            if (num_input_streams > 0 && this->num_gpus > 1)
            {
                iteration_loops = new IterationT[num_input_streams];
                for (int stream_num=0; stream_num< num_input_streams; stream_num++)
                if (retval = iteration_loops[stream_num].Init(
                    this->num_gpus, num_input_streams)) return retval;
                enactor_slice -> input_iteration_loops =
                    (void*) iteration_loops;
            }
            if (num_outpu_streams > 0 && this->num_gpus > 1)
            {
                iteration_loops = new IterationT[num_outpu_streams];
                for (int stream_num=0; stream_num < num_outpu_streams; stream_num++)
                if (retval = iteration_loops[stream_num].Init(
                    this->num_gpus, num_outpu_streams)) return retval;
                enactor_slice -> outpu_iteration_loops =
                    (void*) iteration_loops;
            }
            if (num_subq__streams > 0)
            {
                iteration_loops = new IterationT[num_subq__streams];
                for (int stream_num=0; stream_num < num_subq__streams; stream_num++)
                if (retval = iteration_loops[stream_num].Init(
                    this->num_gpus, num_subq__streams)) return retval;
                enactor_slice -> subq__iteration_loops =
                    (void*) iteration_loops;
            }
            if (num_split_streams > 0 && this->num_gpus > 1)
            {
                iteration_loops = new IterationT;
                if (retval = iteration_loops[0].Init(
                    this->num_gpus, num_split_streams)) return retval;
                iteration_loops -> direction = Enactor::MakeOutHandle::Direction::FORWARD;
                enactor_slice -> split_iteration_loop =
                    (void*) iteration_loops;
            }
        }
        printf("BFSEnactor::Init end.\n");fflush(stdout);
        return retval;
    }

    template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
    cudaError_t ResetBFS(
        VertexId src,
        FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed
        double subq__factor  = 1.0,
        double subq__factor0 = 1.0,
        double subq__factor1 = 1.0,
        double fullq_factor  = 1.0,
        double fullq_factor0 = 1.0,
        double fullq_factor1 = 1.0,
        double input_factor  = 1.0,
        double outpu_factor  = 1.0,
        double split_factor  = 1.0,
        double temp_factor   = 0.1) // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
    {
        cudaError_t retval = cudaSuccess;
        printf("BFSEnactor::Reset begin.\n");fflush(stdout);
        int         gpu    = 0;
        VertexId    tsrc   = src;
        Problem   *problem = (Problem*) this->problem;
 
        //if (queue_sizing1 < 0) queue_sizing1 = queue_sizing;
        if (retval = BaseEnactor::template Reset
            <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>(
            frontier_type, 
            subq__factor, subq__factor0, subq__factor1,
            fullq_factor, fullq_factor0, fullq_factor1,
            input_factor, outpu_factor, split_factor,
            temp_factor)) return retval;

        this -> using_subq = true;
        EnactorSlice<Enactor> *enactor_slices 
            = (EnactorSlice<Enactor>*) this->enactor_slices; 
        // Fillin the initial input_queue for BFS problem
        if (this->num_gpus > 1)
        {   
            gpu = problem->partition_tables [0][src];
            tsrc= problem->convertion_tables[0][src];
        }
        printf("gpu = %d, gpu_idx[gpu] = %d\n", gpu, this->gpu_idx[gpu]);
        fflush(stdout); 
        if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
        if (retval = util::GRError(cudaMemcpy(
            enactor_slices[gpu].subq__frontiers[0].keys[0].GetPointer(util::DEVICE),
            &tsrc, sizeof(VertexId), cudaMemcpyHostToDevice),
            "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) 
            return retval;
        if (retval = enactor_slices[gpu].subq__queue.Push(1,
            enactor_slices[gpu].subq__frontiers[0].keys[0].GetPointer(util::DEVICE))) return retval;
        
        VertexId src_label = 0;
        if (retval = util::GRError(cudaMemcpy(
            problem->data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,
            &src_label, sizeof(VertexId), cudaMemcpyHostToDevice),
            "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__))
            return retval; 

       if (Problem::MARK_PREDECESSORS && !Problem::ENABLE_IDEMPOTENCE) {
            VertexId src_pred = -1; 
            if (retval = util::GRError(cudaMemcpy(
                problem->data_slices[gpu]->preds.GetPointer(util::DEVICE) + tsrc,
                &src_pred, sizeof(VertexId), cudaMemcpyHostToDevice),
                "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__))
                return retval; 
        }   

        for (int gpu_num = 0; gpu_num < this-> num_gpus; gpu_num++)
        {
            if (gpu_num == gpu) continue;
            if (retval = util::SetDevice(this->gpu_idx[gpu_num])) return retval;
            if (retval = enactor_slices[gpu_num].subq__queue.Push(0, NULL))
                return retval;
        }
        printf("BFSEnactor::Reset end.\n");fflush(stdout);
        return retval;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBFS(
        VertexId    src)
    {
        typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
            ThreadSlice;
        clock_t      start_time = clock();
        cudaError_t  retval     = cudaSuccess;
        EnactorSlice<Enactor> *enactor_slices 
            = (EnactorSlice<Enactor>*) this->enactor_slices; 
        ThreadSlice* thread_slices = (ThreadSlice*) this->thread_slices;
        this -> using_subq = true;
        this -> using_fullq = false;
        this -> num_vertex_associates = (this-> num_gpus > 1) ? 
            this->NUM_VERTEX_ASSOCIATES : 0;
        this -> num_value__associates = (this-> num_gpus > 1) ?
            this->NUM_VALUE__ASSOCIATES : 0; 

        if (this-> num_gpus > 1)
        {
            for (int gpu_num = 0; gpu_num < this-> num_gpus; gpu_num++)
            {
                if (this->num_vertex_associates > 0)
                    enactor_slices[gpu_num].vertex_associate_orgs[0]
                        = ((Problem*)this->problem)->data_slices[gpu_num]->labels.GetPointer(util::DEVICE);
                if (this->num_vertex_associates > 1)
                    enactor_slices[gpu_num].vertex_associate_orgs[1]
                        = ((Problem*)this->problem)->data_slices[gpu_num]->preds .GetPointer(util::DEVICE);
            }
        }

        do {
            for (int i=0; i<this->num_threads; i++)
                thread_slices[i].status = ThreadSlice::Status::Running;

            bool all_done = false;
            int  done_counter = 0;
            while (!all_done)
            {
                //std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (done_counter == 20)
                {
                    all_done = All_Done<ThreadSlice>(this, -1);
                    if (all_done) break;
                    done_counter = 0;
                } else done_counter ++;
                std::this_thread::yield();
            }

            for (int i=0; i<this->num_threads; i++)
                thread_slices[i].status = ThreadSlice::Status::Wait;
        } while(0);

        if (this->DEBUG) printf("GPU BFS Done.\n");
        return retval;
    }

    void Show_Mem_Stats()
    {
        Show_Mem_Stats_ <Enactor>(this);
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    ForwardAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    ForwardAdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LBAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LBAdvanceKernelPolicy;

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(VertexId    src)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300)
        {
            if (Problem::ENABLE_IDEMPOTENCE) {
                if (traversal_mode == 0)
                    return EnactBFS<     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(src);
                else
                    return EnactBFS<ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(src);
            } else {
                if (traversal_mode == 0)
                    return EnactBFS<     LBAdvanceKernelPolicy     , FilterKernelPolicy>(src);
                else
                    return EnactBFS<ForwardAdvanceKernelPolicy     , FilterKernelPolicy>(src);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     * @param[in] traversal_mode Traversal Mode for advance operator: Load-balanced or Dynamic cooperative
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        Problem     *problem,
        int         max_grid_size  = 0,
        int         traversal_mode = 0)
    {
        int min_sm_version = -1;
        this -> traversal_mode = traversal_mode;

        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            if (Problem::ENABLE_IDEMPOTENCE) {
                if (traversal_mode == 0)
                    return InitBFS<     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                            problem, max_grid_size);
                else
                    return InitBFS<ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                            problem, max_grid_size);
            } else {
                if (traversal_mode == 0)
                    return InitBFS<     LBAdvanceKernelPolicy     , FilterKernelPolicy>(
                            problem, max_grid_size);
                else
                    return InitBFS<ForwardAdvanceKernelPolicy     , FilterKernelPolicy>(
                            problem, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    cudaError_t Reset(
        VertexId src,
        FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed
        double subq__factor  = 1.0,
        double subq__factor0 = 1.0,
        double subq__factor1 = 1.0,
        double fullq_factor  = 1.0,
        double fullq_factor0 = 1.0,
        double fullq_factor1 = 1.0,
        double input_factor  = 1.0,
        double outpu_factor  = 1.0,
        double split_factor  = 1.0,
        double temp_factor   = 0.1) // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
    {
         if (Problem::ENABLE_IDEMPOTENCE) {
            if (traversal_mode == 0)
                return ResetBFS<     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                        src, frontier_type, 
                        subq__factor, subq__factor0, subq__factor1,
                        fullq_factor, fullq_factor0, fullq_factor1,
                        input_factor, outpu_factor, split_factor,
                        temp_factor);
            else
                return ResetBFS<ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                        src, frontier_type, 
                        subq__factor, subq__factor0, subq__factor1,
                        fullq_factor, fullq_factor0, fullq_factor1,
                        input_factor, outpu_factor, split_factor,
                        temp_factor);
        } else {
            if (traversal_mode == 0)
                return ResetBFS<     LBAdvanceKernelPolicy     , FilterKernelPolicy>(
                        src, frontier_type, 
                        subq__factor, subq__factor0, subq__factor1,
                        fullq_factor, fullq_factor0, fullq_factor1,
                        input_factor, outpu_factor, split_factor,
                        temp_factor);
            else
                return ResetBFS<ForwardAdvanceKernelPolicy     , FilterKernelPolicy>(
                        src, frontier_type, 
                        subq__factor, subq__factor0, subq__factor1,
                        fullq_factor, fullq_factor0, fullq_factor1,
                        input_factor, outpu_factor, split_factor,
                        temp_factor);
        }
        //return cudaSuccess;
    }    

    cudaError_t Release()
    {
         if (Problem::ENABLE_IDEMPOTENCE) {
            if (traversal_mode == 0)
                return ReleaseBFS
                    <     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>();
            else
                return ReleaseBFS
                    <ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>();
        } else {
            if (traversal_mode == 0)
                return ReleaseBFS
                    <     LBAdvanceKernelPolicy     , FilterKernelPolicy>();
            else
                return ReleaseBFS
                    <ForwardAdvanceKernelPolicy     , FilterKernelPolicy>();
        }
        //return cudaSuccess;
    }    

    /** @} */
};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
