// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once
#include <time.h>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/circular_queue.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/enactor_kernel.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {

/**
 * @brief Structure for auxiliary variables used in enactor.
 */
struct EnactorStats
{
    long long                        iteration           ;
    unsigned long long               total_lifetimes     ;
    unsigned long long               total_runtimes      ;
    util::Array1D<int, size_t>       total_queued        ;
    unsigned int                     advance_grid_size   ;
    unsigned int                     filter_grid_size    ;
    util::KernelRuntimeStatsLifetime advance_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats ;
    util::Array1D<int, unsigned int> node_locks          ;
    util::Array1D<int, unsigned int> node_locks_out      ;
    cudaError_t                      retval              ;
    clock_t                          start_time          ;

    EnactorStats() :
        iteration        (0),
        total_lifetimes  (0),
        total_runtimes   (0),
        advance_grid_size(0),
        filter_grid_size (0),
        retval           (cudaSuccess)
    {
        node_locks    .SetName("node_locks"    );
        node_locks_out.SetName("node_locks_out");
        total_queued  .SetName("total_queued");
    }

    template <typename SizeT2>
    void Accumulate(SizeT2 *d_queued, cudaStream_t stream)
    {
        Accumulate_Num<<<1,1,0,stream>>> (d_queued, total_queued.GetPointer(util::DEVICE));
    }
};

/**
 * @brief Structure for auxiliary variables used in frontier operations.
 */
template <typename SizeT>
struct FrontierAttribute
{
    SizeT        queue_length ;
    util::Array1D<SizeT,SizeT>
                 output_length;
    unsigned int queue_index  ;
    SizeT        queue_offset ;
    int          selector     ;
    bool         queue_reset  ;
    int          current_label;
    bool         has_incoming ;
    gunrock::oprtr::advance::TYPE
                 advance_type ;

    FrontierAttribute() :
        queue_length (0    ),
        queue_index  (0    ),
        queue_offset (0    ),
        selector     (0    ),
        queue_reset  (false),
        current_label(0    ),
        has_incoming (false)
    {
        output_length.SetName("output_length");
    }
};

template <
    typename VertexId,
    typename SizeT,
    typename Value,
    SizeT    NUM_VERTEX_ASSOCIATES,
    SizeT    NUM_VALUE__ASSOCIATES>
struct PushRequest
{
public:
    int       iteration;
    int       peer     ;
    int       status   ;
    int       gpu_num  ;
    SizeT     length   ;
    SizeT     offset   ;
    SizeT     num_vertex_associates;
    SizeT     num_value__associates;
    VertexId *vertices ;
    VertexId *vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value    *value__associates[Enactor::NUM_VALUE__ASSOCIATES];
}; // end of PushRequest

template <
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES,
    typename Enactor,
    typename Functor,
    typename Iteration>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>  GraphSlice;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_gpus             =   problem     -> num_gpus;
    int           thread_num           =   thread_data -> thread_num;
    DataSlice    *data_slice           =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSlice   *graph_slice          =   problem     -> graph_slices       [thread_num] ;
    GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attribute   = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);  
    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);  
    util::CtaWorkProgressLifetime
                 *work_progress        = &(enactor     -> work_progress      [thread_num * num_gpus]);
    ContextPtr   *context              =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    SizeT         Total_Length         =   0;
    cudaError_t   tretval              =   cudaSuccess;
    int           grid_size            =   0;
    std::string   mssg                 =   "";
    int           pre_stage            =   0;
    size_t        offset               =   0;
    int           iteration            =   0;
    int           selector             =   0;
    util::DoubleBuffer<SizeT, VertexId, Value>
                 *frontier_queue_      =   NULL;
    FrontierAttribute<SizeT>
                 *frontier_attribute_  =   NULL;
    EnactorStats *enactor_stats_       =   NULL;
    util::CtaWorkProgressLifetime
                 *work_progress_       =   NULL;
    util::Array1D<SizeT, SizeT>
                 *scanned_edges_       =   NULL;
    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;
    bool          over_sized;

    printf("Iteration entered\n");fflush(stdout);
    while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
    {
        Total_Length             = 0;
        data_slice->wait_counter = 0;
        tretval                  = cudaSuccess;
        if (num_gpus>1 && enactor_stats[0].iteration>0)
        {
            frontier_attribute[0].queue_reset  = true;
            frontier_attribute[0].queue_offset = 0;
            for (i=1; i<num_gpus; i++)
            {
                frontier_attribute[i].selector     = frontier_attribute[0].selector;
                frontier_attribute[i].advance_type = frontier_attribute[0].advance_type;
                frontier_attribute[i].queue_offset = 0;
                frontier_attribute[i].queue_reset  = true;
                frontier_attribute[i].queue_index  = frontier_attribute[0].queue_index;
                frontier_attribute[i].current_label= frontier_attribute[0].current_label;
                enactor_stats     [i].iteration    = enactor_stats     [0].iteration;
            }
        } else {
            frontier_attribute[0].queue_offset = 0;
            frontier_attribute[0].queue_reset  = true;
        }
        for (peer=0; peer<num_gpus; peer++)
        {
            stages [peer         ] = 0   ; 
            stages [peer+num_gpus] = 0   ;
            to_show[peer         ] = true; 
            to_show[peer+num_gpus] = true;
            for (i=0; i<data_slice->num_stages; i++)
                data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
        }
        
        while (data_slice->wait_counter < num_gpus*2
           && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
        {
       }

        if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
        {
            for (peer_=0;peer_<num_gpus*2;peer_++)
                data_slice->wait_marker[peer_]=0;
            wait_count=0;
            while (wait_count<num_gpus*2-1 &&
                !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
            {
                for (peer_=0;peer_<num_gpus*2;peer_++)
                {
                    if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                        continue;
                    tretval = cudaStreamQuery(streams[peer_]);
                    if (tretval == cudaSuccess)
                    {
                        data_slice->wait_marker[peer_]=1;
                        wait_count++;
                        continue;
                    } else if (tretval != cudaErrorNotReady)
                    {
                        enactor_stats[peer_%num_gpus].retval = tretval;
                        break;
                    }
                }
            }

            if (Enactor::DEBUG) {printf("%d\t %lld\t \t Subqueue finished. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
            grid_size = Total_Length/256+1;
            if (grid_size > 512) grid_size = 512;

            if (Enactor::SIZE_CHECK)
            {
                if (enactor_stats[0]. retval = 
                    Check_Size<true, SizeT, VertexId> ("total_queue", Total_Length, &data_slice->frontier_queues[0].keys[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;
                if (Problem::USE_DOUBLE_BUFFER)
                    if (enactor_stats[0].retval = 
                        Check_Size<true, SizeT, Value> ("total_queue", Total_Length, &data_slice->frontier_queues[0].values[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;

                offset=frontier_attribute[0].queue_length;
                for (peer_=1;peer_<num_gpus;peer_++)
                if (frontier_attribute[peer_].queue_length !=0) {
                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                        data_slice->frontier_queues[0    ].keys[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                        data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                        frontier_attribute[peer_].queue_length);
                    if (Problem::USE_DOUBLE_BUFFER)
                        util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                            data_slice->frontier_queues[0       ].values[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_   ].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                            frontier_attribute[peer_].queue_length);
                    offset+=frontier_attribute[peer_].queue_length;
                }
            }
            frontier_attribute[0].queue_length = Total_Length;
            if (!Enactor::SIZE_CHECK) frontier_attribute[0].selector = 0;
            frontier_queue_ = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpus == 1)?0:num_gpus]);
       }         
        Iteration::Iteration_Change(enactor_stats->iteration);
    }
}
    
/**
 * @brief Base class for graph problem enactors.
 */
template <
    typename _VertexId,
    typename _SizeT,
    typename _Value,
    bool     _DEBUG,  // if DEBUG is set, print details to stdout
    bool     _SIZE_CHECK,
    _SizeT   _NUM_VERTEX_ASSOCIATES,
    _SizeT   _NUM_VALUE__ASSOCIATES>
class EnactorBase
{
public:
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT   ;
    typedef _Value    Value   ;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
    static const SizeT NUM_VERTEX_ASSOCIATES = _NUM_VERTEX_ASSOCIATES;
    static const SizeT NUM_VALUE__ASSOCIATES = _NUM_VALUE__ASSOCIATES;

    template <typename Type>
    class Array: public util::Array1D<SizeT, Type> {};

    typedef typename PushRequest<
        VertexId, SizeT, Value, 
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        PRequest;

    typedef typename CircularQueue<
        VertexId, SizeT, Value, SIZE_CHECK>
        CircularQueue;

    typedef typename FrontierAttribute<SizeT> FrontierA;
    typedef typename util::DoubleBuffer<SizeT, VertexId, Value> FrontierT;

    int           num_gpus            ; // Number of GPUs
    int          *gpu_idx             ; // GPU indices
    FrontierType  frontier_type       ;
    int           num_vertex_associate; // Number of associate values in VertexId type for each vertex
    int           num_value__associate; // Number of associate values in Value type for each vertex
    int           num_stages          ; // Number of stages
    int           num_input_streams   ;
    int           num_outpu_streams   ;
    int           num_subq__streams   ;
    int           num_split_streams   ;
 
    //Device properties
    util::Array1D<SizeT, util::CudaProperties>          cuda_props        ;

    // Queue size counters and accompanying functionality

    FrontierType GetFrontierType() {return frontier_type;}

protected:  

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
     */
    EnactorBase(
        FrontierType   _frontier_type, 
        int            _num_gpus, 
        int           *_gpu_idx) :
        frontier_type (_frontier_type),
        num_gpus      (_num_gpus),
        gpu_idx       (_gpu_idx ),
        num_vertex_associates(0 ),
        num_value__associates(0 ),
        num_stages           (3 ),
        num_input_streams    (0 ),
        num_outpu_streams    (0 ),
        num_subq__streams    (1 ),
        num_split_streams    (0 )
    {
        cuda_props        .SetName("cuda_props"        );
        work_progress     .SetName("work_progress"     );
        enactor_stats     .SetName("enactor_stats"     );
        frontier_attribute.SetName("frontier_attribute");
        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        work_progress     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable); 
        enactor_stats     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        frontier_attribute.Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
            for (int peer=0;peer<num_gpus;peer++)
            {
                work_progress     [gpu*num_gpus+peer].Setup();
                frontier_attribute[gpu*num_gpus+peer].output_length.Allocate(1, util::HOST | util::DEVICE);
            }
        }
    }

    /**
     * @brief Destructor
     */
    virtual ~EnactorBase()
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            for (int peer=0;peer<num_gpus;peer++)
            {
                enactor_stats     [gpu*num_gpus+peer].node_locks    .Release();
                enactor_stats     [gpu*num_gpus+peer].node_locks_out.Release();
                enactor_stats     [gpu*num_gpus+peer].total_queued  .Release();
                frontier_attribute[gpu*num_gpus+peer].output_length .Release();
                if (work_progress [gpu*num_gpus+peer].HostReset()) return;
            }
        }
        work_progress     .Release();
        cuda_props        .Release();
        enactor_stats     .Release();
        frontier_attribute.Release();
    }

   /**
     * @brief Init function for enactor base class
     *
     * @tparam ProblemData
     *
     * @param[in] problem The problem object for the graph primitive
     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
     * @param[in] advance_occupancy CTA Occupancy for Advance operator
     * @param[in] filter_occupancy CTA Occupancy for Filter operator
     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename Problem>
    cudaError_t Init(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    { 
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0;peer<num_gpus;peer++)
            {
                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus + peer;
                //initialize runtime stats
                enactor_stats_ -> advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
                enactor_stats_ -> filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);

                if (retval = enactor_stats_ -> advance_kernel_stats.Setup(enactor_stats_->advance_grid_size)) return retval;
                if (retval = enactor_stats_ ->  filter_kernel_stats.Setup(enactor_stats_->filter_grid_size)) return retval;
                if (retval = enactor_stats_ -> node_locks    .Allocate(node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> total_queued  .Allocate(1, util::DEVICE | util::HOST)) return retval;
            }
        }
        return retval;
    }

    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0; peer<num_gpus; peer++)
            {
                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus + peer;
                enactor_stats_ -> iteration             = 0;
                enactor_stats_ -> total_runtimes        = 0;
                enactor_stats_ -> total_lifetimes       = 0;
                enactor_stats_ -> total_queued[0]       = 0;
                enactor_stats_ -> total_queued.Move(util::HOST, util::DEVICE);
            }
        }
        return retval;
    }

    template <typename Problem>
    cudaError_t Setup(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = Init(problem, max_grid_size, advance_occupancy, filter_occupancy, node_lock_size)) return retval;
        if (retval = Reset()) return retval;
        return retval;
    }

    /**
     * @brief Utility function for getting the max grid size.
     *
     * @param[in] cta_occupancy CTA occupancy for current architecture
     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
     *
     * \return The maximum number of threadblocks this enactor class can launch.
     */
    int MaxGridSize(int gpu, int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            max_grid_size = this->cuda_props[gpu].device_props.multiProcessorCount * cta_occupancy;
        }

        return max_grid_size;
    } 
};

template <
    typename AdvanceKernelPolicy, 
    typename FilterKernelPolicy, 
    typename Enactor,
    bool     _HAS_SUBQ,
    bool     _HAS_FULLQ,
    bool     _BACKWARD,
    bool     _FORWARD,
    bool     _UPDATE_PREDECESSORS>
struct IterationBase
{
public:
    typedef typename Enactor::SizeT      SizeT     ;   
    typedef typename Enactor::Value      Value     ;   
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = _HAS_SUBQ;
    static const bool HAS_FULLQ  = _HAS_FULLQ;
    static const bool BACKWARD   = _BACKWARD;
    static const bool FORWARD    = _FORWARD;
    static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;

    static void SubQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static void SubQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static bool Stop_Condition(
        EnactorStats                  *enactor_stats,
        FrontierAttribute<SizeT>      *frontier_attribute,
        util::Array1D<SizeT, DataSlice> 
                                      *data_slice,
        int                            num_gpus)
    {
        return All_Done(enactor_stats,frontier_attribute,data_slice,num_gpus);
    }

    static void Iteration_Change(long long &iterations)
    {
        iterations++;
    }

    static void Iteration_Update_Preds(
        GraphSlice                    *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT> 
                                      *frontier_attribute,
        util::DoubleBuffer<SizeT, VertexId, Value> 
                                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {
        if (num_elements == 0) return;
        int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && UPDATE_PREDECESSORS && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                frontier_queue->keys[selector].GetPointer(util::DEVICE),
                data_slice    ->preds         .GetPointer(util::DEVICE),
                data_slice    ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                frontier_queue->keys[selector] .GetPointer(util::DEVICE),
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice    ->temp_preds     .GetPointer(util::DEVICE),
                data_slice    ->preds          .GetPointer(util::DEVICE));//,
        }
    }

    static void Check_Queue_Size(
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        GraphSlice                    *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);fflush(stdout);

        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        if (Problem::USE_DOUBLE_BUFFER)
        {
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        } 
    }

    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (num_gpus < 2) return; 
        bool over_sized = false, keys_over_sized = false;
        int peer_ = 0, t=0, i=0;
        size_t offset = 0;
        SizeT *t_out_length = new SizeT[num_gpus];
        int selector = frontier_attribute->selector;
        int block_size = 256;
        int grid_size  = num_elements / block_size;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size=512;
        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);

        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            t_out_length[peer_] = 0;
            data_slice->out_length[peer_] = 0;
        }
        if (num_elements ==0) return;
 
        over_sized = false;
        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            if (enactor_stats->retval = 
                Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> ("keys_marker", num_elements, &data_slice->keys_marker[peer_], over_sized, thread_num, enactor_stats->iteration, peer_)) break;
            if (over_sized) data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
        }
        if (enactor_stats->retval) return;
        if (over_sized) data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        
        for (t=0; t<2; t++)
        {
            if (t==0 && !FORWARD) continue;
            if (t==1 && !BACKWARD) continue;

            if (BACKWARD && t==1) 
                Assign_Marker_Backward<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->backward_offset   .GetPointer(util::DEVICE),
                    graph_slice   ->backward_partition.GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Assign_Marker<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->partition_table   .GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));

            for (peer_=0;peer_<num_gpus;peer_++)
            {
                Scan<mgpu::MgpuScanTypeInc>(
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    num_elements,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    context[0]);
            }

            if (num_elements>0) for (peer_=0; peer_<num_gpus;peer_++)
            {
                cudaMemcpyAsync(&(t_out_length[peer_]),
                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                        + (num_elements -1),
                    sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            } else {
                for (peer_=0;peer_<num_gpus;peer_++)
                    t_out_length[peer_]=0;
            }
            if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

            keys_over_sized = true;
            for (peer_=0; peer_<num_gpus;peer_++)
            {
                if (enactor_stats->retval = 
                    Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId> (
                        "keys_out", 
                        data_slice->out_length[peer_] + t_out_length[peer_], 
                        peer_!=0 ? &data_slice->keys_out[peer_] : 
                                   &data_slice->frontier_queues[0].keys[selector^1], 
                        keys_over_sized, thread_num, enactor_stats[0].iteration, peer_), 
                        data_slice->out_length[peer_]==0? false: true) break;
                if (keys_over_sized) 
                    data_slice->keys_outs[peer_] = peer_==0 ? 
                        data_slice->frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) : 
                        data_slice->keys_out[peer_].GetPointer(util::DEVICE);
                if (peer_ == 0) continue;

                over_sized = false;
                for (i=0;i<NUM_VERTEX_ASSOCIATES;i++)
                {
                    if (enactor_stats[0].retval = 
                        Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId>(
                            "vertex_associate_outs", 
                            data_slice->out_length[peer_] + t_out_length[peer_], 
                            &data_slice->vertex_associate_out[peer_][i], 
                            over_sized, thread_num, enactor_stats->iteration, peer_),
                            data_slice->out_length[peer_]==0? false: true) break;
                    if (over_sized) data_slice->vertex_associate_outs[peer_][i] = data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

                over_sized = false;
                for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
                {
                    if (enactor_stats->retval = 
                        Check_Size<Enactor::SIZE_CHECK, SizeT, Value   >(
                            "value__associate_outs", 
                            data_slice->out_length[peer_] + t_out_length[peer_], 
                            &data_slice->value__associate_out[peer_][i], 
                            over_sized, thread_num, enactor_stats->iteration, peer_,
                            data_slice->out_length[peer_]==0? false: true)) break;
                    if (over_sized) data_slice->value__associate_outs[peer_][i] = data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            }
            if (enactor_stats->retval) break;
            if (keys_over_sized) data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

            offset = 0;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_markers         .GetPointer(util::HOST),
                      sizeof(SizeT*   ) * num_gpus);
            offset += sizeof(SizeT*   ) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_outs            .GetPointer(util::HOST),
                      sizeof(VertexId*) * num_gpus);
            offset += sizeof(VertexId*) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                      sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
            offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> value__associate_orgs.GetPointer(util::HOST),
                      sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
            offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                         data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            }
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                        data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            }
            memcpy(&(data_slice->make_out_array[offset]),
                     data_slice->out_length.GetPointer(util::HOST),
                      sizeof(SizeT) * num_gpus);
            offset += sizeof(SizeT) * num_gpus;
            data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, stream);

            if (BACKWARD && t==1) 
                Make_Out_Backward<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> backward_offset     .GetPointer(util::DEVICE),
                    graph_slice   -> backward_partition  .GetPointer(util::DEVICE),
                    graph_slice   -> backward_convertion .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Make_Out<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> partition_table     .GetPointer(util::DEVICE),
                    graph_slice   -> convertion_table    .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            for (peer_ = 0; peer_<num_gpus; peer_++)
                data_slice->out_length[peer_] += t_out_length[peer_];
        }
        if (enactor_stats->retval) return;                    
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
        frontier_attribute->selector^=1;
        if (t_out_length!=NULL) {delete[] t_out_length; t_out_length=NULL;}
    }

};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
