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
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/app/enactor_slice.cuh>
#include <gunrock/app/enactor_thread.cuh>
#include <gunrock/app/enactor_loop.cuh>
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
    clock_t                          start_time          ;

    EnactorStats() :
        iteration        (0),
        total_lifetimes  (0),
        total_runtimes   (0),
        advance_grid_size(0),
        filter_grid_size (0)
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

    enum Status {
        New,
        Assigned,
        Running,
        Finished
    };

    long long iteration;
    int       peer     ;
    Status    status   ;
    int       gpu_num  ;
    SizeT     length   ;
    SizeT     offset   ;
    cudaStream_t stream;
    SizeT     num_vertex_associates;
    SizeT     num_value__associates;
    VertexId *vertices ;
    VertexId *vertex_associates[NUM_VERTEX_ASSOCIATES];
    Value    *value__associates[NUM_VALUE__ASSOCIATES];

    PushRequest() :
        iteration (0),
        peer      (0),
        status    (New),
        gpu_num   (0),
        length    (0),
        offset    (0),
        stream    (0),
        num_vertex_associates(0),
        num_value__associates(0),
        vertices  (0)
    {
        
    }

    void operator=(const PushRequest<VertexId, SizeT, Value,
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> &src)
    {
        iteration = src.iteration;
        peer      = src.peer;
        status    = src.status;
        gpu_num   = src.gpu_num;
        length    = src.length;
        offset    = src.offset;
        stream    = src.stream;
        num_vertex_associates = src.num_vertex_associates;
        num_value__associates = src.num_value__associates;
        vertices  = src.vertices;
        for (SizeT i=0; i<num_vertex_associates; i++)
            vertex_associates[i] = src.vertex_associates[i];
        for (SizeT i=0; i<num_value__associates; i++)
            value__associates[i] = src.value__associates[i];
    }

}; // end of PushRequest

template <
    typename _VertexId, 
    typename _SizeT, 
    typename _Value,
    int _NUM_VERTEX_ASSOCIATES,
    int _NUM_VALUE__ASSOCIATES>
struct MakeOutArray
{
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT;
    typedef _Value    Value;
    static const int NUM_VERTEX_ASSOCIATES = _NUM_VERTEX_ASSOCIATES;
    static const int NUM_VALUE__ASSOCIATES = _NUM_VALUE__ASSOCIATES;

    enum Direction{
        FORWARD = 1,
        BACKWARD = 2
    };

    Direction  direction;
    SizeT      num_elements;
    int        num_vertex_associates;
    int        num_value__associates;
    int        target_gpu;
    VertexId  *keys_in;
    VertexId  *keys_out;
    SizeT     *markers;
    VertexId  *vertex_orgs[NUM_VERTEX_ASSOCIATES];
    Value     *value__orgs[NUM_VALUE__ASSOCIATES];
    VertexId  *vertex_outs[NUM_VERTEX_ASSOCIATES];
    Value     *value__outs[NUM_VALUE__ASSOCIATES];
    int       *forward_partition;
    VertexId  *forward_convertion;
    SizeT     *backward_offset;
    int       *backward_partition;
    VertexId  *backward_convertion;
};

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
    //typedef ThreadSlice ThreadSlice;
    typedef EnactorStats EnactorStats;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
    static const bool NUM_STAGES = 3;
    static const SizeT NUM_VERTEX_ASSOCIATES = _NUM_VERTEX_ASSOCIATES;
    static const SizeT NUM_VALUE__ASSOCIATES = _NUM_VALUE__ASSOCIATES;

    template <typename Type>
    class Array: public util::Array1D<SizeT, Type> {};

    typedef PushRequest<
        VertexId, SizeT, Value, 
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        PRequest;

    typedef typename util::CircularQueue<
        VertexId, SizeT, Value, SIZE_CHECK>
        CircularQueue;
    
    typedef MakeOutArray<VertexId, SizeT, Value,
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        MakeOutArray;

    typedef FrontierAttribute<SizeT> FrontierA;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value> FrontierT;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSlice;
    typedef typename util::CtaWorkProgressLifetime WorkProgress;

    int           num_gpus            ; // Number of GPUs
    int           num_threads         ;
    int          *gpu_idx             ; // GPU indices
    FrontierType  frontier_type       ;
    int           num_vertex_associates; // Number of associate values in VertexId type for each vertex
    int           num_value__associates; // Number of associate values in Value type for each vertex
    int           num_stages          ; // Number of stages
    int           num_input_streams   ;
    int           num_outpu_streams   ;
    int           num_subq__streams   ;
    int           num_fullq_stream    ;
    int           num_split_streams   ;
    bool          using_subq          ;
    void         *enactor_slices      ;
    void         *thread_slices       ;
    Array<std::thread>  threads       ;
 
    //Device properties
    Array<util::CudaProperties>   cuda_props   ;

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
        num_gpus      (_num_gpus),
        gpu_idx       (_gpu_idx ),
        frontier_type (_frontier_type),
        num_vertex_associates(0 ),
        num_value__associates(0 ),
        num_stages           (3 ),
        num_input_streams    (0 ),
        num_outpu_streams    (0 ),
        num_subq__streams    (1 ),
        num_split_streams    (0 ),
        using_subq           (false)
    {
        cuda_props        .SetName("cuda_props"        );
        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
       
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
        }
    }

    /**
     * @brief Destructor
     */
    virtual ~EnactorBase()
    {
        if (enactor_slices != NULL)
        {
            util::GRError("enactor_slices is not NULL", __FILE__, __LINE__);
        }
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
    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename Enactor>
    cudaError_t Init(
        typename Enactor::Problem *problem,
        Enactor *enactor,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256,
        int num_input_streams = 0,
        int num_outpu_streams = 0,
        int num_subq__streams = 0,
        int num_fullq_stream  = 0,
        int num_split_streams = 0)
    {
        typedef EnactorSlice<Enactor> EnactorSlice;
        typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
            ThreadSlice;

        cudaError_t retval = cudaSuccess;
        this -> num_input_streams = num_input_streams;
        this -> num_outpu_streams = num_outpu_streams;
        this -> num_subq__streams = num_subq__streams;
        this -> num_fullq_stream  = num_fullq_stream ;
        this -> num_split_streams = num_split_streams;
        num_threads = ( (num_input_streams > 0) ? 1 : 0)
                    + ( (num_outpu_streams > 0) ? 1 : 0)
                    + ( (num_subq__streams > 0) ? 1 : 0)
                    + ( (num_fullq_stream  + num_split_streams > 0) ? 1: 0);
        
        EnactorSlice *enactor_slices = new EnactorSlice[num_gpus];
        this -> enactor_slices = (void*) enactor_slices;
        
        ThreadSlice  *thread_slices  = new ThreadSlice [num_threads * num_gpus];
        this -> thread_slices  = (void*) thread_slices;

        if (retval = threads      .Allocate(num_threads * num_gpus))
            return retval; 
        
        int thread_counter = 0;
        for (int gpu=0; gpu<num_gpus; gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            if (retval = enactor_slices[gpu].Init(
                num_gpus, gpu, gpu_idx[gpu],
                num_input_streams, num_outpu_streams,
                num_subq__streams, num_fullq_stream ,
                num_split_streams)) return retval;

            for (int stream=0; stream < num_subq__streams + num_fullq_stream ; stream++)
            {
                EnactorStats *enactor_stats_ = (stream < num_subq__streams) ?
                    enactor_slices[gpu].subq__enactor_statses + stream :
                    enactor_slices[gpu].fullq_enactor_stats   + stream - num_subq__streams;
                //initialize runtime stats
                enactor_stats_ -> advance_grid_size 
                    = MaxGridSize(gpu, advance_occupancy, max_grid_size);
                enactor_stats_ -> filter_grid_size  
                    = MaxGridSize(gpu, filter_occupancy , max_grid_size);

                if (retval = enactor_stats_ -> advance_kernel_stats.Setup(
                    enactor_stats_ -> advance_grid_size)) return retval;
                if (retval = enactor_stats_ ->  filter_kernel_stats.Setup(
                    enactor_stats_ -> filter_grid_size )) return retval;
                if (retval = enactor_stats_ -> node_locks    .Allocate(
                    node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> node_locks_out.Allocate(
                    node_lock_size, util::DEVICE)) return retval;
                if (retval = enactor_stats_ -> total_queued  .Allocate(
                    1, util::DEVICE | util::HOST)) return retval;
            }

            for (int i=0; i<4; i++)
            {
                ThreadSlice *thread_slice  = thread_slices + thread_counter;
                thread_slice -> thread_num = thread_counter;
                thread_slice -> gpu_num    = gpu;
                thread_slice -> status     = ThreadSlice::Status::Init;
                thread_slice -> problem    = (void*) problem;
                thread_slice -> enactor    = (void*) enactor;
                if      (i==0 && num_input_streams>0)
                    threads[thread_counter] = std::thread(Input_Thread
                        <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 
                        (void*)thread_slice);
                else if (i==1 && num_outpu_streams>0)
                    threads[thread_counter] = std::thread(Outpu_Thread
                        <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 
                        (void*)thread_slice);
                else if (i==2 && num_subq__streams>0)
                    threads[thread_counter] = std::thread(SubQ__Thread
                        <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
                        (void*)thread_slice);
                else if (i==3 && num_fullq_stream + num_split_streams > 0)
                    threads[thread_counter] = std::thread(FullQ_Thread
                        <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
                        (void*)thread_slice);
                else continue;
                thread_counter ++;
            }
        }
        num_threads = thread_counter;

        return retval;
    }

    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename Enactor>
    cudaError_t Release()
    {
        typedef EnactorSlice<Enactor> EnactorSlice;
        typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
            ThreadSlice;
        cudaError_t retval = cudaSuccess;

        for (int i=0; i<num_threads; i++)
            threads[i].join();

        EnactorSlice *enactor_slices
            = (EnactorSlice*) this->enactor_slices;
        ThreadSlice  *thread_slices
            = (ThreadSlice*)  this->thread_slices;

        for (int gpu=0; gpu<num_gpus; gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            EnactorSlice *enactor_slice = 
                ((EnactorSlice*) enactor_slices) + gpu;

            for (int stream=0; stream < num_subq__streams + num_fullq_stream ; stream++)
            {
                EnactorStats *enactor_stats_ = (stream < num_subq__streams) ?
                    enactor_slice -> subq__enactor_statuses + stream :
                    enactor_slice -> fullq_enactor_status   + stream - num_subq__streams;
                if (retval = enactor_stats_ -> node_locks    .Release()) return retval;
                if (retval = enactor_stats_ -> node_locks_out.Release()) return retval;
                if (retval = enactor_stats_ -> total_queued  .Release()) return retval;
                FrontierA *frontier_attribute_ = (stream < num_subq__streams) ?
                    enactor_slice -> subq__frontier_attributes + stream :
                    enactor_slice -> fullq_frontier_attribute  + stream - num_subq__streams;
                if (retval = frontier_attribute_ -> output_length .Release()) return retval;
                util::CtaWorkProgressLifetime *work_progress_ = (stream < num_subq__streams) ?
                    enactor_slice -> subq__work_progresses + stream :
                    enactor_slice -> fullq_work_progress   + stream - num_subq__streams;
                if (retval = work_progress_ -> HostReset()) return retval;
            }
            enactor_slice -> Release();
        }
        if (retval = cuda_props        .Release()) return retval;
        if (retval = threads           .Release()) return retval;
        delete[] enactor_slices; enactor_slices = NULL;
        delete[] thread_slices;  thread_slices = NULL;
        this->enactor_slices = NULL;
        this->thrad_slices = NULL;
        return retval;
    }

    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename Enactor>
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;

        /*for (int gpu=0;gpu<num_gpus;gpu++)
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
        }*/
        return retval;
    }

    /*template <typename Problem>
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
    }*/

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

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
