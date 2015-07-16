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

    virtual ~EnactorStats()
    {
        Release();
    }

    cudaError_t Init(
        unsigned int advance_grid_size,
        unsigned int filter_grid_size,
        unsigned int node_lock_size)
    {
        cudaError_t retval = cudaSuccess;

        this -> advance_grid_size = advance_grid_size;
        this -> filter_grid_size  = filter_grid_size;

        //initialize runtime stats
        if (retval = advance_kernel_stats.Setup(
            advance_grid_size)) return retval;
        if (retval = filter_kernel_stats.Setup(
            filter_grid_size )) return retval;
        if (retval = node_locks    .Allocate(
            node_lock_size, util::DEVICE)) return retval;
        if (retval = node_locks_out.Allocate(
            node_lock_size, util::DEVICE)) return retval;
        if (retval = total_queued  .Allocate(
            1, util::DEVICE | util::HOST)) return retval;

        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = node_locks    .Release()) return retval;
        if (retval = node_locks_out.Release()) return retval;
        if (retval = total_queued  .Release()) return retval;
        return retval;
    }

    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;
        iteration       = 0;
        total_runtimes  = 0;
        total_lifetimes = 0;
        total_queued[0] = 0;
        total_queued.Move(util::HOST, util::DEVICE);
        return retval;
    }

    template <typename SizeT2>
    void Accumulate(SizeT2 *d_queued, cudaStream_t stream)
    {
        Accumulate_Num<<<1,1,0,stream>>> (
            d_queued, total_queued.GetPointer(util::DEVICE));
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

    virtual ~FrontierAttribute()
    {
        Release();
    }

    cudaError_t Init()
    {
        cudaError_t retval = cudaSuccess;

        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        if (retval = output_length .Release()) return retval;
        return retval;
    }

    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;
        queue_index = 0;
        queue_reset = true;
        queue_length = 0;
        output_length[0] = 0;
        if (retval = output_length.Move(util::HOST, util::DEVICE)) return retval;
        queue_offset = 0;
        has_incoming = false;
        return retval;
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
    cudaEvent_t event;

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
        vertices  (NULL)
    {
        for (SizeT i=0; i<num_vertex_associates; i++)
            vertex_associates[i] = NULL;
        for (SizeT i=0; i<num_value__associates; i++)
            value__associates[i] = NULL;
    }

    virtual ~PushRequest()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        vertices = NULL;
        for (SizeT i=0; i<num_vertex_associates; i++)
            vertex_associates[i] = NULL;
        for (SizeT i=0; i<num_value__associates; i++)
            value__associates[i] = NULL;
        return retval;
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
struct MakeOutHandle
{
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT;
    typedef _Value    Value;
    static const int NUM_VERTEX_ASSOCIATES = _NUM_VERTEX_ASSOCIATES;
    static const int NUM_VALUE__ASSOCIATES = _NUM_VALUE__ASSOCIATES;

    enum Direction{
        NONE    = 0,
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

template <
    typename _VertexId,
    typename _SizeT,
    typename _Value,
    int _NUM_VERTEX_ASSOCIATES,
    int _NUM_VALUE__ASSOCIATES>
struct ExpandIncomingHandle
{
    typedef _VertexId VertexId;
    typedef _SizeT    SizeT;
    typedef _Value    Value;
    static const int NUM_VERTEX_ASSOCIATES = _NUM_VERTEX_ASSOCIATES;
    static const int NUM_VALUE__ASSOCIATES = _NUM_VALUE__ASSOCIATES;

    SizeT     num_elements;
    int       num_vertex_associates;
    int       num_value__associates;
    VertexId *keys_in;
    VertexId *keys_out;
    VertexId *vertex_ins [NUM_VERTEX_ASSOCIATES];
    VertexId *vertex_orgs[NUM_VERTEX_ASSOCIATES];
    Value    *value__ins [NUM_VALUE__ASSOCIATES];
    Value    *value__orgs[NUM_VALUE__ASSOCIATES];
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
    static const int  NUM_STAGES = 3;
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
    
    typedef MakeOutHandle<
        VertexId, SizeT, Value,
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        MakeOutHandle;

    typedef ExpandIncomingHandle<
        VertexId, SizeT, Value,
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        ExpandIncomingHandle;

    typedef PushRequest <
        VertexId, SizeT, Value,
        NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        PushRequest;

    typedef FrontierAttribute<SizeT> FrontierA;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value> FrontierT;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSlice;
    typedef typename util::CtaWorkProgressLifetime WorkProgress;

    void         *problem             ;
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
    bool          using_fullq         ;
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
        printf("EnactorBase() begin.\n");fflush(stdout);
        cuda_props        .SetName("cuda_props"        );
        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
       
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
            printf("Using GPU %d : %s \n", gpu_idx[gpu], cuda_props[gpu].device_props.name);
            fflush(stdout);
        }
        problem = NULL;
        printf("EnactorBase() end.\n");fflush(stdout);
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
        problem = NULL;
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

        printf("EnactorBase Init begin. #input_streams = %d, #outpu_streams = %d, #subq__streams = %d, #fullq_streams = %d, #split_streams = %d\n",
            num_input_streams, num_outpu_streams, num_subq__streams,
            num_fullq_stream , num_split_streams);
        fflush(stdout);

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
        for (int gpu_num=0; gpu_num<num_gpus; gpu_num++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu_num])) return retval;
            EnactorSlice *enactor_slice = enactor_slices + gpu_num;
            if (retval = enactor_slice->Init(
                num_gpus, gpu_num, gpu_idx[gpu_num],
                num_input_streams, num_outpu_streams,
                num_subq__streams, num_fullq_stream ,
                num_split_streams)) return retval;

            for (int stream=0; stream < num_subq__streams + num_fullq_stream ; stream++)
            {
                printf("gpu_num = %d, stream = %d, num_subq__streams = %d, num_fullq_stream = %d\n",
                    gpu_num, stream, num_subq__streams, num_fullq_stream);fflush(stdout);
                EnactorStats *enactor_stats_ = (stream < num_subq__streams) ?
                    enactor_slice->subq__enactor_statses + stream :
                    enactor_slice->fullq_enactor_stats   + stream - num_subq__streams;
                if (retval = enactor_stats_ -> Init(
                    MaxGridSize(gpu_num, advance_occupancy, max_grid_size),
                    MaxGridSize(gpu_num, filter_occupancy , max_grid_size),
                    node_lock_size))
                    return retval;
            }

            typename ThreadSlice::Type thread_type = ThreadSlice::Type::Input;
            while (thread_type != ThreadSlice::Type::Last)
            {
                if ((thread_type == ThreadSlice::Type::Input 
                     && num_input_streams <= 0) ||
                    (thread_type == ThreadSlice::Type::Output 
                     && num_outpu_streams <= 0) ||
                    (thread_type == ThreadSlice::Type::SubQ
                     && num_subq__streams <= 0) ||
                    (thread_type == ThreadSlice::Type::FullQ
                     && num_fullq_stream + num_split_streams <= 0))
                {
                    thread_type = ThreadSlice::IncreatmentType(thread_type); 
                    continue;
                }

                if (retval = thread_slices[thread_counter].Init(
                    thread_counter, gpu_num, problem, enactor, 
                    thread_type, threads[thread_counter]))
                    return retval;
                if (thread_type == ThreadSlice::Type::Input)
                    enactor_slice -> input_thread_slice = thread_slices + thread_counter;
                else if (thread_type == ThreadSlice::Type::Output)
                    enactor_slice -> outpu_thread_slice = thread_slices + thread_counter;
                else if (thread_type == ThreadSlice::Type::SubQ)
                    enactor_slice -> subq__thread_slice = thread_slices + thread_counter;
                else if (thread_type == ThreadSlice::Type::FullQ)
                    enactor_slice -> fullq_thread_slice = thread_slices + thread_counter;
                thread_counter ++;
                thread_type = ThreadSlice::IncreatmentType(thread_type); 
            }
        }
        num_threads = thread_counter;

        for (int thread_num = 0; thread_num < num_threads; thread_num++)
        if (thread_slices[thread_num].status != ThreadSlice::Status::Wait)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        printf("EnactorBase Init end. #threads = %d\n", num_threads);
        fflush(stdout);
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

        printf("EnactorBase Release begin.\n");fflush(stdout);
        EnactorSlice *enactor_slices
            = (EnactorSlice*) this->enactor_slices;
        ThreadSlice  *thread_slices
            = (ThreadSlice*)  this->thread_slices;

        for (int i=0; i<num_threads; i++)
        {
            thread_slices[i].status = ThreadSlice::Status::ToKill;
        }
        for (int i=0; i<num_threads; i++)
        {
            threads[i].join();
        }

        for (int gpu=0; gpu<num_gpus; gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            EnactorSlice *enactor_slice = 
                ((EnactorSlice*) enactor_slices) + gpu;
            enactor_slice -> Release();
        }
        if (retval = cuda_props        .Release()) return retval;
        if (retval = threads           .Release()) return retval;
        delete[] enactor_slices; enactor_slices = NULL;
        delete[] thread_slices;  thread_slices = NULL;
        this->enactor_slices = NULL;
        this->thread_slices = NULL;

        printf("EnactorBase Release end.\n");fflush(stdout);
        return retval;
    }

    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename Enactor>
    cudaError_t Reset(
        FrontierType frontier_type,
        double subq__factor  = 1.0,
        double subq__factor0 = 1.0,
        double subq__factor1 = 1.0,
        double fullq_factor  = 1.0,
        double fullq_factor0 = 1.0,
        double fullq_factor1 = 1.0,
        double input_factor  = 1.0,
        double outpu_factor  = 1.0,
        double split_factor  = 1.0,
        double temp_factor   = 0.1)
    {
        cudaError_t retval = cudaSuccess;
        printf("EnactorBase Reset begin.\n");fflush(stdout);
        
        EnactorSlice<Enactor> *enactor_slices
            = (EnactorSlice<Enactor>*) this->enactor_slices;
        ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>* thread_slices
            = (ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>*) this->thread_slices;
        typename Enactor::Problem *problem = (typename Enactor::Problem*) this->problem;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            if (retval = enactor_slices[gpu].Reset(
                frontier_type,
                problem -> graph_slices[gpu],
                Enactor::Problem::USE_DOUBLE_BUFFER,
                problem -> graph_slices[gpu]->in_counter + 0,
                problem -> graph_slices[gpu]->out_counter + 0,
                subq__factor, subq__factor0, subq__factor1,
                fullq_factor, fullq_factor0, fullq_factor1,
                input_factor, outpu_factor, split_factor,
                temp_factor)) return retval;
        }
        for (int i=0; i<num_threads; i++)
        {
            if (retval = util::SetDevice(gpu_idx[thread_slices[i].gpu_num])) return retval;
            if (retval = thread_slices[i].Reset()) return retval;
        }
        printf("EnactorBase Reset end.\n");fflush(stdout);
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

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
