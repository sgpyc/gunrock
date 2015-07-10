// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_loop.cuh
 *
 * @brief Base Iteration Loop
 */

#pragma once

#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/app/enactor_slice.cuh>

namespace gunrock {
namespace app {

template <
    typename _AdvanceKernelPolicy, 
    typename _FilterKernelPolicy, 
    typename _Enactor>
class ThreadSlice;

template <
    typename AdvanceKernelPolicy, 
    typename FilterKernelPolicy, 
    typename Enactor>
struct IterationBase
{
public:
    enum Status {
        New,
        //Init,
        Running
    };

    typedef typename Enactor::SizeT      SizeT     ;   
    typedef typename Enactor::Value      Value     ;   
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef typename Enactor::GraphSlice GraphSlice;
    typedef typename Enactor::FrontierT  FrontierT ;
    typedef typename Enactor::FrontierA  FrontierA ;
    typedef typename Enactor::WorkProgress WorkProgress;
    typedef typename Enactor::EnactorStats EnactorStats;
    typedef typename Enactor::MakeOutHandle MakeOutHandle;
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::ExpandIncomingHandle ExpandIncomingHandle;
    typedef typename Enactor::PushRequest   PushRequest;
    typedef EnactorSlice<Enactor> EnactorSlice;

    template <typename Type>
    using Array = typename Enactor::Array<Type>;

    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;

    bool              has_subq;
    bool              has_fullq;
    bool              backward;
    bool              forward;
    bool              update_predecessors;

    long long         iteration;
    Status            status;
    int               num_gpus;
    //int               thread_num;
    int               gpu_num;
    int               stream_num;
    int               num_streams;
    int               num_vertex_associates;
    int               num_value__associates;
    SizeT             num_elements;
    SizeT             request_length;
    FrontierT        *frontier_queue;
    Array<SizeT>     *scanned_edge;
    FrontierA        *frontier_attribute;
    EnactorStats     *enactor_stats;
    util::Array1D<SizeT, DataSlice> *data_slice;
    util::Array1D<SizeT, DataSlice> *data_slices;
    GraphSlice       *graph_slice;
    WorkProgress     *work_progress;
    ContextPtr        context;
    ContextPtr       *contexts;
    cudaStream_t      stream;
    cudaStream_t     *streams;
    Array<SizeT>     *t_out_lengths;
    cudaEvent_t      *events;
    cudaEvent_t       wait_event;
    int              *done_markers;
    EnactorSlice     *enactor_slice;
    Enactor          *enactor;

    typename MakeOutHandle::Direction direction;
    int               grid_size;
    int               block_size;
    size_t            shared_size;
    VertexId         *d_keys_in;
    VertexId         *d_keys_out;
    ExpandIncomingHandle *d_e_handle;

    SizeT            *d_offsets;
    VertexId         *d_indices;
    VertexId         *d_in_key_queue;
    SizeT             max_in;
    SizeT             max_out;
    gunrock::oprtr::advance::TYPE advance_type;
    bool              express;

    Array<SizeT*> *markerss;
    Array<SizeT>  *markers;
    Array<MakeOutHandle> *m_handles;
    CircularQueue    *subq__queue;
    CircularQueue    *fullq_queue;
    CircularQueue    *outpu_queue;

    Array<VertexId*> *vertex_associate_orgs;
    Array<Value*   > *value__associate_orgs;

    std::mutex       *rqueue_mutex;
    typename std::list<PushRequest> *request_queue;

    IterationBase() :
        iteration         (0   ),
        status            (Status::New),
        has_subq          (false),
        has_fullq         (false),
        backward          (false),
        forward           (false),
        update_predecessors(false),
        num_gpus          (0),
        //thread_num        (0   ),
        gpu_num           (0   ),
        stream_num        (0   ),
        num_streams       (0   ),
        num_elements      (0   ),
        num_vertex_associates(0),
        num_value__associates(0),
        request_length    (0   ),
        frontier_queue    (NULL),
        scanned_edge      (NULL),
        frontier_attribute(NULL),
        enactor_stats     (NULL),
        data_slice        (NULL),
        data_slices       (NULL),
        graph_slice       (NULL),
        work_progress     (NULL),
        stream            (0   ),
        streams           (NULL),
        direction         (MakeOutHandle::Direction::NONE),
        grid_size         (0   ),
        block_size        (0   ),
        shared_size       (0   ),
        d_keys_in         (NULL),
        d_keys_out        (NULL),
        //array_size        (0   ),
        //d_array           (NULL),
        d_offsets         (NULL),
        d_indices         (NULL),
        d_in_key_queue    (NULL),
        max_in            (0   ),
        max_out           (0   ),
        express           (false),
        t_out_lengths      (NULL)
    {
        printf("IterationBase() begin.\n");fflush(stdout);
        printf("IterationBase() end.\n"); fflush(stdout);
    }

    virtual ~IterationBase()
    {
        printf("~IterationBase begin.\n");fflush(stdout);
        Release();
        printf("~IterationBase end.\n");fflush(stdout);
    }

    cudaError_t Init(
        int  num_gpus, 
        int  num_streams,
        bool has_subq,
        bool has_fullq,
        bool backward,
        bool forward,
        bool update_predecessors)
    {
        cudaError_t retval = cudaSuccess;
        printf("Iteration::Init begin.\n");fflush(stdout);
        this-> num_gpus    = num_gpus;
        this-> num_streams = num_streams;
        this-> has_subq    = has_subq;
        this-> has_fullq   = has_fullq;
        this-> backward    = backward;
        this-> forward     = forward;
        this-> update_predecessors = update_predecessors;
         //t_out_length = new SizeT[num_streams];
        done_markers = new int  [num_streams];
        printf("Iteration::Init end.\n");fflush(stdout);
        return retval;
    }

    cudaError_t Release()
    {
        printf("iteration::Release begin.\n");fflush(stdout);
        cudaError_t retval = cudaSuccess;
        frontier_queue     = NULL;
        scanned_edge       = NULL;
        frontier_attribute = NULL;
        enactor_stats      = NULL;
        data_slice         = NULL;
        data_slices        = NULL;
        graph_slice        = NULL;
        work_progress      = NULL;
        //delete[] t_out_length; t_out_length = NULL;
        t_out_lengths      = NULL;
        d_keys_in          = NULL;
        d_keys_out         = NULL;
        //d_array            = NULL;
        d_offsets          = NULL;
        d_indices          = NULL;
        d_in_key_queue     = NULL;
        delete[] done_markers; done_markers = NULL;
        printf("Iteration::Release end.\n");fflush(stdout);
        return retval;
    }

    virtual cudaError_t SubQueue_Gather () 
    {
        printf("Iteration::SubQueue_Gather default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t Compute_OutputLength() 
    {
        printf("Iteration::Compute_OutputLength default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t SubQueue_Core   () 
    {
        printf("Iteration::SubQueue_Core default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t FullQueue_Gather() 
    {
        printf("Iteration::FullQueue_Gather default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t FullQueue_Core  () 
    {
        printf("Iteration::FullQueue_Core default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t Expand_Incoming () 
    {
        printf("Iteration::Expand_Incoming default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual cudaError_t End_Action      ()
    {
        printf("Iteration::End_Action default called.\n");fflush(stdout);
        return cudaSuccess;
    }

    virtual bool        Stop_Condition  ()
    {
        return All_Done<ThreadSlice <AdvanceKernelPolicy, FilterKernelPolicy, Enactor> >(enactor, gpu_num);
    }

    virtual cudaError_t Iteration_Change (long long &iterations)
    {
        printf("Iteration::Iteration_change default called.\n");fflush(stdout);
        iterations++;
        return cudaSuccess;
    }

    virtual cudaError_t Iteration_Update_Preds()
    {
        cudaError_t retval = cudaSuccess;

        if (num_elements == 0) return retval;
        //int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && update_predecessors && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                d_keys_in,
                data_slice[0] ->preds         .GetPointer(util::DEVICE),
                data_slice[0] ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                d_keys_in,
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice[0] ->temp_preds     .GetPointer(util::DEVICE),
                data_slice[0] ->preds          .GetPointer(util::DEVICE));//,
        }
        return retval;
    }

    virtual cudaError_t Check_Queue_Size()
    {
        cudaError_t retval     = cudaSuccess;
        bool        over_sized = false;
        int         selector   = frontier_attribute->selector;
        int         iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
        {
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                gpu_num, iteration, stream_num,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);
            fflush(stdout);
        }

        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", request_length, &frontier_queue->keys  [selector^1], 
            over_sized, gpu_num, iteration, stream_num, false)) 
            return retval;
        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", request_length, &frontier_queue->keys  [selector  ],
            over_sized, gpu_num, iteration, stream_num, true )) 
            return retval;
        if (Problem::USE_DOUBLE_BUFFER)
        {
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", request_length, &frontier_queue->values[selector^1],
                over_sized, gpu_num, iteration, stream_num, false)) 
                return retval;
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", request_length, &frontier_queue->values[selector  ], 
                over_sized, gpu_num, iteration, stream_num, true )) 
                return retval;
        }
        return retval; 
    }

    virtual cudaError_t Make_Output()
    {
        cudaError_t retval          = cudaSuccess;
        bool        over_sized      = false;
        //bool        keys_over_sized = false;
        int         stream_num      = 0;
        //int         t=0, i=0;
        //size_t      offset          = 0;
        int         start_peer      = 0;
        int         target_num_streams = 0;
        //int         selector        = frontier_attribute -> selector;
        int         block_size      = 512;
        int         grid_size       = num_elements / block_size;
        DataSlice*  data_slice      = this->data_slice[0] + 0;
        CircularQueue *t_queue      = NULL;
        SizeT       t_offset        = 0;
        int         stream_counter  = 0;
        MakeOutHandle *m_handle       = NULL;
        cudaEvent_t event;
        PushRequest push_request;

        printf("Iteration::Make_Output begin. gpu_num = %d, num_elements = %d\n", 
            gpu_num, num_elements);fflush(stdout);
        //typename Enactor::Array<SizeT>  *markers  =  enactor_slice -> split_markers;
        //typename Enactor::Array<SizeT*> *markerss = &enactor_slice -> split_markerss;
        //cudaEvent_t *events         = enactor_slice -> split_events + 0;

        if (num_gpus < 2) return retval;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size=512;

        for (stream_num = 0; stream_num < num_streams; stream_num++)
        {
            t_out_lengths[0][stream_num] = 0;
            //data_slice->out_length[peer_] = 0;
        }
        if (num_elements ==0) return retval;
 
        over_sized = false;
        for (stream_num = 0; stream_num < num_streams; stream_num++)
        {
            if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
                "keys_marker", num_elements, markers + stream_num, 
                over_sized, gpu_num, iteration, stream_num)) return retval;
            if (over_sized) 
                markerss[0][stream_num] = markers[stream_num].GetPointer(util::DEVICE);
        }
        if (over_sized) 
            markerss->Move(util::HOST, util::DEVICE, num_streams, 0, stream);
       
        if (retval = util::GRError(cudaStreamWaitEvent(
            streams[0], wait_event, 0), 
            "cudaStreamWaitEvent failed", __FILE__, __LINE__)) return retval; 

        start_peer = 0;
        while (start_peer < num_gpus)
        {
            target_num_streams = (start_peer + num_streams <= num_gpus) ?
                num_streams : num_gpus - start_peer;
            //for (stream_num = 0; stream_num < target_num_streams; stream_num++)
            //    util::MemsetKernel<<<256, 256, 0, streams[0]>>>(
            //        markerss[0][stream_num], (SizeT)0);

            Assign_Marker<VertexId, SizeT, MakeOutHandle>
                <<<grid_size, block_size, num_streams * sizeof(SizeT*), 
                streams[0]>>> (
                direction,
                num_elements,
                target_num_streams,
                start_peer,
                d_keys_in,
                graph_slice -> partition_table   .GetPointer(util::DEVICE),
                graph_slice -> backward_offset   .GetPointer(util::DEVICE),
                graph_slice -> backward_partition.GetPointer(util::DEVICE),
                markerss    -> GetPointer(util::DEVICE));

            if (target_num_streams > 1)
            {
                if (retval = util::GRError(cudaEventRecord(events[0], streams[0]),
                    "cudaEventRecord failed", __FILE__, __LINE__))
                    return retval;
            }

            for (stream_num=0; stream_num<target_num_streams; stream_num++)
            {
                if (stream_num > 0)
                {
                    if (retval = util::GRError(cudaStreamWaitEvent(
                        streams[stream_num], events[0], 0),
                        "cudaStreamWaitEvent failed", __FILE__, __LINE__))
                        return retval;
                }
                Scan<mgpu::MgpuScanTypeInc>(
                    markerss[0][stream_num],
                    num_elements,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    markerss[0][stream_num],
                    contexts[0][stream_num]);

                cudaMemcpyAsync(t_out_lengths[0] + stream_num,
                    markerss[0][stream_num] + num_elements -1,
                    sizeof(SizeT), cudaMemcpyDeviceToHost, streams[stream_num]);
                done_markers[stream_num] = 0;
                if (retval = util::GRError( cudaEventRecord(
                    events[stream_num], streams[stream_num]),
                    "cudaEventRecord failed", __FILE__, __LINE__))
                    return retval;
            }

            stream_counter = 0;
            while (stream_counter < target_num_streams)
            {
                for (stream_num=0; stream_num<target_num_streams; stream_num++)
                {
                    if (done_markers[stream_num] == 1) continue;
                    retval = cudaEventQuery(events[stream_num]);
                    if (retval == cudaErrorNotReady)
                    {
                        retval = cudaSuccess; continue;
                    }
                    if (retval != cudaSuccess) return retval;

                    done_markers[stream_num] = 1; stream_counter ++;
                    m_handle = m_handles[0] + stream_num;
                    m_handle -> direction    = direction;
                    m_handle -> num_elements = num_elements;
                    m_handle -> target_gpu   = stream_num + start_peer;
                    m_handle -> keys_in      = d_keys_in;
                    m_handle -> markers      = markerss[0][stream_num];
                    m_handle -> forward_partition   = graph_slice -> 
                        partition_table    .GetPointer(util::DEVICE);
                    m_handle -> forward_convertion  = graph_slice -> 
                        convertion_table   .GetPointer(util::DEVICE);
                    m_handle -> backward_offset     = graph_slice ->
                        backward_offset    .GetPointer(util::DEVICE);
                    m_handle -> backward_partition  = graph_slice -> 
                        backward_partition .GetPointer(util::DEVICE);
                    m_handle -> backward_convertion = graph_slice ->
                        backward_convertion.GetPointer(util::DEVICE);

                    if (stream_num + start_peer == 0)
                    { // current GPU
                        if (has_subq)
                            t_queue = subq__queue;
                        else t_queue = fullq_queue;
                        if (retval = t_queue -> Push_Addr(
                            t_out_lengths[0][stream_num],
                            m_handle -> keys_out, t_offset))
                            return retval;
                        
                        m_handle -> num_vertex_associates = 0;
                        m_handle -> num_value__associates = 0;
                    } else { // peer GPU
                        t_queue = outpu_queue;
                        if (retval = t_queue -> Push_Addr(
                            t_out_lengths[0][stream_num],
                            m_handle -> keys_out, t_offset,
                            num_vertex_associates, num_value__associates,
                            m_handle -> vertex_outs, m_handle -> value__outs))
                            return retval;
                        m_handle -> num_vertex_associates = num_vertex_associates;
                        m_handle -> num_value__associates = num_value__associates;
                        memcpy(m_handle -> vertex_orgs, vertex_associate_orgs, 
                            sizeof(VertexId*) * num_vertex_associates);
                        memcpy(m_handle -> value__orgs, value__associate_orgs,
                            sizeof(VertexId*) * num_value__associates);
                    } // end of if

                    if (retval = m_handles-> Move(util::HOST, util::DEVICE,
                        1, stream_num, streams[stream_num]))
                        return retval;
                    Make_Out<MakeOutHandle>
                        <<<grid_size, block_size, 0,
                        streams[stream_num]>>> (
                        m_handles->GetPointer(util::DEVICE) + stream_num);
                    if (retval = t_queue -> EventSet(0, t_offset,
                        t_out_lengths[0][stream_num])) return retval;

                    if (stream_num + start_peer != 0)
                    {
                        event = enactor_slice->empty_events.back();
                        enactor_slice -> empty_events.pop_back();
                        if (retval = util::GRError(cudaEventRecord(event,
                            streams[stream_num]), "cudaEventRecord failed", 
                            __FILE__, __LINE__)) return retval;
                        push_request.event = event;
                        push_request.iteration = iteration;
                        push_request.peer = stream_num + start_peer;
                        push_request.gpu_num = gpu_num;
                        push_request.length = t_out_lengths[0][stream_num];
                        push_request.offset = t_offset;
                        push_request.num_vertex_associates = num_vertex_associates;
                        push_request.num_value__associates = num_value__associates;
                        push_request.vertices = m_handle -> keys_out;
                        for (int i=0; i<num_vertex_associates; i++)
                            push_request.vertex_associates[i] =
                                m_handle -> vertex_outs[i];
                        for (int i=0; i<num_value__associates; i++)
                            push_request.value__associates[i] =
                                m_handle -> value__outs[i];
                        push_request.status = PushRequest::Status::Assigned;

                        rqueue_mutex -> lock();
                        request_queue -> push_front(push_request);
                        rqueue_mutex -> unlock();
                    }
                } // end of for stream_num
            } // end of while stream_counter

            start_peer += num_streams;
        } // end of while start_peer
  
        printf("Iteration::Make_Output end. gpu_num = %d\n", gpu_num);fflush(stdout); 
        return retval;
    }

    void PrintMessage(const char* message, long long iteration = -1)
    {
        if (Enactor::DEBUG)
            util::cpu_mt::PrintMessage(message, gpu_num,
            iteration, stream_num);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
