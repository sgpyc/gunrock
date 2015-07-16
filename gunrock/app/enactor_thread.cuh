// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_thread.cuh
 *
 * @brief input, output, subq and fullq threads of enactor
 */

#pragma once

#include <list>
#include <thread>
#include <gunrock/app/enactor_loop.cuh>

namespace gunrock {
namespace app {

/**
 * @brief Structure for per-thread variables used in sub-threads.
 */
template<
    typename _AdvanceKernelPolicy,
    typename _FilterKernelPolicy,
    typename _Enactor>
class ThreadSlice
{    
public:
    typedef _AdvanceKernelPolicy AdvanceKernelPolicy;
    typedef _FilterKernelPolicy  FilterKernelPolicy;
    typedef _Enactor Enactor;
    typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
            ThreadSlice_;
    typedef typename Enactor::PRequest      PRequest;
    typedef typename std::list<PRequest*>   PRList;
    typedef typename PRList::iterator       PRIterator;
    typedef EnactorSlice<Enactor> EnactorSlice;
    typedef typename Enactor::Problem       Problem      ;
    typedef typename Problem::DataSlice     DataSlice    ;
    typedef typename Enactor::GraphSlice    GraphSlice   ;
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::SizeT         SizeT        ;
    typedef typename Enactor::VertexId      VertexId     ;
    typedef typename Enactor::Value         Value        ;
    template <typename Type>
    using Array = typename Enactor::Array<Type>;
    typedef typename Enactor::FrontierA     FrontierA    ;
    typedef typename Enactor::FrontierT     FrontierT    ;
    typedef typename Enactor::WorkProgress  WorkProgress ;
    typedef typename Enactor::EnactorStats  EnactorStats ;
    typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, 
        Enactor> IterationT;
    typedef typename Enactor::ExpandIncomingHandle ExpandIncomingHandle;
    typedef typename Enactor::MakeOutHandle MakeOutHandle;

    enum Status {
        New,
        Inited,
        Start,
        Wait,
        Running,
        Ideal,
        ToKill,
        Ended
    };

    enum Type {
        Input,
        Output,
        SubQ,
        FullQ,
        Last
    };

    static Type IncreatmentType(Type src)
    {
        switch (src)
        {
        case Type::Input  : return Type::Output;
        case Type::Output : return Type::SubQ  ;
        case Type::SubQ   : return Type::FullQ ;
        case Type::FullQ  : return Type::Last  ;
        case Type::Last   : return Type::Input ;
        }
        return Type::Last;
    }


    int           thread_num ;
    Type          thread_type;
    int           gpu_num    ;   
    long long     iteration  ;   
    //int           init_size  ;
    Status        status     ;   
    Problem      *problem    ;   
    Enactor      *enactor    ; 
    cudaError_t   retval     ;   
    util::cpu_mt::CPUBarrier
                 *cpu_barrier;
    IterationBase <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
                 *iteration_loops;

    ThreadSlice() :
        thread_num (0   ),  
        thread_type(Type::Last),  
        gpu_num    (0   ),  
        iteration  (0   ),  
        //init_size  (0   ),  
        status     (New ),  
        problem    (NULL),
        enactor    (NULL),
        retval     (cudaSuccess),
        cpu_barrier(NULL),
        iteration_loops(NULL)
    {
        ShowDebugInfo("ThreadSlice() begin.");
        ShowDebugInfo("ThreadSlice() end.");
    }    

    void ShowDebugInfo(
        const char* message, 
        int stream_num = -1, 
        long long iteration = -1)
    {
        char str[526];
        if (!Enactor::DEBUG) return;
        switch (thread_type)
        {
        case Type::Input  : strcpy(str, "InputThread: "); break;
        case Type::Output : strcpy(str, "OutpuThread: "); break;
        case Type::SubQ   : strcpy(str, "SubQ_Thread: "); break;
        case Type::FullQ  : strcpy(str, "FullQThread: "); break;
        default           : strcpy(str, "UnknoThread: "); break;
        }
        if (iteration == -1) iteration = this -> iteration;
        strcpy(str + 13, message);
        util::cpu_mt::PrintMessage(str, gpu_num, iteration, stream_num);
    }

    virtual ~ThreadSlice()
    {
        ShowDebugInfo("~ThreadSlice() begin.");
        Release();
        ShowDebugInfo("~ThreadSlice() end.");
    }

    cudaError_t Init(
        int      thread_num,
        int      gpu_num,
        Problem *problem,
        Enactor *enactor,
        Type thread_type,
        std::thread &thread)
    {
        cudaError_t retval = cudaSuccess;
        ShowDebugInfo("Init() begin.");
        this -> thread_num = thread_num;
        this -> gpu_num    = gpu_num;
        this -> status     = Status::Inited;
        this -> problem    = problem;
        this -> enactor    = enactor;
        this -> thread_type = thread_type;

        switch (thread_type)
        {
        case Input :
            thread = std::thread(Input_Thread, this);
            break;
        case Output:
            thread = std::thread(Outpu_Thread, this);
            break;
        case SubQ:
            thread = std::thread(SubQ__Thread, this);
            break;
        case FullQ:
            thread = std::thread(FullQ_Thread, this);
            break;
        default:
            break;
        }
        ShowDebugInfo("Init() end.");
        return retval;
    }

    cudaError_t Release()
    {
        ShowDebugInfo("Release() begin.");
        cudaError_t retval = cudaSuccess;    
        problem     = NULL;
        enactor     = NULL;
        cpu_barrier = NULL;
        iteration_loops = NULL;
        ShowDebugInfo("Release() end.");
        return retval;
    }    

    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess; 
        ShowDebugInfo("Reset() begin.");
        if (thread_type == Type::Input)
            iteration = 1;
        else
            iteration = 0;
        status = Status::Wait;
        ShowDebugInfo("Reset() end.");
        return retval;
    }

static void Outpu_Thread(ThreadSlice_ *thread_slice)
{
    Enactor      *enactor          =   thread_slice  -> enactor;
    int           gpu_num          =   thread_slice  -> gpu_num;
    int           num_gpus         =   enactor -> num_gpus;
    EnactorSlice *enactor_slice    = ((EnactorSlice*) enactor -> enactor_slices) + gpu_num;
    PRList       *request_queue    = &(enactor_slice -> outpu_request_queue);
    std::mutex   *rqueue_mutex     = &(enactor_slice -> outpu_request_mutex); 
    cudaStream_t *streams          =   enactor_slice -> outpu_streams + 0;
    int           num_streams      =   enactor_slice -> num_outpu_streams;
    int           stream_selector  = 0;
    //cudaStream_t  stream           = 0;
    PRIterator    it, it_;
    PRequest     *push_request;

    thread_slice -> ShowDebugInfo("Thread begin.");
    if (thread_slice -> retval = util::SetDevice(enactor->gpu_idx[gpu_num]))
        return;
 
    thread_slice -> status = ThreadSlice::Status::Wait;
    while (thread_slice -> status != ThreadSlice::Status::ToKill)
    {
        if (thread_slice -> retval) break;
        if (thread_slice -> status == ThreadSlice::Status::Wait
            || request_queue->empty())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        rqueue_mutex->lock();
        it = request_queue->begin();
        while (it != request_queue->end())
        {
            it_ = it; it++;
            push_request = (*it_);
            request_queue -> erase(it_);

            rqueue_mutex->unlock();
            push_request -> stream = streams[stream_selector];
            stream_selector ++;
            stream_selector = stream_selector % num_streams;
            if (push_request -> peer < num_gpus)
            { // local peer
                if (thread_slice -> retval = 
                    PushNeibor<Enactor> (push_request, enactor))
                    break;
            } else { // remote peer
            }
            rqueue_mutex->lock();

            enactor_slice -> outpu_empty_queue.push_front(push_request);
        }
        if (!thread_slice -> retval)
            rqueue_mutex->unlock();
    } // end of while

    thread_slice -> ShowDebugInfo("Thread end.");
} // end of outpu thread

static void Input_Thread(ThreadSlice_ *thread_slice)
{
    Problem       *problem          =   thread_slice  -> problem;
    Enactor       *enactor          = thread_slice  -> enactor;
    int            gpu_num          =   thread_slice  -> gpu_num;
    util::Array1D<SizeT, DataSlice> 
                  *data_slice       = &(problem       -> data_slices[gpu_num]);
    EnactorSlice  *enactor_slice    = ((EnactorSlice*) enactor -> enactor_slices) + gpu_num;
    CircularQueue *input_queues     =   enactor_slice -> input_queues;
    int            num_streams      =   enactor_slice -> num_input_streams;
    cudaStream_t  *streams          =   enactor_slice -> input_streams + 0;
    int            target_input_count = enactor_slice -> input_target_count;
    Array<ExpandIncomingHandle> 
                  *e_handles        = &(enactor_slice -> input_e_handles);
    //long long     *iteration_       = &(thread_slice  -> iteration); 
    long long      iteration        = 0;
    //VertexId      *s_vertices       = NULL;
    //VertexId      *t_vertices       = NULL;
    //VertexId      *s_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    //Value         *s_value__associates[Enactor::NUM_VALUE__ASSOCIATES];
    //SizeT          s_size_soli      = 0;
    //SizeT          s_size_occu      = 0;
    SizeT          length           = 0;
    SizeT          s_offset         = 0;
    SizeT          t_offset         = 0;
    SizeT          grid_size        = 0;
    //SizeT          min_length       = 1;
    //SizeT          max_length       = 32 * 1024 * 1024;
    cudaStream_t   stream           = 0;
    int            stream_selector  = 0;
    CircularQueue *s_queue          = NULL;
    CircularQueue *t_queue          = NULL;
    //SizeT          e_offset         = 0;
    ExpandIncomingHandle *e_handle  = NULL;
    //VertexId     **o_vertex_associates   = NULL;
    //Value        **o_value__associates   = NULL;
    SizeT          num_vertex_associates = 0;
    SizeT          num_value__associates = 0;
    IterationT    *iteration_loop   = NULL;
    int            s_input_count    = 0;
    //bool           to_show          = true;
    cudaError_t    tretval          = cudaSuccess;
    char           mssg[512];

    thread_slice -> ShowDebugInfo("Thread begin.");
    if (thread_slice -> retval = util::SetDevice(enactor->gpu_idx[gpu_num]))
        return;
    thread_slice -> status = ThreadSlice::Status::Wait;
    while (thread_slice -> status != ThreadSlice::Status::ToKill)
    {
        if (thread_slice -> retval) return;
        if (thread_slice -> status == ThreadSlice::Status::Wait)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        //if ((enactor -> using_subq && 
        //     thread_slice -> iteration 
        //       != ((ThreadSlice_*)(enactor_slice -> subq__thread_slice)) -> iteration) ||
        //    (!enactor -> using_subq &&
        //     thread_slice -> iteration
        //       != ((ThreadSlice_*)(enactor_slice -> fullq_thread_slice)) -> iteration))
        //{
        //    std::this_thread::sleep_for(std::chrono::microseconds(1));
        //    continue;
        //}

        iteration = thread_slice->iteration;
        s_queue   = &(input_queues[iteration%2]);
        //to_show = true;
        e_handle =  e_handles[0] + stream_selector;
        iteration_loop = ((IterationT*)enactor_slice -> input_iteration_loops) + stream_selector;
        t_queue = (enactor -> using_subq) ? &enactor_slice -> subq__queue :
            &enactor_slice -> fullq_queue;
        stream  = streams[stream_selector];
        num_vertex_associates = enactor -> num_vertex_associates;
        num_value__associates = enactor -> num_value__associates;

        tretval = cudaErrorNotReady;
        while (tretval == cudaErrorNotReady && 
            thread_slice -> status != ThreadSlice::Status::ToKill)
        {
            tretval = s_queue->Pop_Addr(
                enactor_slice -> input_min_length, 
                enactor_slice -> input_max_length, 
                e_handle -> keys_in, 
                length, s_offset, stream,
                num_vertex_associates, num_value__associates,
                e_handle -> vertex_ins, e_handle -> value__ins,
                false, true, target_input_count);
            if (tretval == cudaErrorNotReady)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
        if (thread_slice -> status == ThreadSlice::Status::ToKill)
            continue;
        if (tretval)
        {
            thread_slice -> retval = tretval;
            return;
        }

        if (length < enactor_slice -> input_min_length)
        {
            s_input_count = s_queue->GetOutputCount();
            if (enactor->using_subq)
            {
                enactor_slice -> subq__target_count[iteration%2]
                    = s_input_count;
                if (length != 0)
                    enactor_slice -> subq__target_count[iteration%2] ++;
                sprintf(mssg, "subq__target_count[%lld] -> %d",
                    iteration%2,
                    enactor_slice -> subq__target_count[iteration%2]);
                thread_slice -> ShowDebugInfo(mssg);
                enactor_slice -> subq__target_set  [iteration%2]
                    = true;
            } else {
                enactor_slice -> fullq_target_count[iteration%2]
                    = s_input_count;
                if (length != 0)
                    enactor_slice -> fullq_target_count[iteration%2] ++;
                sprintf(mssg, "fullq_target_count[%lld] -> %d",
                    iteration%2,
                    enactor_slice -> fullq_target_count[iteration%2]);
                thread_slice -> ShowDebugInfo(mssg);
                enactor_slice -> fullq_target_set  [iteration%2]
                    = true;
            }
            //s_queue -> ResetCounts();
            s_queue -> ChangeInputCount(0 - target_input_count);
            s_queue -> ResetOutputCount();
            if (thread_slice -> retval = iteration_loop -> 
                Iteration_Change(thread_slice -> iteration))
                return;
            thread_slice -> ShowDebugInfo("iteration changed");
            //to_show = true;
            if (length == 0) 
            {
                if (thread_slice -> retval = 
                    s_queue -> EventFinish(1, s_offset, length))
                    return;
                continue;
            }
        }

        if (thread_slice -> retval = t_queue->Push_Addr(
            length, e_handle -> keys_out, t_offset)) return;

        e_handle -> num_elements = length;
        e_handle -> num_vertex_associates = num_vertex_associates;
        e_handle -> num_value__associates = num_value__associates;
        for (int i=0; i<num_vertex_associates; i++)
            e_handle -> vertex_orgs[i] 
                = enactor_slice -> vertex_associate_orgs[i];
        for (int i=0; i<num_value__associates; i++)
            e_handle -> value__orgs[i]
                = enactor_slice -> value__associate_orgs[i];
        e_handles->Move(util::HOST, util::DEVICE, 1, stream_selector, stream);

        sprintf(mssg, "GotInput, length = %d, "
            "vertex_associates = %d, %p, %p, value_associates = %d, %p, %p",
            length,
            num_vertex_associates, 
            num_vertex_associates > 0 ? e_handle -> vertex_ins [0] : NULL, 
            num_vertex_associates > 0 ? e_handle -> vertex_orgs[0] : NULL,
            num_value__associates,
            num_value__associates > 0 ? e_handle -> value__ins [0] : NULL,
            num_value__associates > 0 ? e_handle -> value__orgs[0] : NULL);
        thread_slice -> ShowDebugInfo(mssg, iteration, stream_selector); 
        
        grid_size = length/256+1;
        if (grid_size>512) grid_size=512;
        iteration_loop -> num_vertex_associates = num_vertex_associates;
        iteration_loop -> num_value__associates = num_value__associates;
        iteration_loop -> grid_size             = grid_size;
        iteration_loop -> block_size            = 256;
        iteration_loop -> stream                = stream;
        iteration_loop -> num_elements          = length;
        iteration_loop -> data_slice            = data_slice;
        iteration_loop -> d_e_handle            = e_handles -> GetPointer(util::DEVICE) + stream_selector;

        if (thread_slice -> retval = iteration_loop -> Expand_Incoming())
            return;

        if (thread_slice -> retval = 
            s_queue -> EventSet(1, s_offset, length, stream))
            return;
        if (thread_slice -> retval = 
            t_queue -> EventSet(0, t_offset, length, stream))
            return;

        stream_selector++;
        if (stream_selector >= num_streams) stream_selector = 0;
    } // end of while

    thread_slice -> ShowDebugInfo("Thread end.");
} // end of input thread

static void SubQ__Thread(ThreadSlice_ *thread_slice)
{
    Problem       *problem          =   thread_slice  -> problem;
    Enactor       *enactor          =   thread_slice  -> enactor;
    int            gpu_num          =   thread_slice  -> gpu_num;
    int            thread_num       =   thread_slice  -> thread_num;
    util::Array1D<SizeT, DataSlice> 
                  *data_slice       =   problem       -> data_slices + gpu_num;
    GraphSlice    *graph_slice      =   problem       -> graph_slices [gpu_num];
    EnactorSlice  *enactor_slice    = ((EnactorSlice*) 
                                        enactor -> enactor_slices) + gpu_num;
    CircularQueue *s_queue          = &(enactor_slice -> subq__queue);
    int            num_streams      =   enactor_slice -> num_subq__streams;
    cudaStream_t  *streams          =   enactor_slice -> subq__streams   + 0;
    bool          *to_shows         =   enactor_slice -> subq__to_shows  + 0;
    int           *stages           =   enactor_slice -> subq__stages    + 0;
    FrontierT     *frontiers        =   enactor_slice -> subq__frontiers + 0;
    ContextPtr    *contexts         =   enactor_slice -> subq__contexts  + 0;
    EnactorStats  *enactor_statses  =   enactor_slice -> subq__enactor_statses + 0;
    FrontierA     *frontier_attributes 
                                    =   enactor_slice -> subq__frontier_attributes + 0;
    WorkProgress  *work_progresses  =   enactor_slice -> subq__work_progresses + 0;
    long long     *stream_iterations =  enactor_slice -> subq__iterations + 0; 
    Array<SizeT>  *scanned_edges    =   enactor_slice -> subq__scanned_edges;
    SizeT         *s_lengths          = enactor_slice -> subq__s_lengths + 0;
    SizeT         *s_offsets          = enactor_slice -> subq__s_offsets + 0;
    int            stream_num         = 0;
    long long      iteration          = 0;
    //long long      iteration_         = 0;
    std::string    mssg               = "";
    IterationT    *iteration_loops    = NULL;
    IterationT    *iteration_loop     = NULL;
    cudaStream_t   stream             = 0;
    FrontierA     *frontier_attribute = NULL;
    FrontierT     *frontier           = NULL;
    EnactorStats  *enactor_stats      = NULL;
    Array<SizeT>  *scanned_edge       = NULL;
    WorkProgress  *work_progress      = NULL;
    int            selector           = 0;
    bool           over_sized         = false;
    CircularQueue *t_queue            = NULL;
    VertexId      *vertex_array       = NULL;
    Value         *value__array       = NULL;
    SizeT          t_offset           = 0;
    int            pre_stage          = 0;
    bool           show_wait          = true;
    cudaError_t    tretval            = cudaSuccess;
    char           cmssg[512];
 
    thread_slice -> ShowDebugInfo("Thread begin.");
    if (thread_slice -> retval = util::SetDevice(enactor->gpu_idx[gpu_num]))
        return;
    thread_slice -> status = ThreadSlice::Status::Wait;
    while (thread_slice -> status != ThreadSlice::Status::ToKill)
    {
        if (thread_slice -> retval) return;
        if (thread_slice -> status == ThreadSlice::Status::Wait 
            || !enactor -> using_subq) 
        {
            if (show_wait)
            {
                thread_slice -> ShowDebugInfo("Waiting...");
                show_wait = false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        show_wait = true;
        iteration  = thread_slice -> iteration;
        //iteration_ = iteration % 4;
        iteration_loops = (IterationT*)enactor_slice -> subq__iteration_loops;
        if (enactor -> num_gpus > 1 || enactor -> using_fullq)
        {
            t_queue = &enactor_slice -> fullq_queue;
        } else {
            t_queue = &enactor_slice -> subq__queue;
        }

        for (stream_num=0; stream_num < num_streams; stream_num ++)
        {
            stream              = streams[stream_num];
            iteration_loop      = iteration_loops     + stream_num;
            frontier_attribute  = frontier_attributes + stream_num;
            frontier            = frontiers           + stream_num;
            scanned_edge        = scanned_edges       + stream_num;
            work_progress       = work_progresses     + stream_num;
            enactor_stats       = enactor_statses     + stream_num; 
            selector            = frontier_attribute -> selector;
            enactor_stats->iteration = thread_slice -> iteration;
            
            //if (stages[stream_num] == 0)
            //{
            //    if (s_queue ->GetSoliSize() < enactor_slice -> subq__min_length)
            //    {
            //        continue;
            //    }
            //}

            if (iteration_loop -> status == IterationT::Status::New)
            {
                iteration_loop -> frontier_attribute 
                    = frontier_attribute;
                iteration_loop -> d_offsets
                    = graph_slice -> row_offsets   .GetPointer(util::DEVICE);
                iteration_loop -> d_indices
                    = graph_slice -> column_indices.GetPointer(util::DEVICE);
                // iteration_loop -> d_keys_in = 
                iteration_loop -> graph_slice  = graph_slice;
                iteration_loop -> scanned_edge = scanned_edge;
                iteration_loop -> max_in       = graph_slice -> nodes;
                iteration_loop -> max_out      = graph_slice -> edges;
                iteration_loop -> context      = contexts[stream_num];
                iteration_loop -> stream       = stream;
                iteration_loop -> advance_type = gunrock::oprtr::advance::V2V;
                iteration_loop -> express      = true;
                iteration_loop -> enactor_stats = enactor_stats;
                iteration_loop -> gpu_num      = gpu_num;
                iteration_loop -> stream_num   = stream_num;
                iteration_loop -> frontier_queue = frontier;
                iteration_loop -> data_slice   = data_slice;
                iteration_loop -> work_progress = work_progress;
                iteration_loop -> status       = IterationT::Status::Running;
            }
            if (Enactor::DEBUG && to_shows[stream_num])
            {
                mssg=" "; mssg[0]='0' + enactor_slice -> subq__wait_counter;
                gunrock::app::ShowDebugInfo<Enactor>(
                    gpu_num,
                    2,
                    thread_num,
                    stream_num,
                    enactor,
                    iteration,
                    mssg,
                    streams[stream_num]);
            }
            to_shows[stream_num] = true;
            pre_stage = stages[stream_num];
            //printf("stream_num = %d, stage = %d\n", stream_num, stages[stream_num]);fflush(stdout);

            switch (stages[stream_num])
            {
            case 0: // Compute Length
                //printf("gpu_num = %d, iteration = %lld, subq__target_count = %d\n",
                //    gpu_num, iteration, enactor_slice -> subq__target_count[iteration%2]);
                //fflush(stdout);
                //if (!enactor_slice -> subq__target_set[iteration%2])
                //{
                //    std::this_thread::sleep_for(std::chrono::microseconds(1));
                //    to_shows[stream_num] = false;
                //    continue;
                //}
                tretval = s_queue -> Pop_Addr(
                    enactor_slice -> subq__min_length, 
                    enactor_slice -> subq__max_length, 
                    iteration_loop -> d_keys_in,
                    s_lengths[stream_num], s_offsets[stream_num], 
                    stream, 0, 0, NULL, NULL, false,
                    enactor_slice -> subq__target_set[iteration%2], 
                    enactor_slice -> subq__target_count[iteration%2]);
                if (tretval == cudaErrorNotReady) 
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    to_shows[stream_num] = false;
                    continue;
                }
                if (tretval) 
                {
                    thread_slice -> retval = tretval;
                    return;
                }

                stream_iterations[stream_num] = iteration;
                if (s_lengths[stream_num] < enactor_slice -> subq__min_length)
                { // iteration change
                    if (enactor -> using_fullq || enactor -> num_gpus >1)
                    {
                        enactor_slice -> fullq_target_count[iteration%2]
                            = s_queue -> GetOutputCount();
                        if (s_lengths[stream_num] == 0)
                            enactor_slice -> fullq_target_count[iteration%2] --;
                        sprintf(cmssg,"fullq_target[%lld] -> %d", iteration%2, 
                            enactor_slice -> fullq_target_count[iteration%2]);
                        thread_slice -> ShowDebugInfo(cmssg, -1, iteration);
                        enactor_slice -> fullq_target_set  [iteration%2]
                            = true;
                        s_queue -> ResetOutputCount();
                        s_queue -> ChangeInputCount(0 - enactor_slice -> subq__target_count[iteration%2]);
                    } else {
                        enactor_slice -> subq__target_count[(iteration+1)%2]
                            = s_queue -> GetOutputCount();
                        s_queue -> ResetOutputCount();
                        s_queue -> ChangeInputCount(0 - enactor_slice -> subq__target_count[iteration%2]);
                        sprintf(cmssg, "subq__target[%lld] -> %d", 
                            (iteration+1)%2,
                            enactor_slice -> subq__target_count[(iteration+1)%2]);
                        thread_slice -> ShowDebugInfo(cmssg, -1, iteration);
                        enactor_slice -> subq__target_set[(iteration+1)%2]
                            = true;
                    }
                    enactor_slice -> subq__target_set[iteration%2] = false;
                    iteration_loop -> Iteration_Change(thread_slice -> iteration);
                    thread_slice -> ShowDebugInfo("Iteration change");
                    if (s_lengths[stream_num] == 0) 
                    {
                        if (thread_slice -> retval = 
                            s_queue -> EventFinish(1, s_offsets[stream_num], 0))
                            return;
                        frontier_attribute -> queue_length = 0;
                        continue;
                    }
                }

                frontier_attribute -> queue_length = s_lengths[stream_num];
                iteration_loop -> num_elements = s_lengths[stream_num];
                if (thread_slice -> retval = 
                    iteration_loop -> Compute_OutputLength())
                    return;

                frontier_attribute -> output_length.Move(
                    util::DEVICE, util::HOST, 1, 0, stream);
                if (Enactor::SIZE_CHECK)
                {
                    Set_Record(enactor_slice, 2, stream_iterations[stream_num],
                        stream_num, stages[stream_num]);
                }
                break;

            case 1: // SubQ Core
                if (Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = Check_Record(
                        enactor_slice, 2, stream_iterations[stream_num], 
                        stream_num, stages[stream_num] -1, stages[stream_num],
                        to_shows[stream_num])) return;
                    if (to_shows[stream_num] == false) continue;
                    iteration_loop -> request_length 
                        = frontier_attribute -> output_length[0] + 2;
                    if (thread_slice -> retval = 
                        iteration_loop -> Check_Queue_Size())
                        return;
                }
                enactor_stats -> iteration = stream_iterations[stream_num];
                if (thread_slice -> retval =
                    iteration_loop -> SubQueue_Core())
                    return;

                if (thread_slice -> retval =
                    work_progress->GetQueueLength(
                        frontier_attribute -> queue_index,
                        frontier_attribute -> queue_length,
                        false, stream, true))
                    return;

                if (thread_slice -> retval = 
                    s_queue -> EventSet(1, s_offsets[stream_num],
                    s_lengths[stream_num], stream)) return;

                if (thread_slice -> retval = 
                    Set_Record(enactor_slice, 2, stream_iterations[stream_num],
                        stream_num, stages[stream_num])) return;
                break;

            case 2: // Copy
                if (thread_slice -> retval = Check_Record(
                    enactor_slice, 2, stream_iterations[stream_num], 
                    stream_num, stages[stream_num] -1, stages[stream_num],
                    to_shows[stream_num])) return;
                if (to_shows[stream_num] == false) continue;
                if (!Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = Check_Size
                        <false, SizeT, VertexId>(
                        "queue3", frontier_attribute -> output_length[0] + 2,
                        &frontier -> keys[selector^1], over_sized,
                        gpu_num, stream_iterations[stream_num], 
                        stream_num, false)) return;
                }

                if (enactor -> num_gpus >1 || enactor -> using_fullq)
                {
                    if (((ThreadSlice_*)(enactor_slice -> fullq_thread_slice)) 
                        -> iteration != stream_iterations[stream_num])
                        continue;
                }

                if (thread_slice -> retval = t_queue -> Push_Addr(
                    frontier_attribute -> queue_length,
                    vertex_array, t_offset, 0, Problem::USE_DOUBLE_BUFFER?1:0,
                    NULL, &value__array)) return;
                util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
                    vertex_array,
                    frontier -> keys[selector].GetPointer(util::DEVICE),
                    frontier_attribute -> queue_length);
                if (Problem::USE_DOUBLE_BUFFER)
                    util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
                    value__array,
                    frontier -> values[selector].GetPointer(util::DEVICE),
                    frontier_attribute -> queue_length);
                if (thread_slice -> retval = t_queue -> EventSet(
                    0, t_offset, frontier_attribute -> queue_length,
                    stream)) return;
                frontier_attribute -> queue_length = 0;
                break;

            /*case 3: // Accumulate
                enactor_slice -> subq__wait_counter++;
                sprintf(cmssg, "Accumulate count = %d, target = %d", 
                    enactor_slice -> subq__wait_counter, 
                    enactor_slice -> subq__target_count[iteration%2]);
                thread_slice -> ShowDebugInfo(cmssg, -1, stream_iterations[stream_num]);
                if (enactor_slice -> subq__wait_counter == 
                    enactor_slice -> subq__target_count[iteration%2])
                {
                }
                to_shows[stream_num] = false;
                stages[stream_num] = -1;
                break;*/

            default:
                to_shows[stream_num] = false;
                stages[stream_num] = -1;
                break;
            }

            if (Enactor::DEBUG && !thread_slice -> retval)
            {
                mssg="stage 0 @ gpu 0, stream 0 failed";
                mssg[6]=char(pre_stage+'0');
                mssg[14]=char(gpu_num+'0');
                mssg[24]=char(stream_num+'0');
                if (thread_slice ->retval = util::GRError(
                    mssg, __FILE__, __LINE__)) break;
            }
            stages[stream_num] ++;
            if (thread_slice -> retval) break;
        } // end of for stream_num   
    } // end of while

    thread_slice -> ShowDebugInfo("Thread end.");
} // end of subq thread

static void FullQ_Thread(ThreadSlice_ *thread_slice)
{
    Problem       *problem          =   thread_slice  -> problem;
    Enactor       *enactor          =   thread_slice  -> enactor;
    int            num_gpus         =   enactor       -> num_gpus;
    int            gpu_num          =   thread_slice  -> gpu_num;
    int            thread_num       =   thread_slice  -> thread_num;
    util::Array1D<SizeT, DataSlice>
                  *data_slice       =   problem       -> data_slices + gpu_num;
    GraphSlice    *graph_slice      =   problem       -> graph_slices [gpu_num];
    EnactorSlice  *enactor_slice    = ((EnactorSlice*)
                                        enactor -> enactor_slices) + gpu_num;
    CircularQueue *s_queue          = &(enactor_slice -> fullq_queue);
    int            num_streams      =   enactor_slice -> num_fullq_stream;
    cudaStream_t   stream           =   (num_streams > 0) ?
                                        enactor_slice -> fullq_stream[0] : 
                                        enactor_slice -> split_streams[0];
    bool          *to_shows         =   enactor_slice -> fullq_to_show  + 0;
    int           *stages           =   enactor_slice -> fullq_stage    + 0;
    FrontierT     *frontier         =   enactor_slice -> fullq_frontier + 0;
    ContextPtr    *contexts         =   enactor_slice -> fullq_context  + 0;
    EnactorStats  *enactor_stats    =   enactor_slice -> fullq_enactor_stats + 0;
    FrontierA     *frontier_attribute
                                    =   enactor_slice -> fullq_frontier_attribute + 0;
    WorkProgress  *work_progress    =   enactor_slice -> fullq_work_progress + 0;
    Array<SizeT>  *scanned_edge     =   enactor_slice -> fullq_scanned_edge;
    int            stream_num         = 0;
    long long      iteration          = 0;
    std::string    mssg               = "";
    IterationT    *iteration_loop     = NULL;
    int            selector           = 0;
    bool           over_sized         = false;
    CircularQueue *t_queue            = NULL;
    SizeT          t_offset           = 0;
    SizeT          s_soli             = 0;
    SizeT          s_length           = 0;
    SizeT          s_offset           = 0;
    SizeT          s_input_count      = 0;
    SizeT          s_target_count     = 0;
    VertexId      *t_vertex           = NULL;
    bool           s_target_set       = false;
    cudaError_t    tretval            = cudaSuccess;
    char           cmssg[512];

    thread_slice -> ShowDebugInfo("Thread begin.");
    if (thread_slice -> retval = util::SetDevice(enactor->gpu_idx[gpu_num]))
        return;
    thread_slice -> status = ThreadSlice::Status::Wait;
    while (thread_slice -> status != ThreadSlice::Status::ToKill)
    {
        if (thread_slice -> retval) return;
        iteration  = thread_slice -> iteration;
        s_target_set   = enactor_slice -> fullq_target_set  [iteration%2];
        if (s_target_set)
        {
            s_input_count = s_queue -> GetInputCount();
            s_target_count = enactor_slice -> fullq_target_count[iteration%2];
        }
        if (thread_slice -> status == ThreadSlice::Status::Wait
            || !s_target_set || s_input_count < s_target_count)
        {
            if (s_target_set)
            {
                //sprintf(cmssg, "Waiting. input_count = %d, target_count = %d", 
                //    s_input_count, s_target_count);
                //thread_slice -> ShowDebugInfo(cmssg);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        sprintf(cmssg, "Got job. input_count = %d, target_count = %d", 
            s_input_count, s_target_count);
        thread_slice -> ShowDebugInfo(cmssg);
        enactor_slice -> fullq_target_set [iteration%2] = false;
        //s_queue -> ResetCounts();
        s_queue -> ChangeInputCount(0 - s_target_count);
        s_queue -> ResetOutputCount();

        if (num_streams >0 && enactor -> using_fullq)
        {
            iteration_loop = (IterationT*)enactor_slice -> fullq_iteration_loop;
            if (iteration_loop -> status == IterationT::Status::New)
            {
                iteration_loop -> frontier_attribute
                    = frontier_attribute;
                iteration_loop -> d_offsets
                    = graph_slice -> row_offsets   .GetPointer(util::DEVICE);
                iteration_loop -> d_indices
                    = graph_slice -> column_indices.GetPointer(util::DEVICE);
                iteration_loop -> graph_slice  = graph_slice;
                iteration_loop -> scanned_edge = scanned_edge;
                iteration_loop -> max_in       = graph_slice -> nodes;
                iteration_loop -> max_out      = graph_slice -> edges;
                iteration_loop -> context      = contexts[stream_num];
                iteration_loop -> stream       = stream;
                iteration_loop -> advance_type = gunrock::oprtr::advance::V2V;
                iteration_loop -> express      = true;
                iteration_loop -> enactor_stats = enactor_stats;
                iteration_loop -> gpu_num      = gpu_num;
                iteration_loop -> stream_num   = stream_num;
                iteration_loop -> frontier_queue = frontier;
                iteration_loop -> data_slice   = data_slice;
                iteration_loop -> work_progress = work_progress;
                iteration_loop -> status       = IterationT::Status::Running;
                
            }

            s_queue -> GetSize(s_length, s_soli);
            tretval = cudaErrorNotReady;
            while (tretval == cudaErrorNotReady)
            {
                tretval = s_queue -> Pop_Addr(
                    s_length, s_length, iteration_loop -> d_keys_in,
                    iteration_loop -> num_elements, s_offset, stream);
            }
            if (tretval) 
            {
                thread_slice -> retval = tretval;
                return ;
            }

            frontier_attribute -> queue_length = iteration_loop -> num_elements;
            frontier_attribute -> queue_offset = 0;
            frontier_attribute -> queue_reset  = true;
            frontier_attribute -> selector     = 0;
            if (thread_slice -> retval = 
                iteration_loop -> FullQueue_Gather()) return;
            selector = frontier_attribute -> selector;
       
            if (frontier_attribute -> queue_length != 0)
            {
                stages[stream_num] = 0;
                if (Enactor::DEBUG) {
                    mssg = "";
                    gunrock::app::ShowDebugInfo<Enactor>(
                        gpu_num,
                        3,
                        thread_num,
                        stream_num,
                        enactor,
                        iteration,
                        mssg,
                        stream);
                }

                if (thread_slice -> retval = 
                    iteration_loop -> Compute_OutputLength()) return;
                frontier_attribute -> output_length.Move(util::DEVICE, util::HOST, 1, 0, stream);
                if (Enactor::SIZE_CHECK)
                {
                    Set_Record(enactor_slice, 3, iteration, stream_num, 
                        stages[stream_num]); 
                }

                stages[stream_num] ++;
                if (Enactor::SIZE_CHECK)
                {
                    to_shows[stream_num] = false;
                    while (to_shows[stream_num] == false)
                    {
                        if (thread_slice -> retval = Check_Record(
                            enactor_slice, 3, iteration, stream_num,
                            stages[stream_num] -1, stages[stream_num],
                            to_shows[stream_num])) return;
                    }
                    if (thread_slice -> retval) return;
                    iteration_loop -> request_length
                        = frontier_attribute -> output_length[0] + 2;
                    if (thread_slice -> retval = 
                        iteration_loop -> Check_Queue_Size())
                        return;
                }
                if (thread_slice -> retval = 
                    iteration_loop -> FullQueue_Core())
                    return;
                if (thread_slice -> retval = 
                    s_queue -> EventSet(0, s_length, s_offset, stream))
                    return;
                if (thread_slice -> retval =
                    work_progress -> GetQueueLength(
                        frontier_attribute -> queue_index,
                        frontier_attribute -> queue_length,
                        false, stream, true))
                    return;
                Set_Record(enactor_slice, 3, iteration, stream_num,
                    stages[stream_num]);

                stages[stream_num] ++;
                to_shows[stream_num] = false;
                while (to_shows[stream_num] == false)
                {
                    if (thread_slice -> retval = Check_Record(
                        enactor_slice, 3, iteration, stream_num,
                        stages[stream_num] -1, stages[stream_num],
                        to_shows[stream_num])) return;
                }
                selector = frontier_attribute -> selector;
                if (!Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = 
                        Check_Size <false, SizeT, VertexId>(
                        "queue3", frontier_attribute -> output_length[0] + 2,
                        &frontier -> keys[selector^1], over_sized,
                        gpu_num, iteration, stream_num, false)) return; 
                } 
            } // end of if (queue_length != 0)
            sprintf(cmssg, "Fullqueue finished. Queue_Length = %d",
                frontier_attribute->queue_length);
            thread_slice -> ShowDebugInfo(cmssg);
        } // end of if (has_fullq)

        if (enactor -> using_fullq)
        {
            iteration_loop -> d_keys_in
                = frontier->keys[frontier_attribute->selector].GetPointer(util::DEVICE);
            iteration_loop -> num_elements = frontier_attribute->queue_length;
        } else if (num_gpus > 1) {
            iteration_loop = (IterationT*)enactor_slice -> split_iteration_loop;
            s_queue->GetSize(s_length, s_soli);
            tretval = cudaErrorNotReady;
            while (tretval == cudaErrorNotReady)
            {
                tretval = s_queue -> Pop_Addr(
                    s_length, s_length,
                    iteration_loop -> d_keys_in, 
                    iteration_loop -> num_elements, s_offset, stream);
            }
            if (tretval)
            {
                thread_slice -> retval = tretval;
                return;
            }
        }

        if (num_gpus > 1)
        { // update and distribute the queue
            iteration_loop = (IterationT*)enactor_slice -> split_iteration_loop;
            if (iteration_loop -> status == IterationT::Status::New)
            {
                iteration_loop -> graph_slice   = graph_slice;
                iteration_loop -> data_slice    = data_slice;
                iteration_loop -> frontier_attribute = NULL; //frontier_attribute;
                iteration_loop -> gpu_num       = gpu_num;
                iteration_loop -> streams       = enactor_slice -> split_streams+ 0;
                iteration_loop -> num_gpus      = num_gpus;
                iteration_loop -> scanned_edge  = NULL;//scanned_edge;
                iteration_loop -> enactor_stats = NULL;//enactor_stats;
                iteration_loop -> contexts      = enactor_slice -> split_contexts + 0;
                //iteration_loop -> context       = contexts[stream_num];
                //iteration_loop -> stream_num    = stream_num;
                iteration_loop -> work_progress = NULL;//work_progress;
                iteration_loop -> markers       = enactor_slice -> split_markers;
                iteration_loop -> markerss      = &enactor_slice -> split_markerss;
                iteration_loop -> wait_event    = enactor_slice -> split_wait_event;
                iteration_loop -> t_out_lengths = &enactor_slice -> split_lengths;
                iteration_loop -> events        = enactor_slice -> split_events + 0;
                iteration_loop -> m_handles     = &enactor_slice -> split_m_handles;
                iteration_loop -> outpu_queue   = &enactor_slice -> outpu_queue;
                iteration_loop -> fullq_queue   = &enactor_slice -> fullq_queue;
                iteration_loop -> subq__queue   = &enactor_slice -> subq__queue;
                iteration_loop -> enactor_slice = enactor_slice;
                iteration_loop -> rqueue_mutex  = &enactor_slice -> outpu_request_mutex;
                iteration_loop -> request_queue  = &enactor_slice -> outpu_request_queue;
                iteration_loop -> status        = IterationT::Status::Running;
            }
            iteration_loop -> num_streams = enactor_slice -> num_split_streams;
            iteration_loop -> streams     = enactor_slice -> split_streams + 0;
            iteration_loop -> num_vertex_associates = enactor -> num_vertex_associates;
            iteration_loop -> num_value__associates = enactor -> num_value__associates;
            iteration_loop -> vertex_associate_orgs = &enactor_slice -> vertex_associate_orgs;
            iteration_loop -> value__associate_orgs = &enactor_slice -> value__associate_orgs;
            iteration_loop -> iteration   = iteration;
            if (thread_slice -> retval = 
                iteration_loop -> Iteration_Update_Preds())
                return;
            
            //if (thread_slice -> retval = util::GRError(
            //    cudaStreamSynchronize(stream),
            //    "cudaStream Synchronize failed", __FILE__, __LINE__))
            //    return;

            if (thread_slice -> retval = util::GRError(cudaEventRecord(
                iteration_loop -> wait_event, stream),
                "cudaEventRecord failed", __FILE__, __LINE__)) return;
            if (thread_slice -> retval =
                iteration_loop -> Make_Output())
                return;
            if (!iteration_loop -> has_fullq)
            {
                if (thread_slice -> retval = s_queue -> 
                    EventSet(1, s_offset, s_length, stream))
                    return;
            }
        } else { // push back to input queue
            if (iteration_loop -> has_fullq)
            {
                if (iteration_loop -> has_subq)
                    t_queue = &enactor_slice -> subq__queue;
                else t_queue = &enactor_slice -> fullq_queue;
                if (thread_slice -> retval = t_queue->Push_Addr(
                    iteration_loop -> num_elements,
                    t_vertex, t_offset)) return;
                util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
                    t_vertex, iteration_loop -> d_keys_in,
                    iteration_loop -> num_elements);
            } 
        }

        iteration_loop -> Iteration_Change(thread_slice -> iteration);
        thread_slice -> ShowDebugInfo("Iteration change");
    } // end of while

    thread_slice -> ShowDebugInfo("Thread end.");
} // end of fullq thread

}; // end of ThreadSlice
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
