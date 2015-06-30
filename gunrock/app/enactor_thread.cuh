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
#include <gunrock/app/enactor_loop.cuh>

namespace gunrock {
namespace app {

/**
 * @brief Structure for per-thread variables used in sub-threads.
 */
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
class ThreadSlice
{    
public:

    enum Status {
        New,
        Init,
        Start,
        Wait,
        Running,
        Ideal,
        ToKill,
        Ended
    };  

    int           thread_num ;
    int           thread_type;
    int           gpu_num    ;   
    long long     iteration  ;   
    int           init_size  ;
    Status        status     ;   
    void         *problem    ;   
    void         *enactor    ;   
    cudaError_t   retval     ;   
    util::cpu_mt::CPUBarrier
                 *cpu_barrier;
    IterationBase <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
                 *iteration_loops;

    ThreadSlice() :
        thread_num (0   ),  
        thread_type(0   ),  
        gpu_num    (0   ),  
        iteration  (0   ),  
        init_size  (0   ),  
        status     (New ),  
        problem    (NULL),
        enactor    (NULL),
        retval     (cudaSuccess),
        cpu_barrier(NULL),
        iteration_loops(NULL)
    {    
    }    

    virtual ~ThreadSlice()
    {    
        problem     = NULL;
        enactor     = NULL;
        cpu_barrier = NULL;
        iteration_loops = NULL;
    }    
}; // end of ThreadSlice

template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
void Outpu_Thread(void *_thread_slice)
{
    typedef typename Enactor::PRequest   PRequest;
    typedef typename std::list<PRequest> PRList;
    typedef typename Enactor::Problem    Problem;
    typedef typename Problem::DataSlice  DataSlice;
    typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, 
        Enactor> ThreadSlice;
    typedef EnactorSlice<Enactor> EnactorSlice;

    ThreadSlice  *thread_slice = (ThreadSlice*) _thread_slice;
    //Problem      *problem = (Problem*) thread_slice  -> problem;
    Enactor      *enactor = (Enactor*) thread_slice  -> enactor;
    int           gpu_num          =   thread_slice  -> gpu_num;
    int           num_gpus         =   enactor -> num_gpus;
    EnactorSlice *enactor_slice    = ((EnactorSlice*) enactor -> enactor_slices) + gpu_num;
    PRList       *request_queue    = &(enactor_slice -> outpu_request_queue);
    std::mutex   *rqueue_mutex     = &(enactor_slice -> outpu_request_mutex); 
    int           stream_selector  = 0;
    cudaStream_t  stream           = 0;
    cudaStream_t *streams          = &(enactor_slice -> outpu_streams[0]);
    typename std::list<PRequest>::iterator it, it_;

    while (thread_slice -> status != 4)
    {
        if (!request_queue->empty())
        {
            rqueue_mutex->lock();
            it = request_queue->begin();
            while (it != request_queue->end())
            {
                it_ = it; it++;
                if ((*it_).status == 1) // requested
                {
                    stream = streams[stream_selector];
                    stream_selector ++;
                    stream_selector = stream_selector % enactor_slice->num_outpu_streams;
                    if ((*it_).peer < num_gpus) // local peer
                    {
                        (*it_).stream = stream;
                        if (thread_slice -> retval = PushNeibor <Enactor> (&(*it_), enactor))
                            break;
                        request_queue -> erase(it_);
                    } else { // remote peer
                    }
                }
            }
            rqueue_mutex->unlock();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (thread_slice -> retval) break;
    }
}

template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
void Input_Thread(void *_thread_slice)
{
    typedef typename Enactor::Problem       Problem      ;
    typedef typename Problem::DataSlice     DataSlice    ;
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::SizeT         SizeT        ;
    typedef typename Enactor::VertexId      VertexId     ;
    typedef typename Enactor::Value         Value        ;
    typedef ThreadSlice<AdvanceKernelPolicy, FilterKernelPolicy, 
        Enactor> ThreadSlice;
    //template <typename Type>
    //using Array = typename Enactor::Array<Type>;
 
    ThreadSlice  *thread_slice = (ThreadSlice*) _thread_slice;
    Problem       *problem = (Problem*) thread_slice  -> problem;
    Enactor       *enactor = (Enactor*) thread_slice  -> enactor;
    int            gpu_num          =   thread_slice  -> gpu_num;
    util::Array1D<SizeT, DataSlice> 
                  *data_slice       = &(problem       -> data_slices[gpu_num]);
    EnactorSlice<Enactor> 
                  *enactor_slice    = ((EnactorSlice<Enactor>*) enactor -> enactor_slices) + gpu_num;
    CircularQueue *input_queues     =   enactor_slice -> input_queues;
    cudaStream_t  *streams          = &(enactor_slice -> input_streams[0]);
    int            target_input_count = enactor_slice -> input_target_count;
    typename Enactor::Array<char>   
                  *e_arrays         =   enactor_slice -> input_e_arrays;
    long long     *iteration_       = &(thread_slice  -> iteration); 
    long long      iteration        = 0;
    VertexId      *s_vertices       = NULL;
    VertexId      *t_vertices       = NULL;
    VertexId      *s_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value         *s_value__associates[Enactor::NUM_VALUE__ASSOCIATES];
    SizeT          s_size_soli      = 0;
    SizeT          s_size_occu      = 0;
    SizeT          length           = 0;
    SizeT          s_offset         = 0;
    SizeT          t_offset         = 0;
    SizeT          grid_size        = 0;
    SizeT          min_length       = 1;
    SizeT          max_length       = 32 * 1024 * 1024;
    cudaStream_t   stream           = 0;
    int            stream_selector  = 0;
    CircularQueue *s_queue          = NULL;
    CircularQueue *t_queue          = NULL;
    SizeT          e_offset         = 0;
    typename Enactor::Array<char>   
                  *e_array          = NULL;
    VertexId     **o_vertex_associates   = NULL;
    Value        **o_value__associates   = NULL;
    SizeT          num_vertex_associates = 0;
    SizeT          num_value__associates = 0;
    IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
                  *iteration_loop = NULL;

    while (thread_slice -> status != ThreadSlice::ToKill)
    {
        iteration = iteration_[0];
        s_queue   = &(input_queues[iteration%2]);
        if (!s_queue->Empty())
        {
            e_array = &(e_arrays[stream_selector]);
            s_queue -> GetSize(s_size_occu, s_size_soli);
            t_queue = (enactor -> using_subq) ? &enactor_slice -> subq__queue :
                &enactor_slice -> fullq_queue;
            stream  = streams[stream_selector];
            length = (s_queue->GetInputCount() == target_input_count) ? 
                      s_size_occu : s_size_soli;
            o_vertex_associates = enactor_slice -> 
                vertex_associate_orgs.GetPointer(util::HOST);
            o_value__associates = enactor_slice -> 
                value__associate_orgs.GetPointer(util::HOST);
            num_vertex_associates = enactor -> num_vertex_associates;
            num_value__associates = enactor -> num_value__associates;

            if (thread_slice -> retval = s_queue->Pop_Addr(
                min_length, max_length, s_vertices, length, s_offset, stream,
                num_vertex_associates, num_value__associates,
                s_vertex_associates, s_value__associates)) return;

            if (thread_slice -> retval = t_queue->Push_Addr(
                length, t_vertices, t_offset)) return;

            e_offset = 0;
            memcpy( &(e_array[e_offset]),     s_vertex_associates ,
                        sizeof(VertexId*) * num_vertex_associates);
            e_offset += sizeof(VertexId*) * num_vertex_associates ;
            memcpy( &(e_array[e_offset]),     s_value__associates ,
                        sizeof(Value   *) * num_value__associates);
            e_offset += sizeof(Value   *) * num_value__associates ;
            memcpy( &(e_array[e_offset]),     o_vertex_associates ,
                        sizeof(VertexId*) * num_vertex_associates);
            e_offset += sizeof(VertexId*) * num_vertex_associates ;
            memcpy( &(e_array[e_offset]),     o_value__associates ,
                        sizeof(Value   *) * num_value__associates);
            e_offset += sizeof(Value   *) * num_value__associates ;
            e_array->Move(util::HOST, util::DEVICE, e_offset, 0, stream);

            grid_size = length/256+1;
            if (grid_size>512) grid_size=512;
            iteration_loop = thread_slice -> iteration_loops + stream_selector;
            iteration_loop -> num_vertex_associates = num_vertex_associates;
            iteration_loop -> num_value__associates = num_value__associates;
            iteration_loop -> grid_size             = grid_size;
            iteration_loop -> block_size            = 256;
            iteration_loop -> shared_size           = e_offset;
            iteration_loop -> stream                = stream;
            iteration_loop -> num_elements          = length;
            iteration_loop -> d_keys_in             = s_vertices;
            iteration_loop -> d_keys_out            = t_vertices;
            iteration_loop -> array_size            = e_offset;
            iteration_loop -> d_array               = e_array->GetPointer(util::DEVICE);
            iteration_loop -> data_slice            = data_slice;

            if (thread_slice -> retval = iteration_loop -> Expand_Incoming())
                return;

            if (thread_slice -> retval = 
                s_queue -> EventSet(1, s_offset, length, stream))
                return;
            if (thread_slice -> retval = 
                t_queue -> EventSet(0, t_offset, length, stream))
                return;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (thread_slice -> retval) break;
    }

}

template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
void SubQ__Thread(void *_thread_slice)
{
    typedef typename Enactor::Problem       Problem      ;   
    typedef typename Problem::DataSlice     DataSlice    ;
    typedef typename Enactor::GraphSlice    GraphSlice   ; 
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::SizeT         SizeT        ;   
    typedef typename Enactor::VertexId      VertexId     ;   
    typedef typename Enactor::Value         Value        ;   
    typedef typename Enactor::FrontierA     FrontierA    ;
    typedef typename Enactor::FrontierT     FrontierT    ;
    typedef typename Enactor::WorkProgress  WorkProgress ;
    typedef typename Enactor::EnactorStats  EnactorStats ;
    typedef ThreadSlice  <AdvanceKernelPolicy, FilterKernelPolicy, 
        Enactor> ThreadSlice;
    typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, 
        Enactor> IterationT;
    typedef EnactorSlice<Enactor>           EnactorSlice;

    ThreadSlice   *thread_slice = (ThreadSlice*) _thread_slice;
    Problem       *problem = (Problem*) thread_slice  -> problem;
    Enactor       *enactor = (Enactor*) thread_slice  -> enactor;
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
    util::CtaWorkProgressLifetime
                  *work_progresses  =   enactor_slice -> subq__work_progresses + 0; 
    typename Enactor::Array<SizeT> 
                  *scanned_edges    =   enactor_slice -> subq__scanned_edges;
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
    typename Enactor::Array<SizeT> 
                  *scanned_edge       = NULL;
    util::CtaWorkProgressLifetime 
                  *work_progress      = NULL;
    int            selector           = 0;
    bool           over_sized         = false;
    CircularQueue *t_queue            = NULL;
    VertexId      *vertex_array       = NULL;
    Value         *value__array       = NULL;
    SizeT          t_offset           = 0;
    int            pre_stage          = 0;
 
    thread_slice -> status = ThreadSlice::Running;
    while (thread_slice -> status != ThreadSlice::ToKill)
    {
        if (thread_slice -> retval) break;
        if (thread_slice -> status == ThreadSlice::Status::Wait 
            || s_queue -> Empty()) 
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        iteration  = thread_slice -> iteration;
        //iteration_ = iteration % 4;
        iteration_loops = (IterationT*)enactor_slice -> subq__iteration_loops;
        if (!iteration_loops[0].has_subq) continue;
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
                ShowDebugInfo<Enactor>(
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

            switch (stages[stream_num])
            {
            case 0: // Compute Length
                if (thread_slice -> retval = 
                    iteration_loop -> Compute_OutputLength())
                    break;
                frontier_attribute -> output_length.Move(
                    util::DEVICE, util::HOST, 1, 0, stream);
                if (Enactor::SIZE_CHECK)
                {
                    Set_Record(enactor_slice, 2, iteration, stream_num, 
                        stages[stream_num]);
                }
                break;

            case 1: // SubQ Core
                if (Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = Check_Record(
                        enactor_slice, 2, iteration, stream_num,
                        stages[stream_num] -1, stages[stream_num],
                        to_shows[stream_num])) break;
                    if (to_shows[stream_num] == false) break;
                    iteration_loop -> request_length 
                        = frontier_attribute -> output_length[0] + 2;
                    if (thread_slice -> retval = 
                        iteration_loop -> Check_Queue_Size())
                        break;
                }
                if (thread_slice -> retval =
                    iteration_loop -> SubQueue_Core())
                    break;
                if (thread_slice -> retval =
                    work_progress->GetQueueLength(
                        frontier_attribute -> queue_index,
                        frontier_attribute -> queue_length,
                        false, stream, true))
                    break;
                Set_Record(enactor_slice, 2, iteration, stream_num,
                        stages[stream_num]);
                break;

            case 2: // Copy
                if (thread_slice -> retval = Check_Record(
                    enactor_slice, 2, iteration, stream_num,
                    stages[stream_num] -1, stages[stream_num],
                    to_shows[stream_num])) break;
                if (to_shows[stream_num] == false) break;
                if (!Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = Check_Size
                        <false, SizeT, VertexId>(
                        "queue3", frontier_attribute -> output_length[0] + 2,
                        &frontier -> keys[selector^1], over_sized,
                        gpu_num, iteration, stream_num, false)) break;
                } 

                if (thread_slice -> retval = t_queue -> Push_Addr(
                    frontier_attribute -> queue_length,
                    vertex_array, t_offset, 0, Problem::USE_DOUBLE_BUFFER?1:0,
                    NULL, &value__array)) break;
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
                    stream)) break;
                break;

            case 3: // Accumulate
                enactor_slice -> subq__wait_counter++;
                to_shows[stream_num] = false;
                stages[stream_num] = -1;
                break;

            default:
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
}

template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
void FullQ_Thread(void *_thread_slice)
{
    typedef typename Enactor::Problem       Problem      ;
    typedef typename Problem::DataSlice     DataSlice    ;
    typedef typename Enactor::GraphSlice    GraphSlice   ;
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::SizeT         SizeT        ;
    typedef typename Enactor::VertexId      VertexId     ;
    typedef typename Enactor::Value         Value        ;
    typedef typename Enactor::FrontierA     FrontierA    ;
    typedef typename Enactor::FrontierT     FrontierT    ;
    typedef typename Enactor::WorkProgress  WorkProgress ;
    typedef typename Enactor::EnactorStats  EnactorStats ;
    typedef ThreadSlice  <AdvanceKernelPolicy, FilterKernelPolicy,
        Enactor> ThreadSlice;
    typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy,
        Enactor> IterationT;
    typedef EnactorSlice<Enactor>           EnactorSlice;

    ThreadSlice   *thread_slice = (ThreadSlice*) _thread_slice;
    Problem       *problem = (Problem*) thread_slice  -> problem;
    Enactor       *enactor = (Enactor*) thread_slice  -> enactor;
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
    cudaStream_t   stream           =   enactor_slice -> fullq_stream[0];
    bool          *to_shows         =   enactor_slice -> fullq_to_show  + 0;
    int           *stages           =   enactor_slice -> fullq_stage    + 0;
    FrontierT     *frontier         =   enactor_slice -> fullq_frontier + 0;
    ContextPtr    *contexts         =   enactor_slice -> fullq_context  + 0;
    EnactorStats  *enactor_stats    =   enactor_slice -> fullq_enactor_stats + 0;
    FrontierA     *frontier_attribute
                                    =   enactor_slice -> fullq_frontier_attribute + 0;
    util::CtaWorkProgressLifetime
                  *work_progress    =   enactor_slice -> fullq_work_progress + 0;
    typename Enactor::Array<SizeT>
                  *scanned_edge     =   enactor_slice -> fullq_scanned_edge;
    int            stream_num         = 0;
    long long      iteration          = 0;
    //long long      iteration_         = 0;
    std::string    mssg               = "";
    IterationT    *iteration_loop     = NULL;
    int            selector           = 0;
    bool           over_sized         = false;
    CircularQueue *t_queue            = NULL;
    //VertexId      *vertex_array       = NULL;
    //Value         *value__array       = NULL;
    SizeT          t_offset           = 0;
    //int            pre_stage          = 0;
    SizeT          s_soli             = 0;
    SizeT          s_length           = 0;
    SizeT          s_offset           = 0;
    VertexId      *t_vertex           = NULL;

    thread_slice -> status = ThreadSlice::Running;
    while (thread_slice -> status != ThreadSlice::ToKill)
    {
        if (thread_slice -> retval) break;
        if (thread_slice -> status == ThreadSlice::Status::Wait
            || s_queue -> Empty())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }

        iteration  = thread_slice -> iteration;
        iteration_loop = (IterationT*)enactor_slice -> fullq_iteration_loop;
        if (num_streams >0 && iteration_loop -> has_fullq)
        {
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

            frontier_attribute -> queue_offset = 0;
            frontier_attribute -> queue_reset  = true;
            frontier_attribute -> selector     = 0;
            if (thread_slice -> retval = 
                iteration_loop -> FullQueue_Gather()) break;
            selector = frontier_attribute -> selector;
       
            if (frontier_attribute -> queue_length != 0)
            {
                stages[stream_num] = 0;
                if (Enactor::DEBUG) {
                    mssg = "";
                    ShowDebugInfo<Enactor>(
                        gpu_num,
                        3,
                        thread_num,
                        stream_num,
                        enactor,
                        iteration,
                        mssg,
                        stream);
                }

                // iteration_loop -> d_keys_in = 
                if (thread_slice -> retval = 
                    iteration_loop -> Compute_OutputLength()) break;
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
                            to_shows[stream_num])) break;
                    }
                    if (thread_slice -> retval) break;
                    iteration_loop -> request_length
                        = frontier_attribute -> output_length[0] + 2;
                    if (thread_slice -> retval = 
                        iteration_loop -> Check_Queue_Size())
                        break;
                }
                if (thread_slice -> retval = 
                    iteration_loop -> FullQueue_Core())
                    break;
                if (thread_slice -> retval =
                    work_progress -> GetQueueLength(
                        frontier_attribute -> queue_index,
                        frontier_attribute -> queue_length,
                        false, stream, true))
                    break;
                Set_Record(enactor_slice, 3, iteration, stream_num,
                    stages[stream_num]);

                stages[stream_num] ++;
                to_shows[stream_num] = false;
                while (to_shows[stream_num] == false)
                {
                    if (thread_slice -> retval = Check_Record(
                        enactor_slice, 3, iteration, stream_num,
                        stages[stream_num] -1, stages[stream_num],
                        to_shows[stream_num])) break;
                }
                selector = frontier_attribute -> selector;
                if (!Enactor::SIZE_CHECK)
                {
                    if (thread_slice -> retval = 
                        Check_Size <false, SizeT, VertexId>(
                        "queue3", frontier_attribute -> output_length[0] + 2,
                        &frontier -> keys[selector^1], over_sized,
                        gpu_num, iteration, stream_num, false)) break; 
                } 
            } // end of if (queue_length != 0)
            if (Enactor::DEBUG) 
            {
                printf("%d\t %lld\t \t Fullqueue finished. Queue_Length= %d\n",
                    gpu_num, iteration, frontier_attribute->queue_length);
                fflush(stdout);
            }
        } // end of if (has_fullq)

        if (iteration_loop -> has_fullq)
        {
            iteration_loop -> d_keys_in
                = frontier->keys[frontier_attribute->selector].GetPointer(util::DEVICE);
            iteration_loop -> num_elements = frontier_attribute->queue_length;
        } else if (num_gpus > 1) {
            s_queue->GetSize(iteration_loop -> num_elements, s_soli);
            if (thread_slice -> retval = 
                s_queue -> Pop_Addr(
                    iteration_loop -> num_elements,
                    iteration_loop -> num_elements,
                    iteration_loop -> d_keys_in, 
                    s_length, s_offset, stream))
                break;
        }

        if (num_gpus > 1)
        { // update and distribute the queue
            if (iteration_loop -> status == IterationT::Status::New)
            {
                iteration_loop -> graph_slice = graph_slice;
                iteration_loop -> data_slice  = data_slice;
                iteration_loop -> frontier_attribute = frontier_attribute;
                iteration_loop -> gpu_num     = gpu_num;
                iteration_loop -> stream      = stream;
                iteration_loop -> num_gpus    = num_gpus;
                iteration_loop -> scanned_edge = scanned_edge;
                iteration_loop -> enactor_stats = enactor_stats;
                iteration_loop -> context     = contexts[stream_num];
                iteration_loop -> stream_num  = stream_num;
                iteration_loop -> work_progress = work_progress;
                iteration_loop -> status      = IterationT::Status::Running;
            }
            iteration_loop -> num_streams = enactor_slice -> num_split_streams;
            iteration_loop -> streams     = enactor_slice -> split_streams + 0;
            iteration_loop -> iteration   = iteration;
            if (thread_slice -> retval = 
                iteration_loop -> Iteration_Update_Preds())
                break;

            if (thread_slice -> retval =
                iteration_loop -> Make_Output())
                break;
        } else { // push back to input queue
            if (iteration_loop -> has_fullq)
            {
                if (iteration_loop -> has_subq)
                    t_queue = &enactor_slice -> subq__queue;
                else t_queue = &enactor_slice -> fullq_queue;
                if (thread_slice -> retval = t_queue->Push_Addr(
                    iteration_loop -> num_elements,
                    t_vertex, t_offset)) break;
                util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
                    t_vertex, iteration_loop -> d_keys_in,
                    iteration_loop -> num_elements);
            } 
        }
    } // end of while

    /*if (Iteration::HAS_FULLQ)
    {
        if (!Enactor::SIZE_CHECK) frontier_attribute_->selector     = 0;

        
        if (frontier_attribute_->queue_length !=0)
        {

            selector = frontier_attribute[peer_].selector;
            Total_Length = frontier_attribute[peer_].queue_length;
        } else {
            Total_Length = 0;
            for (peer__=0;peer__<num_gpus;peer__++)
                data_slice->out_length[peer__]=0;
        }
        frontier_queue_ = &(data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus]);
        if (num_gpus==1) data_slice->out_length[0]=Total_Length;
    }
    
    for (peer_=0;peer_<num_gpus;peer_++)
        frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];
    */
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
