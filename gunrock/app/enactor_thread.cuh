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
 * @brief Structure for per-thread variables used in sub-threads.
 */
class ThreadSlice
{    
public:
    int           thread_num ;
    int           gpu_num    ;
    int           init_size  ;
    CUTThread     thread_Id  ;
    int           stats      ;
    void         *problem    ;
    void         *enactor    ;
    ContextPtr   *context    ;
    util::cpu_mt::CPUBarrier
                 *cpu_barrier;

    ThreadSlice() :
        thread_num (0   ),
        init_size  (0   ),
        stats      (-2  ),
        problem    (NULL),
        enactor    (NULL),
        context    (NULL),
        cpu_barrier(NULL)
    {    
    }    

    virtual ~ThreadSlice()
    {    
        problem     = NULL;
        enactor     = NULL;
        context     = NULL;
        cpu_barrier = NULL;
    }    
}; // end of ThreadSlice

template<typename Enactor>
void Send_Thread(ThreadSlice *thread_data)
{
    typedef Enactor::PRequest            PRequest;
    typedef typename std::list<PRequest> PRList;
    typedef Enactor::Problem             Problem;
    typedef Problem::DataSlice           DataSlice;

    Problem      *problem       = (Problem*) thread_data->problem;
    Enactor      *enactor       = (Enactor*) thread_data->enactor;
    int           gpu_num       =   thread_data   -> gpu_num;
    EnactorSlice *enactor_slice = &(enactor       -> enactor_slices[gpu_num]);
    PRList       *request_queue = &(enactor_slice -> outpu_request_queue);
    std::mutex   *rqueue_mutex  = &(enactor_slice -> outpu_request_mutex); 
    
    while (thread_data -> status != 4)
    {
        if (!request_queue.empty())
        {
            rqueue_mutex->lock();
            PRList::iterator it_, it = request_queue->begin();
            while (it != request_queue->end())
            {
                it_ = it; it++;
                if ((*it_).status == 1) // requested
                {
                    if ((*it_).peer < num_gpus) // local peer
                    {
                        if (thread_data -> retval = PushNeibor <Enactor> (&(*it_), problem))
                            break;
                        request_queue -> earse(it_);
                    } else { // remote peer
                    }
                }
            }
            rqueue_mutex->unlock();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (thread_data -> retval) break;
    }
}

template <typename Enactor>
void Recv_Thread(ThreadSlice *thread_data)
{
    typedef typename Enactor::Problem       Problem      ;
    typedef typename Problem::DataSlice     DataSlice    ;
    typedef typename Enactor::CircularQueue CircularQueue;
    typedef typename Enactor::SizeT         SizeT        ;
    typedef typename Enactor::VertexId      VertexId     ;
    typedef typename Enactor::Value         Value        ;

    int            gpu_num     = thread_data -> gpu_num;
    Problem       *problem     = (Problem*) thread_data -> problem;
    Enactor       *enactor     = (Enactor*) thread_data -> enactor;
    DataSlice     *data_slice  = &(problem ->data_slices[gpu_num][0]);
    EnactorSlice  *enactor_slice = &(enactor -> enactor_slices[gpu_num]);
    CircularQueue *input_queue = data_slice->input_queue;
    SizeT         *iteration_  = &(thread_data->iteration); 
    SizeT          iteration   = 0;
    VertexId      *s_vertices  = NULL;
    VertexId      *t_vertices  = NULL;
    VertexId      *s_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value         *s_value__associates[Enactor::NUM_VALUE__ASSOCIATES];
    SizeT          s_size_soli = 0;
    SizeT          s_size_occu = 0;
    SizeT          length      = 0;
    SizeT          s_offset    = 0;
    SizeT          min_length  = 1;
    SizeT          max_length  = 1024 * 1024;
    SizeT          stream    ; // =?
    CircularQueue *s_cq        = NULL;
    CircularQueue *t_cq      ; // =?
    SizeT          e_offset    = 0;
    util::Array1D<SizeT, char> *e_array = NULL;
    VertexId      *o_vertex_associates = NULL;
    SizeT          grid_size   = 0;

    while (thread_data -> status != 4)
    {
        iteration = iteratino_[0];
        s_cq = &(input_cq[iteration%2]);
        if (!s_cq->empty())
        {
            e_array = &(data_slice->expand_incoming_array[0 /**/]);
            s_cq->GetSize(size_occu, size_soli);
            length = (s_cq->GetInputCount() == to_input) ? size_occu : size_soli;
            o_vertex_associates = data_slice -> vertex_associate_orgs.GetPointer(util::HOST);
            o_value__associates = data_slice -> value__associate_orgs.GetPointer(util::HOST);
            if (thread_data -> retval = s_cq->Pop_Addr(
                min_length, max_length, s_vertices, length, s_offset, stream,
                num_vertex_associates, num_value__associates,
                s_vertex_associates, s_value__associates)) return retval;

            if (thread_data -> retval = t_cq->Push_Addr(
                length, t_vertices, t_offset)) return retval;

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
            data_slice->iteration->Expand_Incoming(
                num_vertex_associates,
                num_value__associates,
                grid_size, 
                256, 
                e_offset,
                stream,
                length,
                s_vertices,
                t_vertices,
                e_offset,
                e_array->GetPointer(util::DEVICE),
                data_slice);

            if (thread_data -> retval = s_cq->EventSet(1, s_offset, length, stream))
                return retval;
            if (thread_data -> retval = t_cq->EventSet(0, t_offset, length, stream))
                return retval;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (thread_data -> retval) break;
    }

}

template <typename Enactor>
void SubQ_Thread(ThreadSlice *thread_data)
{
    for (peer__=0; peer__<num_gpus*2; peer__++)
    {
        peer_               = (peer__%num_gpus);
        peer                = peer_<= thread_num? peer_-1   : peer_       ;
        gpu_                = peer <  thread_num? thread_num: thread_num+1;
        iteration           = enactor_stats[peer_].iteration;
        iteration_          = iteration%4;
        pre_stage           = stages[peer__];
        selector            = frontier_attribute[peer_].selector;
        frontier_queue_     = &(data_slice->frontier_queues[peer_]);
        scanned_edges_      = &(data_slice->scanned_edges  [peer_]);
        frontier_attribute_ = &(frontier_attribute         [peer_]);
        enactor_stats_      = &(enactor_stats              [peer_]);
        work_progress_      = &(work_progress              [peer_]);

        if (Enactor::DEBUG && to_show[peer__])
        {
            mssg=" ";mssg[0]='0'+data_slice->wait_counter;
            ShowDebugInfo<Problem>(
                thread_num,
                peer__,
                frontier_attribute_,
                enactor_stats_,
                data_slice,
                graph_slice,
                work_progress_,
                mssg,
                streams[peer__]);
        }
        to_show[peer__]=true;

        switch (stages[peer__])
        {
        case 1: //Comp Length
            if (enactor_stats_->retval = Iteration::Compute_OutputLength(
                frontier_attribute_,
                graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                frontier_queue_->keys[selector]  .GetPointer(util::DEVICE),
                scanned_edges_,
                graph_slice    ->nodes, 
                graph_slice    ->edges,
                context          [peer_][0],
                streams          [peer_],
                gunrock::oprtr::advance::V2V, true)) break;

            frontier_attribute_->output_length.Move(util::DEVICE, util::HOST,1,0,streams[peer_]);
            if (Enactor::SIZE_CHECK)
            {
                Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
            }
            break;

        case 2: //SubQueue Core
            if (Enactor::SIZE_CHECK)
            {
                if (enactor_stats_ -> retval = Check_Record (
                    data_slice, iteration, peer_, 
                    stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                if (to_show[peer_]==false) break;
                Iteration::Check_Queue_Size(
                    thread_num,
                    peer_,
                    frontier_attribute_->output_length[0] + 2,
                    frontier_queue_,
                    frontier_attribute_,
                    enactor_stats_,
                    graph_slice);
            }

            Iteration::SubQueue_Core(
                thread_num,
                peer_,
                frontier_queue_,
                scanned_edges_,
                frontier_attribute_,
                enactor_stats_,
                data_slice,
                s_data_slice[thread_num].GetPointer(util::DEVICE),
                graph_slice,
                &(work_progress[peer_]),
                context[peer_],
                streams[peer_]);

            if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
                frontier_attribute_->queue_index,
                frontier_attribute_->queue_length,
                false,
                streams[peer_],
                true)) break;
            if (num_gpus>1)
                Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
            break;

        case 3: //Copy
            if (num_gpus <=1) 
            {
                if (enactor_stats_-> retval = util::GRError(cudaStreamSynchronize(streams[peer_]), "cudaStreamSynchronize failed",__FILE__, __LINE__)) break;
                Total_Length = frontier_attribute_->queue_length; 
                to_show[peer_]=false;break;
            }
            if (Iteration::HAS_SUBQ || peer_!=0) {
                if (enactor_stats_-> retval = Check_Record(
                    data_slice, iteration, peer_, 
                    stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                if (to_show[peer_] == false) break;
            }

            if (!Enactor::SIZE_CHECK)
            {
                if (Iteration::HAS_SUBQ)
                {
                    if (enactor_stats_->retval = 
                        Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute_->output_length[0]+2, &frontier_queue_->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) break;
                }
                if (frontier_attribute_->queue_length ==0) break;

                if (enactor_stats_->retval = 
                    Check_Size<false, SizeT, VertexId> ("total_queue", Total_Length + frontier_attribute_->queue_length, &data_slice->frontier_queues[num_gpus].keys[0], over_sized, thread_num, iteration, peer_, false)) break;
                
                util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                    data_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE) + Total_Length,
                    frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                    frontier_attribute_->queue_length);
                if (Problem::USE_DOUBLE_BUFFER)
                    util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                        data_slice->frontier_queues[num_gpus].values[0].GetPointer(util::DEVICE) + Total_Length,
                        frontier_queue_->values[selector].GetPointer(util::DEVICE),
                        frontier_attribute_->queue_length);
            }

            Total_Length += frontier_attribute_->queue_length;
            break;

        case 4: //End
            data_slice->wait_counter++;
            to_show[peer__]=false;
            break;
        default:
            stages[peer__]--;
            to_show[peer__]=false;
        }

        if (Enactor::DEBUG && !enactor_stats_->retval)
        {
            mssg="stage 0 @ gpu 0, peer_ 0 failed";
            mssg[6]=char(pre_stage+'0');
            mssg[14]=char(thread_num+'0');
            mssg[23]=char(peer__+'0');
            if (enactor_stats_->retval = util::GRError(mssg, __FILE__, __LINE__)) break;
        }
        stages[peer__]++;
        if (enactor_stats_->retval) break;
    }

}

template <typename Enactor>
void FullQ_Thread(ThreadSlice *thread_data)
{
    if (Iteration::HAS_FULLQ)
    {
        peer_               = 0;
        frontier_queue_     = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpus==1)?0:num_gpus]);
        scanned_edges_      = &(data_slice->scanned_edges  [(Enactor::SIZE_CHECK || num_gpus==1)?0:num_gpus]);
        frontier_attribute_ = &(frontier_attribute[peer_]);
        enactor_stats_      = &(enactor_stats[peer_]);
        work_progress_      = &(work_progress[peer_]);
        iteration           = enactor_stats[peer_].iteration;
        frontier_attribute_->queue_offset = 0;
        frontier_attribute_->queue_reset  = true;
        if (!Enactor::SIZE_CHECK) frontier_attribute_->selector     = 0;

        Iteration::FullQueue_Gather(
            thread_num,
            peer_,
            frontier_queue_,
            scanned_edges_,
            frontier_attribute_,
            enactor_stats_,
            data_slice,
            s_data_slice[thread_num].GetPointer(util::DEVICE),
            graph_slice,
            work_progress_,
            context[peer_],
            streams[peer_]); 
        selector            = frontier_attribute[peer_].selector;
        if (enactor_stats_->retval) break;
        
        if (frontier_attribute_->queue_length !=0)
        {
            if (Enactor::DEBUG) {
                mssg = "";
                ShowDebugInfo<Problem>(
                    thread_num,
                    peer_,
                    frontier_attribute_,
                    enactor_stats_,
                    data_slice,
                    graph_slice,
                    work_progress_,
                    mssg,
                    streams[peer_]);
            }

            enactor_stats_->retval = Iteration::Compute_OutputLength(
                frontier_attribute_,
                graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                scanned_edges_,
                graph_slice    ->nodes, 
                graph_slice    ->edges,
                context          [peer_][0],
                streams          [peer_],
                gunrock::oprtr::advance::V2V, true);
            if (enactor_stats_->retval) break;

            frontier_attribute_->output_length.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
            if (Enactor::SIZE_CHECK)
            {
                tretval = cudaStreamSynchronize(streams[peer_]);
                if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}

                Iteration::Check_Queue_Size(
                    thread_num,
                    peer_,
                    frontier_attribute_->output_length[0] + 2,
                    frontier_queue_,
                    frontier_attribute_,
                    enactor_stats_,
                    graph_slice);

            }
            
            Iteration::FullQueue_Core(
                thread_num,
                peer_,
                frontier_queue_,
                scanned_edges_,
                frontier_attribute_,
                enactor_stats_,
                data_slice,
                s_data_slice[thread_num].GetPointer(util::DEVICE),
                graph_slice,
                work_progress_,
                context[peer_],
                streams[peer_]); 
            if (enactor_stats_->retval) break;
            if (!Enactor::SIZE_CHECK)
            {
                if (enactor_stats_->retval = 
                    Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute->output_length[0]+2, &frontier_queue_->keys[selector^1], over_sized, thread_num, iteration, peer_, false)) break;
            }
            selector = frontier_attribute[peer_].selector;
            Total_Length = frontier_attribute[peer_].queue_length;
        } else {
            Total_Length = 0;
            for (peer__=0;peer__<num_gpus;peer__++)
                data_slice->out_length[peer__]=0;
        }
        if (Enactor::DEBUG) {printf("%d\t %lld\t \t Fullqueue finished. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
        frontier_queue_ = &(data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus]);
        if (num_gpus==1) data_slice->out_length[0]=Total_Length;
    }
    
    if (num_gpus > 1)
    {
        Iteration::Iteration_Update_Preds(
            graph_slice,
            data_slice,
            &frontier_attribute[0],
            &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus],
            Total_Length,
            streams[0]);
        Iteration::template Make_Output <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
            thread_num,
            Total_Length,
            num_gpus,
            &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpus],
            &data_slice->scanned_edges[0],
            &frontier_attribute[0],
            enactor_stats,
            &problem->data_slices[thread_num],
            graph_slice,
            &work_progress[0],
            context[0],
            streams[0]);
    } else data_slice->out_length[0]= Total_Length;

    for (peer_=0;peer_<num_gpus;peer_++)
        frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];

}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
