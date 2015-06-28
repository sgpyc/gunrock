// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_helper.cuh
 *
 * @brief helper functions for enactor base
 */

#pragma once

#include <gunrock/app/enactor_slice.cuh>

namespace gunrock {
namespace app {

template <typename Enactor>
bool All_Done(Enactor *enactor,
              int      gpu_num  = 0)
{
    Enactor_Slice<Enactor> *enactor_slices 
        = (Enactor_Slice<Enactor>*) enactor->enactor_slices;
    for (int i=0; i<enactor->num_threads; i++)
    {
        cudaError_t retval = enactor->thread_slices[i].retval;
        if (retval == cudaSuccess) continue;
        printf("(CUDA error %d @ GPU %d, thread %d: %s\n",
            retval, enactor->thread_slices[i].gpu_num, 
            enactor->thread_slices[i].thread_num,
            cudaGetErrorString(retval)); 
        fflush(stdout);
        return true;
    }   

    for (int gpu=0; gpu < enactor->num_gpus; gpu++)
    {
        Enactor_Slice<Enactor> *enactor_slice = &enactor_slices[gpu];
        for (int stream=0; stream < enactor->num_subq__streams + enactor->num_fullq_streams; stream++)
        {
            Enactor::FrontierA *frontier_attribute = (stream < enactor->num_subq_streams) ? 
                enactor_slice->subq__frontier_attributes + stream :
                enactor_slice->fullq_frontier_attributes + stream - enactor->num_subq_streams;
            if (frontier_attribute->queue_length != 0 ||
                frontier_attribute->has_incoming)
            {
                //printf("frontier_attribute[%d].queue_length = %d\n",gpu,frontier_attribute[gpu].queue_length);   
                return false;
            }
        }

        for (int i=0; i<2; i++)
        if (!enactor_slice -> input_queues[i].empty())
        {
            //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
            return false;
        }

        if (!enactor_slice -> outpu_queue.empty())
        {
            //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]);
            return false;
        }

        if (!enactor_slice -> subq__queue.empty())
        {
            return false;
        }

        if (!enactor_slice -> fullq_queue.empty())
        {
            return false;
        }
    }

    return true;
} 

template <
    bool     SIZE_CHECK,
    typename SizeT,
    typename Type>
cudaError_t Check_Size(
    const char *name,
    SizeT       target_length,
    util::Array1D<SizeT, Type> 
               *array,
    bool       &oversized,
    int         thread_num = -1,
    int         iteration  = -1,
    int         peer_      = -1,
    bool        keep_content = false)
{
    cudaError_t retval = cudaSuccess;

    if (target_length > array->GetSize())
    {
        printf("%d\t %d\t %d\t %s \t oversize :\t %d ->\t %d\n",
            thread_num, iteration, peer_, name,
            array->GetSize(), target_length);
        fflush(stdout);
        oversized=true;
        if (SIZE_CHECK)
        {
            if (array->GetSize() != 0) retval = array->EnsureSize(target_length, keep_content);
            else retval = array->Allocate(target_length, util::DEVICE);
        } else {
            char temp_str[]=" oversize", str[256];
            memcpy(str, name, sizeof(char) * strlen(name));
            memcpy(str + strlen(name), temp_str, sizeof(char) * strlen(temp_str));
            str[strlen(name)+strlen(temp_str)]='\0';
            retval = util::GRError(cudaErrorLaunchOutOfResources, str, __FILE__, __LINE__);
        }
    }
    return retval;
}

template <typename Enactor>
cudaError_t PushNeibor(
    typename Enactor::PRequest *request,
    Enactor           *enactor)
{
    typedef typename Enactor::SizeT         SizeT;
    typedef typename Enactor::VertexId      VertexId;
    typedef typename Enactor::Value         Value;
    typedef EnactorSlice<Enactor>  EnactorSlice;
    typedef typename Enactor::CircularQueue CircularQueue;

    int            s_gpu_num             =   request -> gpu_num;
    int            t_gpu_num             =   request -> peer;
    SizeT          iteration             =   request -> iteration;
    SizeT          length                =   request -> length;
    EnactorSlice  *enactor_slices        = (EnactorSlice*) enactor->enactor_slices;
    EnactorSlice  *s_enactor_slice       = &(enactor_slices[s_gpu_num]);
    EnactorSlice  *t_enactor_slice       = &(enactor_slices[t_gpu_num]);
    CircularQueue *s_queue               = &(s_enactor_slice -> outpu_queue);             // source output cq
    CircularQueue *t_queue               = &(t_enactor_slice -> input_queues[iteration%2]); // target input cq
    SizeT          num_vertex_associates =   request -> num_vertex_associates;
    SizeT          num_value__associates =   request -> num_value__associates;
    VertexId      *s_vertices            =   request -> vertices;
    VertexId     **s_vertex_associates   =   request -> vertex_associates;
    Value        **s_value__associates   =   request -> value__associates;
    VertexId      *t_vertices            =   NULL;
    VertexId      *t_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value         *t_value__associates[Enactor::NUM_VALUE__ASSOCIATES];
    SizeT          s_offset              =   request -> offset;
    SizeT          t_offset              =   0;
    cudaError_t    retval                =   cudaSuccess;
    cudaStream_t   s_stream              =   request -> stream;
    cudaStream_t   t_stream              =   t_enactor_slice -> input_streams[0];

    if (retval = t_queue->Push_Addr(length, t_vertices, t_offset, 
        num_vertex_associates, num_value__associates, 
        t_vertex_associates, t_value__associates, true)) return retval;

    if (retval = util::GRError(cudaMemcpyAsync(
        t_vertices, s_vertices, sizeof(VertexId) * length,
        cudaMemcpyDefault, s_stream),
        "cudaMemcpyAsync vertices failed", __FILE__, __LINE__)) return retval;

    for (SizeT i=0; i<num_vertex_associates; i++)
    {
        if (retval = util::GRError(cudaMemcpyAsync(
            t_vertex_associates[i], s_vertex_associates[i], sizeof(VertexId) * length,
            cudaMemcpyDefault, s_stream),
            "cudaMemcpyAsync vertex_associates failed", __FILE__, __LINE__)) return retval;
    }

    for (SizeT i=0; i<num_value__associates; i++)
    {
        if (retval = util::GRError(cudaMemcpyAsync(
            t_value__associates[i], s_value__associates[i], sizeof(VertexId) * length,
            cudaMemcpyDefault, s_stream),
            "cudaMemcpyAsync value__associates failed", __FILE__, __LINE__)) return retval;
    }

    if (retval = s_queue->EventSet(1, t_offset, length, s_stream)) return retval;
    if (retval = t_queue->EventSet(0, s_offset, length, t_stream, false, true)) return retval;
    return retval;
}

template <typename Enactor>
void ShowDebugInfo(
    int           gpu_num,
    int           thread_type, // 0:input, 1:output, 2:subq, 3:fullq
    int           thread_num,
    int           stream_num,
    Enactor      *enactor,
    std::string   check_name = "",
    cudaStream_t  stream = 0) 
{    
    typedef typename Enactor::SizeT    SizeT;
    typedef typename Enactor::VertexId VertexId;
    typedef typename Enactor::Value    Value;
    SizeT queue_length = -1;
    int   stage        = -1;

    EnactorSlice<Enactor> *enactor_slice 
        = ((EnactorSlice<Encator>*) enactor->enactor_slices) + gpu_num;
    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
    //if (frontier_attribute->queue_reset)
    switch (thrad_type)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        queue_length = enactor_slice->subq__attributes[stream_num].queue_length;
        stage        = enactor_slice->subq__stages    [stream_num];
        break;
    case 3:
        queue_length = enactor_slice->fullq_attribute [stream_num].queue_length;
        stage        = enactor_slice->fullq_stage     [stream_num];
        break;
    default:
        break;
    }
    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", 
        gpu_num, enactor->thread_slices[thread_num]->iteration, stream_num, 
        stage, check_name.c_str(), queue_length);fflush(stdout);
    //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), data_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,peer_, stream);
    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::MARK_PREDECESSORS)
    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::ENABLE_IDEMPOTENCE)
    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
}  

template <typename DataSlice>
cudaError_t Set_Record(
    DataSlice *data_slice,
    int iteration,
    int peer_,
    int stage,
    cudaStream_t stream)
{
    cudaError_t retval = cudaEventRecord(data_slice->events[iteration%4][peer_][stage],stream);
    data_slice->events_set[iteration%4][peer_][stage]=true;
    return retval;
}

template <typename DataSlice>
cudaError_t Check_Record(
    DataSlice *data_slice,
    int iteration,
    int peer_,
    int stage_to_check,
    int &stage,
    bool &to_show)
{
    cudaError_t retval = cudaSuccess;
    to_show = true;                 
    if (!data_slice->events_set[iteration%4][peer_][stage_to_check])
    {   
        to_show = false;
        stage--;
    } else {
        retval = cudaEventQuery(data_slice->events[iteration%4][peer_][stage_to_check]);
        if (retval == cudaErrorNotReady)
        {   
            to_show=false;
            stage--;
            retval = cudaSuccess; 
        } else if (retval == cudaSuccess)
        {
            data_slice->events_set[iteration%4][peer_][stage_to_check]=false;
        }
    }
    return retval;
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
