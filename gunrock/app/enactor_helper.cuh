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

template <typename SizeT, typename DataSlice>
bool All_Done(EnactorStats                    *enactor_stats,
              FrontierAttribute<SizeT>        *frontier_attribute, 
              util::Array1D<SizeT, DataSlice> *data_slice, 
              int                              num_gpus)
{   
    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (enactor_stats[gpu].retval!=cudaSuccess)
    {   
        printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
        return true;
    }   

    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (frontier_attribute[gpu].queue_length!=0 || frontier_attribute[gpu].has_incoming)
    {
        //printf("frontier_attribute[%d].queue_length = %d\n",gpu,frontier_attribute[gpu].queue_length);   
        return false;
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    for (int i=0;i<2;i++)
    if (data_slice[gpu]->in_length[i][peer]!=0)
    {
        //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
        return false;
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    if (data_slice[gpu]->out_length[peer]!=0)
    {
        //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]);
        return false;
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
    PushRequest<Enactor> *request,
    Enactor::Problem     *problem)
{
    typedef Enactor::Problem       Problem;
    typedef Enactor::SizeT         SizeT;
    typedef Enactor::VertexId      VertexId;
    typedef Enactor::Value         Value;
    typedef Problem::DataSlice     DataSlice;
    typedef Enactor::CircularQueue CircularQueue;

    int           *s_gpu_num             =   request -> gpu_num;
    int           *t_gpu_num             =   request -> peer;
    SizeT          iteration             =   request -> iteration;
    SizeT          length                =   request -> length;
    DataSlice     *s_data                = &(problem -> data_slice[s_gpu_num][0]);
    DataSlice     *t_data                = &(problem -> data_slice[t_gpu_num][0]);
    CircularQueue *s_cq                  = &(s_data  -> output_cq);             // source output cq
    CircularQueue *t_cq                  = &(t_data  -> input_cq[iteration%2]); // target input cq
    SizeT          num_vertex_associates =   request -> num_vertex_associates;
    SizeT          num_value__associates =   request -> num_value__associates;
    VertexId      *s_vertices            =   request -> vertices;
    VertexId     **s_vertex_associates   =   request -> vertex_associates;
    Value        **s_value__associates   =   request -> value__associates;
    VertexId      *t_vertices            =   NULL;
    VertexId      *t_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value         *t_value__associates[Enactor::NUM_Value__ASSOCIATES];
    SizeT          s_offset              =   request -> offset;
    SizeT          t_offset              =   0;
    cudaError_t    retval                =   cudaSuccess;
    cudaStream_t   s_stream              =   request -> stream;
    cudaStream_t   t_stream              ; // =?

    if (retval = t_cq->Push_Addr(length, t_vertices, t_offset, 
        num_vertex_associates, num_value__associates, 
        t_vertex_associates, t_value__associates, true)) return retval;

    if (retval = util::GRError(cudaMemcpyAsync(
        t_vertices, s_vertices, sizeof(VertexId) * length,
        cudaMemcpyDefault, stream),
        "cudaMemcpyAsync vertices failed", __FILE__, __LINE__)) return retval;

    for (SizeT i=0; i<num_vertex_associates; i++)
    {
        if (retval = util::GRError(cudaMemcpyAsync(
            t_vertex_associates[i], s_vertex_associates[i], sizeof(VertexId) * length,
            cudaMemcpyDefault, stream),
            "cudaMemcpyAsync vertex_associates failed", __FILE__, __LINE__)) return retval;
    }

    for (SizeT i=0; i<num_value__associates; i++)
    {
        if (retval = util::GRError(cudaMemcpyAsync(
            t_value__associates[i], s_value__associates[i], sizeof(VertexId) * length,
            cudaMemcpyDefault, stream),
            "cudaMemcpyAsync value__associates failed", __FILE__, __LINE__)) return retval;
    }

    if (retval = s_cq->EventSet(1, t_offset, length, s_stream)) return retval;
    if (retval = t_cq->EventSet(0, s_offset, length, t_stream, false, true)) return retval;
    return retval;
}

template <typename Problem>
void ShowDebugInfo(
    int           thread_num,
    int           peer_,
    FrontierAttribute<typename Problem::SizeT>      
                 *frontier_attribute,
    EnactorStats *enactor_stats,
    typename Problem::DataSlice  
                 *data_slice,
    GraphSlice<typename Problem::SizeT, typename Problem::VertexId, typename Problem::Value> 
                 *graph_slice,
    util::CtaWorkProgressLifetime 
                 *work_progress,
    std::string   check_name = "",
    cudaStream_t  stream = 0) 
{    
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    SizeT queue_length;

    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
    //if (frontier_attribute->queue_reset) 
        queue_length = frontier_attribute->queue_length;
    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
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
