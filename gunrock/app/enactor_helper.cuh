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

/*
 * @brief
 *
 * @tparam SizeT
 * @tparam DataSlice
 *
 * @param[in] enactor_stats Pointer to the enactor stats.
 * @param[in] frontier_attribute Pointer to the frontier attribute.
 * @param[in] data_slice Pointer to the data slice we process on.
 * @param[in] num_gpus Number of GPUs used for testing.
 */
template <typename ThreadSlice>
bool All_Done(typename ThreadSlice::Enactor *enactor,
              int      gpu_num  = 0)
{
    typedef typename ThreadSlice::Enactor Enactor;
    static int pre_gpu_num = -1;
    static int pre_k = -1;
    static int pre_i = -1;
    static int pre_size = 0;
    static int last_repeat = 0;

    EnactorSlice<Enactor> *enactor_slices 
        = (EnactorSlice<Enactor>*) enactor->enactor_slices;
    ThreadSlice *thread_slices = (ThreadSlice*) enactor->thread_slices;
    int occu_size = 0;
 
    //printf("\t \t \t Checking done.\n");fflush(stdout);
 
    for (int i=0; i<enactor->num_threads; i++)
    {
        cudaError_t retval = thread_slices[i].retval;
        if (retval == cudaSuccess) continue;
        printf("(CUDA error %d @ GPU %d, thread %d: %s\nlast_repeat = %d\n",
            retval, thread_slices[i].gpu_num, 
            thread_slices[i].thread_num,
            cudaGetErrorString(retval),
            last_repeat); 
        fflush(stdout);
        return true;
    }   

    for (int t=0; t<2; t++)
    for (int gpu_num = 0; gpu_num < enactor->num_gpus; gpu_num++)
    {
        EnactorSlice<Enactor> *enactor_slice = &enactor_slices[gpu_num]; 
        for (int i=0; i<2; i++)
        {
            occu_size = enactor_slice -> input_queues[i].GetOccuSize();
            if (occu_size == 0) continue;
            if (pre_gpu_num != gpu_num || pre_k !=1 
                || pre_i != i || pre_size != occu_size)
            {
                printf("%d\t \t \t Not done, input_queues[%d].occu_size = %d, "
                    "last_repeat = %d\n", 
                    gpu_num, i, occu_size, last_repeat);
                fflush(stdout);
                pre_gpu_num = gpu_num;
                pre_k = 1;
                pre_i = i;
                pre_size = occu_size;
                last_repeat = 0;
            } else last_repeat ++;
            return false;
        }

        for (int stream_num = 0; stream_num < enactor->num_subq__streams; stream_num++)
        {
             typename Enactor::FrontierA *frontier_attribute
                = enactor_slice -> subq__frontier_attributes + stream_num;
            if (frontier_attribute -> queue_length != 0 ||
                frontier_attribute -> has_incoming)
            {
                if (pre_gpu_num != gpu_num || pre_k != 2
                    || pre_i != stream_num 
                    || pre_size != frontier_attribute -> queue_length)
                {
                    printf("%d\t \t %d\t Not done, subq__length = %lld, "
                        "last_repeat = %lld\n",
                        gpu_num, stream_num, 
                        (long long)frontier_attribute -> queue_length,
                        (long long)last_repeat);
                    fflush(stdout);
                    pre_gpu_num = gpu_num;
                    pre_k = 2;
                    pre_i = stream_num;
                    pre_size = frontier_attribute -> queue_length;
                    last_repeat = 0;
                } else last_repeat ++;
                return false;
            }
        }

        occu_size = enactor_slice -> subq__queue.GetOccuSize();
        if (occu_size != 0)
        {
            if (pre_gpu_num != gpu_num || pre_k != 3 || pre_size != occu_size)
            {
                printf("%d\t \t \t Not done, subq__queue.occu_size = %d, "
                    "last_repeat = %d\n", 
                    gpu_num, occu_size, last_repeat);
                fflush(stdout);
                pre_gpu_num = gpu_num;
                pre_k = 3;
                pre_size = occu_size;
                last_repeat = 0;
            } else last_repeat ++;
            return false;
        }

        occu_size = enactor_slice -> fullq_queue.GetOccuSize();
        if (occu_size != 0)
        {
            if (pre_gpu_num != gpu_num || pre_k != 4 || pre_size != occu_size)
            {
                printf("%d\t \t \t Not done, fullq_queue.occu_size = %d, "
                    "last_repeat = %d\n",
                    gpu_num, occu_size, last_repeat);
                fflush(stdout);
                pre_gpu_num = gpu_num;
                pre_k = 4;
                pre_size = occu_size;
                last_repeat = 0;
            } else last_repeat ++;
            return false;
        }
 
        for (int stream_num = 0; stream_num < enactor->num_fullq_stream ; stream_num++)
        {
            typename Enactor::FrontierA *frontier_attribute
                = enactor_slice -> fullq_frontier_attribute  + stream_num;
            if (frontier_attribute -> queue_length != 0 ||
                frontier_attribute -> has_incoming)
            {
                if (pre_gpu_num != gpu_num || pre_k != 5 || pre_i != stream_num
                    || pre_size != frontier_attribute -> queue_length)
                {
                    printf("%d\t \t %d\t Not done, fullq_length = %lld, "
                        "last_repeat = %lld\n",
                        gpu_num, stream_num, 
                        (long long)frontier_attribute -> queue_length,
                        (long long)last_repeat);   
                    fflush(stdout);
                    pre_gpu_num = gpu_num;
                    pre_k = 5;
                    pre_i = stream_num;
                    pre_size = frontier_attribute -> queue_length;
                    last_repeat = 0;
                } else last_repeat ++;
                return false;
            }
        }

        occu_size = enactor_slice -> fullq_queue.GetOccuSize();
        if (occu_size != 0)
        {
            if (pre_gpu_num != gpu_num || pre_k != 6 || pre_size != occu_size)
            {
                printf("%d\t \t \t Not done, fullq_queue.occu_size = %d, "
                    "last_repeat = %d\n",
                    gpu_num, occu_size, last_repeat);
                fflush(stdout);
                pre_gpu_num = gpu_num;
                pre_k = 6;
                pre_size = occu_size;
                last_repeat = 0;
            } else last_repeat ++;
            return false;
        }

        occu_size = enactor_slice -> outpu_queue.GetOccuSize();
        if (occu_size != 0)
        {
            if (pre_gpu_num != gpu_num || pre_k != 7 || pre_size != occu_size)
            {
                printf("%d\t \t \t Not done, outpu_queue.occu_size = %d, "
                    "last_repeat = %d\n", 
                    gpu_num, occu_size, last_repeat);
                fflush(stdout);
                pre_gpu_num = gpu_num;
                pre_k = 7;
                pre_size = occu_size;
                last_repeat = 0;
            } else last_repeat ++;
            return false;
        }
    }

    printf("%d\t All_Done. last_repeat = %d\n", gpu_num, last_repeat); 
    fflush(stdout);
    return true;
} 

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam Type
 *
 * @param[in] name
 * @param[in] target_length
 * @param[in] array
 * @param[in] oversized
 * @param[in] thread_num
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] keep_content
 *
 * \return cudaError_t object Indicates the success of all CUDA calls.
 */
template <
    //bool     SIZE_CHECK,
    typename SizeT,
    typename Type>
cudaError_t Check_Size(
    bool        size_check,
    const char *name,
    SizeT       target_length,
    util::Array1D<SizeT, Type> 
               *array,
    bool       &oversized,
    int         gpu_num      = -1,
    long long   iteration    = -1,
    int         stream_num   = -1,
    bool        keep_content = false)
{
    cudaError_t retval = cudaSuccess;

    if (target_length > array->GetSize())
    {
        printf("%d\t %lld\t %d\t %s \t oversize :\t %lld ->\t %lld\n",
            gpu_num, iteration, stream_num, name,
            (long long)array->GetSize(), (long long)target_length);
        fflush(stdout);
        oversized=true;
        if (size_check)
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

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam VertexId
 * @tparam Value
 * @tparam GraphSlice
 * @tparam DataSlice
 * @tparam num_vertex_associate
 * @tparam num_value__associate
 *
 * @param[in] gpu
 * @param[in] peer
 * @param[in] array
 * @param[in] queue_length
 * @param[in] enactor_stats
 * @param[in] data_slice_l
 * @param[in] data_slice_p
 * @param[in] graph_slice_l Graph slice local
 * @param[in] graph_slice_p
 * @param[in] stream CUDA stream.
 */
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
    long long      iteration             =   request -> iteration;
    SizeT          length                =   request -> length;
    EnactorSlice  *enactor_slices        = (EnactorSlice*) enactor->enactor_slices;
    EnactorSlice  *s_enactor_slice       = &(enactor_slices[s_gpu_num]);
    EnactorSlice  *t_enactor_slice       = &(enactor_slices[t_gpu_num]);
    CircularQueue *s_queue               = &(s_enactor_slice -> outpu_queue);             // source output cq
    CircularQueue *t_queue               = &(t_enactor_slice -> input_queues[(iteration+1)%2]); // target input cq
    SizeT          num_vertex_associates =   request -> num_vertex_associates;
    SizeT          num_value__associates =   request -> num_value__associates;
    VertexId      *s_vertices            =   request -> vertices;
    VertexId     **s_vertex_associates   =   request -> vertex_associates;
    Value        **s_value__associates   =   request -> value__associates;
    cudaEvent_t    event                 =   request -> event;
    VertexId      *t_vertices            =   NULL;
    VertexId      *t_vertex_associates[Enactor::NUM_VERTEX_ASSOCIATES];
    Value         *t_value__associates[Enactor::NUM_VALUE__ASSOCIATES];
    SizeT          s_offset              =   request -> offset;
    SizeT          t_offset              =   0;
    cudaError_t    retval                =   cudaSuccess;
    cudaStream_t   s_stream              =   request -> stream;
    cudaStream_t   t_stream              =   t_enactor_slice -> input_streams[0];
    cudaEvent_t    event2;
 
    printf("%d\t %lld\t %d\t PushNeibor\t To, length = %lld\n",
        request -> gpu_num, iteration, request -> peer, (long long)length);
    fflush(stdout);
    printf("%d\t %lld\t %d\t PushNeibor\t From, length = %lld\n",
        request -> peer, iteration+1, request -> gpu_num, (long long)length);
    fflush(stdout);

    if (length > 0)
        if (retval = util::GRError(cudaStreamWaitEvent(s_stream, event, 0),
            "cudaStreamWaitEvent failed", __FILE__, __LINE__)) return retval;

    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pushing keys", s_vertices, length, request->gpu_num, iteration, request -> peer, s_stream);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pushing labels", s_vertex_associates[0], length, request->gpu_num, iteration, request -> peer, s_stream);

    if (retval = t_queue->Push_Addr(length, t_vertices, t_offset, 
        num_vertex_associates, num_value__associates, 
        t_vertex_associates, t_value__associates, true)) return retval;

    if (length > 0)
    {
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

        if (retval = s_queue->EventSet(util::CqEvent<SizeT>::Block, s_offset, length, s_stream, 
            false, false, NULL, &event2)) return retval;
        //if (retval = s_queue->EventSet(util::CqEvent<SizeT>::Out, s_offset, length, s_stream)) 
        //    return retval;
        if (retval = t_queue->EventSet(util::CqEvent<SizeT>::In , t_offset, length, t_stream,
            false, true, &event2, NULL)) return retval;
    } else {
        if (retval = s_queue->EventFinish(util::CqEvent<SizeT>::Block, s_offset, length, s_stream))
            return retval;
        //if (retval = s_queue->EventFinish(util::CqEvent<SizeT>::Out, s_offset, length, s_stream))
        //    return retval;
        if (retval = t_queue->EventFinish(util::CqEvent<SizeT>::In , t_offset, length, t_stream))
            return retval;
    }
    return retval;
}

/*
 * @brief Show debug information function.
 *
 * @tparam Problem
 *
 * @param[in] thread_num
 * @param[in] peer_
 * @param[in] frontier_attribute
 * @param[in] enactor_stats
 * @param[in] data_slice
 * @param[in] graph_slice
 * @param[in] work_progress
 * @param[in] check_name
 * @param[in] stream CUDA stream.
 */
template <typename Enactor>
void ShowDebugInfo(
    int           gpu_num,
    int           thread_type, // 0:input, 1:output, 2:subq, 3:fullq
    int           thread_num,
    int           stream_num,
    Enactor      *enactor,
    long long     iteration,
    std::string   check_name = "",
    cudaStream_t  stream = 0) 
{    
    typedef typename Enactor::SizeT    SizeT;
    typedef typename Enactor::VertexId VertexId;
    typedef typename Enactor::Value    Value;
    SizeT queue_length = -1;
    int   stage        = -1;

    EnactorSlice<Enactor> *enactor_slice 
        = ((EnactorSlice<Enactor>*) enactor->enactor_slices) + gpu_num;
    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
    //if (frontier_attribute->queue_reset)
    switch (thread_type)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        queue_length = enactor_slice->subq__frontier_attributes[stream_num].queue_length;
        stage        = enactor_slice->subq__stages    [stream_num];
        break;
    case 3:
        queue_length = enactor_slice->fullq_frontier_attribute [stream_num].queue_length;
        stage        = enactor_slice->fullq_stage     [stream_num];
        break;
    default:
        break;
    }
    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %lld\n", 
        gpu_num, iteration, stream_num, 
        stage, check_name.c_str(), (long long)queue_length);
    fflush(stdout);
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

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage
 * @param[in] stream CUDA stream.
 */
template <typename EnactorSlice>
cudaError_t Set_Record(
    EnactorSlice *enactor_slice,
    int thread_type,
    int iteration,
    int stream_num,
    int stage)
{
    cudaError_t retval = cudaSuccess;
    if (thread_type == 2) // subq
    {
        //printf("subq__events = %d, event_set = %s\n",
        //    enactor_slice -> subq__events[iteration%4][stream_num][stage],
        //    enactor_slice -> subq__event_sets[iteration%4][stream_num][stage] ? "true" : "false"); fflush(stdout);
        retval = cudaEventRecord(
            enactor_slice -> subq__events[iteration%4][stream_num][stage],
            enactor_slice -> subq__streams[stream_num]);
        enactor_slice -> subq__event_sets[iteration%4][stream_num][stage] = true;
    } else if (thread_type == 3) // fullq
    {
        retval = cudaEventRecord(
            enactor_slice -> fullq_event [iteration%4][stream_num][stage],
            enactor_slice -> fullq_stream[stream_num]);
        enactor_slice -> fullq_event_set [iteration%4][stream_num][stage] = true;
    }
    return retval;
}

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage_to_check
 * @param[in] stage
 * @param[in] to_show
 */
template <typename EnactorSlice>
cudaError_t Check_Record(
    EnactorSlice *enactor_slice,
    int thread_type,
    int iteration,
    int stream_num,
    int stage_to_check,
    int &stage,
    bool &to_show)
{
    cudaError_t retval = cudaSuccess;
    to_show = true;
    bool *event_set = NULL;
    cudaEvent_t event;

    if (thread_type == 2) 
    { // subq
        event     = enactor_slice ->
            subq__events    [iteration%4][stream_num][stage_to_check];
        event_set = enactor_slice -> 
            subq__event_sets[iteration%4][stream_num] + stage_to_check;
    } else if (thread_type == 3) 
    { // fullq
        event     = enactor_slice ->
            fullq_event     [iteration%4][stream_num][stage_to_check];
        event_set = enactor_slice ->
            fullq_event_set [iteration%4][stream_num] + stage_to_check;
    }
    if (!event_set[0])
    {   
        to_show = false;
    } else {
        retval = cudaEventQuery(event);
        if (retval == cudaErrorNotReady)
        {   
            to_show=false;
            retval = cudaSuccess; 
        } else if (retval == cudaSuccess)
        {
            event_set[0] = false;
        }
    }
    return retval;
}

template <typename Enactor>
void Show_Mem_Stats_(
    Enactor *enactor)
{
    //typedef typename EnactorSlice::Enactor Enactor;
    //typedef typename EnactorSlice<Enactor> EnactorSlice;
    typedef typename Enactor::Problem Problem;
    double factors[5][4];
    for (int i=0; i<5; i++)
    for (int j=0; j<4; j++)
        factors[i][j] = -1;

    printf("\nGPU\t      Queue\tType\tStream\tSize\tBase\tFactor\tTemp Size\tTemp Factor\n");
    for (int gpu = 0; gpu < enactor->num_gpus; gpu ++)
    {
        EnactorSlice<Enactor> *enactor_slice = (EnactorSlice<Enactor>*)(enactor -> enactor_slices) + gpu;
        for (int queue_num = 0; queue_num < 5; queue_num ++)
        {
            std::string queue_name = "";
            Problem *problem = (Problem*) enactor -> problem;
            typename Enactor::SizeT base = problem -> sub_graphs[gpu].nodes;
            typename Enactor::CircularQueue *cq = NULL;
            typename Enactor::FrontierT     *ft = NULL;
            int stream_limit = 0;

            switch (queue_num)
            {
            case  0: 
                queue_name = "Sub   Queue "; 
                cq = &(enactor_slice -> subq__queue    ); 
                ft =   enactor_slice -> subq__frontiers + 0;
                stream_limit = enactor_slice -> num_subq__streams;
                break;

            case  1: 
                queue_name = "Full  Queue "; 
                cq = &(enactor_slice -> fullq_queue    ); 
                ft =   enactor_slice -> fullq_frontier + 0;
                stream_limit = enactor_slice -> num_fullq_stream;
                if (stream_limit <1) stream_limit = 1;
                break;

            case  2: 
                queue_name = "Input Queue0"; 
                cq = &(enactor_slice -> input_queues[0]); 
                stream_limit = 1;
                break;

            case  3: 
                queue_name = "Input Queue1"; 
                cq = &(enactor_slice -> input_queues[1]); 
                stream_limit = 1;
                break;
                
            case  4: 
                queue_name = "OutputQueue "; 
                cq = &(enactor_slice -> outpu_queue    ); 
                stream_limit = 1;
                break;

            default:
                break; 
            }
            
            for (int stream_num = 0; stream_num < stream_limit; stream_num ++)
            for (int type_num = 0; type_num < 3; type_num++)
            {
                typename Enactor::SizeT size = -1, temp_size = -1;
                if (queue_num !=0 && queue_num !=1 && type_num >0) continue;
                if (type_num == 0 && stream_num > 0) continue;

                std::string type_name = "";
                switch (type_num)
                {
                case  0: 
                    type_name = "CQ";
                    if (cq != NULL) size = cq -> GetCapacity();
                    if (cq != NULL) temp_size = cq -> GetTempCapacity();
                    break;

                case  1: 
                    type_name = "FT0";
                    if (ft!= NULL) size = ft[stream_num].keys[0].GetSize();
                    break;

                case  2:
                    type_name = "FT1";
                    if (ft!= NULL) size = ft[stream_num].keys[1].GetSize();
                    break;

                default:
                    break;
                }

                if (size < 0) continue;

                double factor = 1.0 * size / base;
                if (queue_num != 3 && queue_num != 2 &&
                    factor > factors[queue_num][type_num])
                    factors[queue_num][type_num] = factor;
                else if (queue_num == 3 || queue_num == 2)
                {
                    if (type_num == 0 && factor > factors[2][queue_num-1])
                        factors[2][queue_num-1] = factor;
                }

                printf("%d\t%s\t%s\t", gpu, queue_name.c_str(), type_name.c_str());
                if (queue_num == 0 && type_num > 0)
                    printf(" %d\t", stream_num);
                else printf("\t");
                printf(" %lld\t %lld\t %.3lf", (long long)size, (long long)base, factor);

                if (temp_size != -1)
                {
                    factor = 1.0 * temp_size / size;
                    if (queue_num != 3 && factor > factors[queue_num][3])
                        factors[queue_num][3] = factor;
                    else if (queue_num == 3 && factor > factors[2][3])
                        factors[2][3] = factor;
                    printf("\t %lld\t %.3lf", (long long)temp_size, factor);
                    
                }
                printf("\n");
            }
        }
    }

    char argument[1024] = "";

    printf("\n\t      Queue\tCQ factor\tFT0 factor\tFT1 factor\tTemp factor\n");
    for (int queue_num = 0; queue_num < 5; queue_num ++)
    {
        if (queue_num == 3) continue;
        std::string queue_name = "";
        std::string argu_name = "";
        switch (queue_num)
        {
        case  0: 
            queue_name = "Sub   Queue ";
            argu_name  = "subq";
            break;

        case  1: 
            queue_name = "Full  Queue "; 
            argu_name  = "fullq";
            break;

        case  2: 
            queue_name = "Input Queue ";
            argu_name  = "input";
            break;

        case  3: 
            queue_name = "Input Queue1"; 
            argu_name  = "input";
            break;
            
        case  4: 
            queue_name = "OutputQueue "; 
            argu_name  = "output";
            break;

        default:
            break; 
        }

        printf("\t%s", queue_name.c_str());
        for (int type_num = 0; type_num < 4; type_num ++)
        {
            if (factors[queue_num][type_num] > 0)
            {
                printf("\t %.3lf", factors[queue_num][type_num]);
                sprintf(argument, "%s --%s_factor", argument, argu_name.c_str());
                //if (queue_num == 2 || queue_num == 3)
                //    sprintf(argument, "%s%d", argument, queue_num == 2? 0 : 1);
                if (type_num > 0)
                    sprintf(argument, "%s%d", argument, type_num -1);
                sprintf(argument, "%s=%.3lf", argument, factors[queue_num][type_num] * 1.10);
            }
            else printf("\t      ");
        }
        printf("\n");
    }
    printf("Suggest factors: %s\n", argument);
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
