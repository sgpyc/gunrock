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

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

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
 * @param[in] num_local_gpus Number of local GPUs used for testing.
 * @param[in] num_total_gpus Total number of GPUs
 */
template <typename SizeT, typename DataSlice>
bool All_Done(EnactorStats<SizeT>             *enactor_stats,
              FrontierAttribute<SizeT>        *frontier_attribute,
              util::Array1D<SizeT, DataSlice> *data_slice,
              int                              num_local_gpus,
              int                              num_total_gpus)
{
    for (int gpu = 0; gpu < num_local_gpus * num_total_gpus; gpu++)
    if (enactor_stats[gpu].retval!=cudaSuccess)
    {
        printf("(CUDA error %d @ GPU %d: %s\n",
            enactor_stats[gpu].retval, gpu % num_total_gpus,
            cudaGetErrorString(enactor_stats[gpu].retval));
        fflush(stdout);
        return true;
    }

    for (int gpu = 0; gpu < num_local_gpus * num_total_gpus; gpu++)
    if (frontier_attribute[gpu].queue_length!=0
        || frontier_attribute[gpu].has_incoming)
    {
        //printf("frontier_attribute[%d].queue_length = %d\n",
        //    gpu,frontier_attribute[gpu].queue_length);
        return false;
    }

    for (int gpu  = 0; gpu  < num_local_gpus; gpu++ )
    for (int peer = 1; peer < num_total_gpus; peer++)
    for (int i    = 0; i    < 2       ; i++   )
    if (data_slice[gpu] -> in_length[i][peer] != 0)
    {
        //printf("data_slice[%d]->in_length[%d][%d] = %d\n",
        //    gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
        return false;
    }

    for (int gpu  = 0; gpu  < num_local_gpus; gpu++ )
    for (int peer = 1; peer < num_total_gpus; peer++)
    if (data_slice[gpu] -> out_length[peer] != 0)
    {
        //printf("data_slice[%d]->out_length[%d] = %d\n",
        //    gpu, peer, data_slice[gpu]->out_length[peer]);
        return false;
    }

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
    int         thread_num = -1,
    int         iteration  = -1,
    int         peer_      = -1,
    bool        keep_content = false)
{
    cudaError_t retval = cudaSuccess;

    if (target_length > array->GetSize())
    {
        printf("%d\t %d\t %d\t %s \t oversize :\t %lld ->\t %lld\n",
            thread_num, iteration, peer_, name,
            (long long)array->GetSize(), (long long)target_length);
        //fflush(stdout);
        oversized=true;
        if (size_check)
        {
            if (array->GetSize() != 0)
                retval = array->EnsureSize(target_length, keep_content);
            else retval = array->Allocate(target_length, util::DEVICE);
        } else {
            //char str[256];
            //memcpy(str, name, sizeof(char) * strlen(name));
            //memcpy(str + strlen(name), temp_str, sizeof(char) * strlen(temp_str));
            //str[strlen(name)+strlen(temp_str)]='0';
            //sprintf(str,"%s oversized", name);
            retval = util::GRError(cudaErrorLaunchOutOfResources,
                std::string(name) + " oversized", __FILE__, __LINE__);
        }
    }
    return retval;
}

template <
    typename Enactor,
    typename GraphSliceT,
    typename DataSlice,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
void Pre_Send_Remote(
    Enactor           *enactor,
    int                gpu_rank_local,
    int                peer_gpu_rank,
    typename Enactor::SizeT
                      &queue_length,
    EnactorStats<typename Enactor::SizeT>
                      *enactor_stats,
    util::Array1D<typename Enactor::SizeT, DataSlice>
                      *s_data_slices,
    cudaStream_t       stream,
    float              communicate_multipy)
{
    typedef typename Enactor::VertexId VertexId;
    typedef typename Enactor::SizeT    SizeT;
    typedef typename Enactor::Value    Value;

    cudaError_t &retval = enactor_stats -> retval;
    int mpi_rank       = enactor -> mpi_rank;
    auto &local_data_slice = s_data_slices[gpu_rank_local][0];
    int gpu_rank       = gpu_rank_local + mpi_rank * enactor -> num_local_gpus;
    //int peer_rank      = peer_gpu_rank / enactor -> num_local_gpus;
    //int t              = enactor_stats -> iteration % 2;
    //int tag_base       = gpu_rank * 16 + t * 8;
    int gpu_rank_      = gpu_rank;
    if (peer_gpu_rank > gpu_rank) gpu_rank_ ++;
    int peer_gpu_rank_ = peer_gpu_rank;
    if (peer_gpu_rank < gpu_rank) peer_gpu_rank_ ++;
    bool over_sized = false;

    if (communicate_multipy > 1)
        queue_length *= communicate_multipy;

    //MPI_Request request;
    //MPI_Isend(&queue_length, sizeof(SizeT), MPI_BYTE,
    //    peer_rank, tag_base, MPI_COMM_WORLD, request);
    //local_data_slice -> send_requests[peer_gpu_rank].push_back(mpi_request);

    if (local_data_slice.keys_out[peer_gpu_rank_]
        .GetPointer(util::DEVICE) != NULL)
    if (retval = Check_Size<SizeT, VertexId>(
        enactor -> size_check, "temp_keys_out", queue_length,
        &(local_data_slice.temp_keys_out[peer_gpu_rank_]),
        over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank_))
        return;

    if (retval = Check_Size<SizeT, VertexId>(
        enactor -> size_check, "temp_vertex_associate_out",
        queue_length * NUM_VERTEX_ASSOCIATES,
        &(local_data_slice.vertex_associate_out[peer_gpu_rank_]),
        over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank_))
        return;

    if (retval = Check_Size<SizeT, Value>(
        enactor -> size_check, "temp_value__associate_out",
        queue_length * NUM_VALUE__ASSOCIATES,
        &(local_data_slice.value__associate_out[peer_gpu_rank_]),
        over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank))
        return;

    if (local_data_slice.keys_out[peer_gpu_rank_]
        .GetPointer(util::DEVICE) != NULL)
    if (retval = util::GRError(cudaMemcpyAsync(
        local_data_slice.temp_keys_out[peer_gpu_rank_].GetPointer(util::HOST),
        local_data_slice.     keys_out[peer_gpu_rank_].GetPointer(util::DEVICE),
        queue_length * sizeof(VertexId), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync temp_keys_out D2H failed", __FILE__, __LINE__))
        return;

    if (retval = util::GRError(cudaMemcpyAsync(
        local_data_slice.temp_vertex_associate_out[peer_gpu_rank_].GetPointer(util::HOST),
        local_data_slice.     vertex_associate_out[peer_gpu_rank_].GetPointer(util::DEVICE),
        queue_length * sizeof(VertexId) * NUM_VERTEX_ASSOCIATES,
        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync temp_vertex_associate_out D2H failed", __FILE__, __LINE__))
        return;

    if (retval = util::GRError(cudaMemcpyAsync(
        local_data_slice.temp_value__associate_out[peer_gpu_rank_].GetPointer(util::HOST),
        local_data_slice.     value__associate_out[peer_gpu_rank_].GetPointer(util::DEVICE),
        queue_length * sizeof(VertexId) * NUM_VERTEX_ASSOCIATES,
        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync temp_value__associate_out D2H failed", __FILE__, __LINE__))
        return;
}

template <
    typename Enactor,
    typename GraphSliceT,
    typename DataSlice,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
void Send_Remote(
    Enactor           *enactor,
    int                gpu_rank_local,
    int                peer_gpu_rank,
    typename Enactor::SizeT
                      &queue_length,
    EnactorStats<typename Enactor::SizeT>
                      *enactor_stats,
    util::Array1D<typename Enactor::SizeT, DataSlice>
                      *s_data_slices,
    cudaStream_t       stream,
    float              communicate_multipy)
{
    typedef typename Enactor::VertexId VertexId;
    typedef typename Enactor::SizeT    SizeT;
    typedef typename Enactor::Value    Value;
    cudaError_t &retval = enactor_stats -> retval;
    int mpi_rank = enactor -> mpi_rank;

    auto &local_data_slice = s_data_slices[gpu_rank_local][0];
    int gpu_rank = gpu_rank_local + mpi_rank * enactor -> num_local_gpus;
    int peer_rank = peer_gpu_rank / enactor -> num_local_gpus;
    int t        = enactor_stats -> iteration % 2;
    int tag_base = gpu_rank * 16 + t * 8;
    int gpu_rank_ = gpu_rank;
    if (peer_gpu_rank > gpu_rank)
        gpu_rank_ ++;
    int peer_gpu_rank_ = peer_gpu_rank;
    if (peer_gpu_rank < gpu_rank)
        peer_gpu_rank_ ++;

    if (communicate_multipy > 1)
        queue_length *= communicate_multipy;

    if (local_data_slice.keys_out[peer_gpu_rank_]
        .GetPointer(util::DEVICE) != NULL)
    if (retval = util::Mpi_Isend_Bulk(
        local_data_slice.temp_keys_out[peer_gpu_rank_].GetPointer(util::HOST),
        queue_length, peer_rank, tag_base + 1,
        MPI_COMM_WORLD, local_data_slice.send_requests[peer_gpu_rank_]))
        return;

    if (retval = util::Mpi_Isend_Bulk(
        local_data_slice.temp_vertex_associate_out[peer_gpu_rank_].GetPointer(util::HOST),
        queue_length * NUM_VERTEX_ASSOCIATES, peer_rank, tag_base + 2,
        MPI_COMM_WORLD, local_data_slice.send_requests[peer_gpu_rank_]))
        return;

    if (retval = util::Mpi_Isend_Bulk(
        local_data_slice.temp_value__associate_out[peer_gpu_rank_].GetPointer(util::HOST),
        queue_length * NUM_VALUE__ASSOCIATES, peer_rank, tag_base + 3,
        MPI_COMM_WORLD, local_data_slice.send_requests[peer_gpu_rank_]))
        return;
}


template <
    //bool     SIZE_CHECK,
    //typename VertexId,
    //typename SizeT,
    //typename Value,
    typename Enactor,
    typename GraphSliceT,
    typename DataSlice,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
void Send_Local(
    Enactor           *enactor,
    int                gpu_rank_local,
    int                peer_gpu_rank,
    typename Enactor::SizeT
                       queue_length,
    EnactorStats<typename Enactor::SizeT>
                      *enactor_stats,
    util::Array1D<typename Enactor::SizeT, DataSlice>
                      *s_data_slices,
    //DataSlice         *data_slice_l,
    //DataSlice         *data_slice_p,
    //GraphSliceT       *graph_slice_l,
    //GraphSliceT       *graph_slice_p,
    cudaStream_t       stream,
    float              communicate_multipy)
{
    typedef typename Enactor::VertexId VertexId;
    typedef typename Enactor::SizeT    SizeT;
    typedef typename Enactor::Value    Value;
    cudaError_t &retval = enactor_stats -> retval;
    bool &size_check    = enactor -> size_check;
    int num_local_gpus  = enactor -> num_local_gpus;
    //int mpi_num_tasks
    //    = enactor -> num_total_gpus / num_local_gpus;
    int mpi_rank = enactor -> mpi_rank;

    int gpu_rank  = gpu_rank_local + mpi_rank * num_local_gpus;
    //int peer_rank = peer_gpu_rank / num_local_gpus;

    // Same GPU
    if (gpu_rank == peer_gpu_rank) return;

    int gpu_rank_ = gpu_rank;
    if (peer_gpu_rank > gpu_rank) gpu_rank_ ++;
    int peer_gpu_rank_ = peer_gpu_rank;
    if (peer_gpu_rank < gpu_rank) peer_gpu_rank_ ++;


    int peer_gpu_rank_local = peer_gpu_rank % num_local_gpus;
    int t    = enactor_stats->iteration%2;
    bool to_reallocate = false;
    bool over_sized    = false;
    auto &peer_data_slice  = s_data_slices[peer_gpu_rank_local][0];
    auto &local_data_slice = s_data_slices[     gpu_rank_local][0];

    peer_data_slice.in_length   [t][gpu_rank_]
        = queue_length;
    peer_data_slice.in_iteration[t][gpu_rank_]
        = enactor_stats -> iteration;
    if (queue_length == 0) return;

    if (communicate_multipy > 1)
        queue_length *= communicate_multipy;

    if (local_data_slice.keys_out[peer_gpu_rank_]
        .GetPointer(util::DEVICE) != NULL &&
        peer_data_slice.keys_in[t][gpu_rank_]
        .GetSize() < queue_length)
        to_reallocate = true;
    if (peer_data_slice.vertex_associate_in[t][gpu_rank_]
        .GetSize() < queue_length * NUM_VERTEX_ASSOCIATES)
        to_reallocate = true;
    if (peer_data_slice.value__associate_in[t][gpu_rank_]
        .GetSize() < queue_length * NUM_VALUE__ASSOCIATES)
        to_reallocate = true;

    if (to_reallocate)
    {
        if (size_check)
            if (retval = util::SetDevice(peer_data_slice.gpu_idx))
                return;
        if (local_data_slice.keys_out[peer_gpu_rank_]
            .GetPointer(util::DEVICE) != NULL)
        if (retval = Check_Size<SizeT, VertexId>(
            size_check, "keys_in", queue_length,
            &(peer_data_slice.keys_in[t][gpu_rank_]),
            over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank))
            return;
        if (retval = Check_Size<SizeT, VertexId>(
            size_check, "vertex_associate_in",
            queue_length * NUM_VERTEX_ASSOCIATES,
            &(peer_data_slice.vertex_associate_in[t][gpu_rank_]),
            over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank))
            return;
        if (retval = Check_Size<SizeT, Value>(
            size_check, "value__associate_in",
            queue_length * NUM_VALUE__ASSOCIATES,
            &(peer_data_slice.value__associate_in[t][gpu_rank_]),
            over_sized, gpu_rank_local, enactor_stats -> iteration, peer_gpu_rank))
            return;
        if (size_check)
            if (retval = util::SetDevice(local_data_slice.gpu_idx))
                return;
    }

    if (local_data_slice.keys_out[peer_gpu_rank_]
        .GetPointer(util::DEVICE) != NULL)
    {
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys_out",
        //    local_data_slice.keys_out[peer_gpu_rank_].GetPointer(util::DEVICE),
        //    queue_length, gpu, enactor_stats -> iteration, peer_, stream);

        if (retval = util::GRError(cudaMemcpyAsync(
             peer_data_slice.keys_in [t][  gpu_rank_].GetPointer(util::DEVICE),
            local_data_slice.keys_out[peer_gpu_rank_].GetPointer(util::DEVICE),
            sizeof(VertexId) * queue_length, cudaMemcpyDeviceToDevice, stream),
            "cudamemcpyPeer keys failed", __FILE__, __LINE__))
            return;
        //printf("%d @ %p -> %d @ %p, size = %d\n",
        //    gpu , local_data_slice.keys_out[peer_gpu_rank_].GetPointer(util::DEVICE),
        //    peer,  peer_data_slice.keys_in[t][gpu_rank_].GetPointer(util::DEVICE),
        //    sizeof(VertexId) * queue_length);
    } else {
        //printf("push key skiped\n");
    }

    if (NUM_VERTEX_ASSOCIATES != 0)
    if (retval = util::GRError(cudaMemcpyAsync(
        peer_data_slice .vertex_associate_in [t][  gpu_rank_]
            .GetPointer(util::DEVICE),
        local_data_slice.vertex_associate_out[peer_gpu_rank_]
            .GetPointer(util::DEVICE),
        sizeof(VertexId) * queue_length * NUM_VERTEX_ASSOCIATES,
        cudaMemcpyDeviceToDevice, stream),
        "cudamemcpyPeer keys failed", __FILE__, __LINE__))
        return;

    if (NUM_VALUE__ASSOCIATES != 0)
    if (retval = util::GRError(cudaMemcpyAsync(
        peer_data_slice .value__associate_in [t][  gpu_rank_]
            .GetPointer(util::DEVICE),
        local_data_slice.value__associate_out[peer_gpu_rank_]
            .GetPointer(util::DEVICE),
        sizeof(Value) * queue_length * NUM_VALUE__ASSOCIATES,
        cudaMemcpyDeviceToDevice, stream),
        "cudamemcpyPeer keys failed", __FILE__, __LINE__))
        return;

#ifdef ENABLE_PERFORMANCE_PROFILING
    //enactor_stats -> iter_out_length.back().push_back(queue_length);
#endif
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
template <typename Problem>
void ShowDebugInfo(
    int           thread_num,
    int           peer_,
    FrontierAttribute<typename Problem::SizeT>
                 *frontier_attribute,
    EnactorStats<typename Problem::SizeT>
                 *enactor_stats,
    typename Problem::DataSlice
                 *data_slice,
    GraphSlice<typename Problem::VertexId, typename Problem::SizeT, typename Problem::Value>
                 *graph_slice,
    util::CtaWorkProgressLifetime<typename Problem::SizeT>
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
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %lld\n",
        thread_num, enactor_stats->iteration, peer_,
        data_slice->stages[peer_], check_name.c_str(),
        (long long)queue_length);
    fflush(stdout);
    //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), data_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,peer_, stream);
    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::MARK_PREDECESSOR)
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
template <typename DataSlice>
cudaError_t Check_Record(
    DataSlice *data_slice,
    int iteration,
    int peer_,
    int stage_to_check,
    int stage,
    bool &to_show)
{
    cudaError_t retval = cudaSuccess;
    to_show = true;
    if (!data_slice->events_set[iteration%4][peer_][stage_to_check])
    {
        to_show = false;
    } else {
        retval = cudaEventQuery(data_slice->events[iteration%4][peer_][stage_to_check]);
        if (retval == cudaErrorNotReady)
        {
            to_show= false;
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
