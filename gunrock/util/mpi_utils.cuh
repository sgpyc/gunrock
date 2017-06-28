// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mpi_utils.cuh
 *
 * @brief warpers to transmit large trunk of data via MPI
 */

#pragma once

#include <vector>
#include <mpi.h>

#ifndef GR_MPI_CHUNK_BYTES
    // 32MB chunk size
    #define GR_MPI_CHUNK_BYTES 1LL<<25
#endif

#define MPI_DEBUG false

namespace gunrock {
namespace util {

int Get_Recv_Tag(
    int target_gpu_rank_local,
    int num_total_gpus,
    int source_gpu_rank,
    int sub_tag)
{
    return (target_gpu_rank_local * num_total_gpus + source_gpu_rank) * 32 + sub_tag;
}

int Get_Send_Tag(
    int target_gpu_rank_remote,
    int num_total_gpus,
    int source_gpu_rank,
    int sub_tag)
{
    return (target_gpu_rank_remote * num_total_gpus + source_gpu_rank) * 32 + sub_tag;
}

template <typename T, typename SizeT>
cudaError_t Mpi_Isend_Bulk(
    T     *buf,
    SizeT  length,
    int    target,
    int    tag,
    MPI_Comm comm,
    std::vector<MPI_Request> &requests)
{
    cudaError_t retval = cudaSuccess;
    size_t current_offset = 0;
    size_t total_bytes = length * sizeof(T);

    if (length <= 0) return retval;
    if (MPI_DEBUG)
        PrintMsg("Sending \t" + std::to_string(length) +
            "\t x \t" + std::to_string(sizeof(T)) + 
            "\t bytes to rank \t" + std::to_string(target) + 
            "\t , tag \t" + std::to_string(tag));

    while (current_offset < total_bytes)
    {
        MPI_Request request;
        int current_count =
            (total_bytes - current_offset > GR_MPI_CHUNK_BYTES) ?
            GR_MPI_CHUNK_BYTES : (total_bytes - current_offset);
        int mpi_retval = MPI_Isend(
            ((char*)buf) + current_offset,
            current_count, MPI_BYTE, target,
            tag, comm, &request);
        requests.push_back(request);
        if (mpi_retval != MPI_SUCCESS)
        {
            retval = GRError("Mpi_Isend error " + std::to_string(mpi_retval),
                __FILE__, __LINE__);
            return retval;
        }
        current_offset += GR_MPI_CHUNK_BYTES;
    }
    return retval;
}

template <typename T, typename SizeT>
cudaError_t Mpi_Irecv_Bulk(
    T     *buf,
    SizeT  length,
    int    source,
    int    tag,
    MPI_Comm comm,
    std::vector<MPI_Request> &requests)
{
    cudaError_t retval = cudaSuccess;
    size_t current_offset = 0;
    size_t total_bytes = length * sizeof(T);

    if (length <= 0) return retval;
    if (MPI_DEBUG)
        PrintMsg("Receiving \t" + std::to_string(length) +
            "\t x \t" + std::to_string(sizeof(T)) + 
            "\t bytes from rank \t" + std::to_string(source) + 
            "\t , tag \t" + std::to_string(tag));

    while (current_offset < total_bytes)
    {
        MPI_Request request;
        int current_count =
            (total_bytes - current_offset > GR_MPI_CHUNK_BYTES) ?
            GR_MPI_CHUNK_BYTES : (total_bytes - current_offset);
        int mpi_retval = MPI_Irecv(
            ((char*)buf) + current_offset,
            current_count, MPI_BYTE, source,
            tag, comm, &request);
        requests.push_back(request);
        if (mpi_retval != MPI_SUCCESS)
        {
            retval = GRError("Mpi_Irecv error " + std::to_string(mpi_retval),
                __FILE__, __LINE__);
            return retval;
        }
        current_offset += GR_MPI_CHUNK_BYTES;
    }
    return retval;
}

cudaError_t Mpi_Waitall(
    std::vector<MPI_Request> &requests,
    std::vector<MPI_Status > &statuses)
{
    cudaError_t retval = cudaSuccess;
    int num_requests = requests.size();
    statuses.resize(num_requests);
    int mpi_retval = MPI_Waitall(num_requests,
        requests.data(), statuses.data());
    if (mpi_retval != MPI_SUCCESS)
    {
        retval = GRError("Mpi_Waitall error " + std::to_string(mpi_retval),
            __FILE__, __LINE__);
        return retval;
    }
    return retval;
}

cudaError_t Mpi_Waitall(
    std::vector<MPI_Request> &requests)
{
    cudaError_t retval = cudaSuccess;
    int num_requests = requests.size();
    int mpi_retval = MPI_Waitall(num_requests,
        requests.data(), MPI_STATUSES_IGNORE);
    if (mpi_retval != MPI_SUCCESS)
    {
        retval = GRError("Mpi_Waitall error " + std::to_string(mpi_retval),
            __FILE__, __LINE__);
        return retval;
    }
    return retval;
}

cudaError_t Mpi_Test(
    std::vector<MPI_Request> &requests)
{
    cudaError_t retval = cudaSuccess;
    for (auto it = requests.begin(); it!= requests.end(); )
    {
        auto &request = *it;
        int flag;
        int mpi_retval = MPI_Test(&request, &flag, MPI_STATUSES_IGNORE);
        if (mpi_retval != MPI_SUCCESS)
        {
            retval = GRError("Mpi_Test error " + std::to_string(mpi_retval),
                __FILE__, __LINE__);
            return retval;
        }

        if (flag)
        {
            it = requests.erase(it);
        } else it++;
    }
    return retval;
}

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
