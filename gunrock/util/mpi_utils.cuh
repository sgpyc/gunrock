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

namespace gunrock {
namespace util {

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

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
