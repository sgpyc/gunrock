// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * pinned_memory.cuh
 *
 * @brief pinned memory operation
 */

#pragma once

#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace util{

template <typename T>
cudaError_t FreePinned(
    T*    &pointer,
    bool  pinned = true)
{
    cudaError_t retval = cudaSuccess;

    if (pointer != (T*)NULL)
    {
        if (pinned)
        {
            if (retval = GRError(
                cudaFreeHost(pointer),
                "cudaFreeHost failed", __FILE__, __LINE__))
                return retval;
        } else {
            free(pointer);
        }
    }
    pointer = (T*)NULL;

    return retval;
}

template <typename T, typename SizeT>
cudaError_t MallocPinned(
    T*    &pointer,
    SizeT length,
    bool  pinned = true)
{
    cudaError_t retval = cudaSuccess;

    if (pointer != (T*)NULL)
    {
        if (retval = FreePinned(pointer, pinned))
            return retval;
    }

    if (pinned)
    {
        if (retval = GRError(
            cudaHostAlloc((void**)&pointer, sizeof(T) * length,
                cudaHostAllocMapped),
            "cudaHostAlloc failed", __FILE__, __LINE__))
            return retval;
    } else {
        pointer = (T*)malloc(sizeof(T) * length);
        if (pointer == (T*)NULL)
        {
            retval = GRError(cudaErrorMemoryAllocation,
                "malloc failed", __FILE__, __LINE__);
            return retval;
        }
    }

    return retval;
}

template <typename T1, typename T2, typename SizeT>
cudaError_t MemcpyPinned(
    T1*      &target,
    T2*      source,
    SizeT   length,
    bool    pinned = true)
{
    cudaError_t retval = cudaSuccess;

    if (source == NULL)
    {
        if (retval = FreePinned(target, pinned))
            return retval;
        return retval;
    }

    if (target == NULL)
    {
        if (retval = MallocPinned(target, length, pinned))
            return retval;
    }
    memcpy(target, source, sizeof(T1) * length);

    return retval;
}

} // namespace util
} // namespace gunrock
