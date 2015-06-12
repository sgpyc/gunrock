// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * circular_queue.cuh
 *
 * @brief asynchronous circular queue
 */

#pragma once

#include <string>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

template <
    typename SizeT,
    typename Value>
struct CircularQueue
{

    struct CqEvent{
    public:
        int   direction, status;
        SizeT offset, length;
        cudaEvent_t event;
    }; // end of CqEvent

private:
    std::string  name;
    SizeT        capacity;
    SizeT        size;
    unsigned int allocated;
    Array1D<SizeT, Value> array;
    SizeT        head_a, head_b, tail_a, tail_b;
    list<CqEvent> events[2]; // 0 for in events, 1 for out events

    CircularQueue() :
        name     (""  ),
        capacity (0   ),
        size     (0   ),
        allocated(NONE),
        head_a   (0   ),
        head_b   (0   ),
        tail_a   (0   ),
        tail_b   (0   )
    {
    }

    ~CircularQueue()
    {
        Release();
    }

    void SetName(std::string name)
    {
        this->name = name;
        array.SetName(name+"_array");
    }

    cudaError_t Init(
        SizeT        capacity, 
        unsigned int target   = HOST,
        SizeT        num_events = 10)
    {
        cudaError_t retval = cudaSuccess;

        this->capacity = capacity;
        if (retval = array.Allocate(size, target)) return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = array.Release()) return retval;

    }

    SizeT GetSize()
    {
        return size;
    }

    SizeT GetCapacity()
    {
        return capacity;
    }

    cudaError_t Push(SizeT length, Value* ptr, cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lens   [2] = {0, 0};

        if (retval = AddSize(length, offsets, lens)) return retval;

        if (allocated == HOST)
        {
            // Add in_event
            memcpy(array + offsets[0], ptr, sizeof(Value) * lens[0]);
            // in_event finish
            if (lens[1] != 0)
            {
                // Add in_event
                memcpy(array, ptr + lens[0], sizeof(Value) * lens[1]);
                // in_event finish
            }
               
        } else if (allocated == DEVICE)
        {
            MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                array.GetPointer(util::DEVICE) + offsets[0],
                ptr, lens[0]);
            // Add in_event
            if (lens[1] != 0)
            {
                MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                    array.GetPointer(util::DEVICE),
                    ptr + lens[0], lens[1]);
                // Add in_event
            }
        } 
        return retval;
    }

    cudaError_t Pop(SizeT min_length, SizeT max_length, Value* ptr, cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
          
    }

    cudaError_t AddSize(SizeT length, SizeT *offsets, SizeT* lens)
    {
        // in critical section
        if (length + size > capacity) EnsureCapacity(length + size, true);

        if (head_a + length > capacity)
        { // splict
            offsets[0] = head_a;
            lens   [0] = capacity - head_a;
            offsets[1] = 0;
            lens   [1] = length - lens[0];
            head_a     = lens[1];
        } else { // no splict
            offsets[0] = head_a;
            lens   [0] = length;
            offsets[1] = 0;
            lens   [1] = 0;
            head_a += length;
            if (head_a == capacity) head_a = 0;
        }
        size += length;

    }

    cudaError_t EnsureCapacity(SizeT capacity_, bool in_critical = false)
    {
        if (capacity_ > capacity)
        {
        }
    }

    void EventStart(int direction, SizeT offset, SizeT length)
    {
    }

    void EventSet(int direction, SizeT offset, SizeT length)
    {
    }
} // end of struct CircularQueue

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

