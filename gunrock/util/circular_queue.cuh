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

#include <mutex>
#include <string>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

template <
    typename SizeT,
    typename Value,
    bool AUTO_RESIZE = true>
struct CircularQueue
{

    struct CqEvent{
    public:
        int   status;
        SizeT offset, length;
        cudaEvent_t event;

        CqEvent(
            SizeT offset_,
            SizeT length_) :
            status(0      ),
            offset(offset_),
            length(length_)
        {
        }
    }; // end of CqEvent

private:
    std::string  name;
    SizeT        capacity;
    SizeT        size;
    unsigned int allocated;
    Array1D<SizeT, Value> array;
    SizeT        head_a, head_b, tail_a, tail_b;
    list<CqEvent> events[2]; // 0 for in events, 1 for out events
    cudaEvent_t  *gpu_events;
    SizeT        num_events;
    mutex        queue_mutex;
    int          waiting_resize;

    CircularQueue() :
        name      (""  ),
        capacity  (0   ),
        size      (0   ),
        allocated (NONE),
        head_a    (0   ),
        head_b    (0   ),
        tail_a    (0   ),
        tail_b    (0   ),
        gpu_events(NULL),
        num_events(0   ),
        wait_resize(0  )
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
        allocated = target;
        head_a = 0; head_b = 0;
        tail_a = 0; tail_b = 0;
        size   = 0; wait_resize = 0;
        
        gpu_events = new cudaEvent_t[num_events];
        this -> num_events = num_events;
        for (SizeT i=0; i<num_events; i++)
        {
            if (retval = cudaEventCreateWithFlags(gpu_events + i, cudaEventDisableTiming)) return retval;
        }

        list[0].clear();
        list[1].clear();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        for (SizeT i=0; i<num_events; i++)
        {
            if (retval = cudaEventDestroy(gpu_events[i])) return retval;
        }
        if (retval = array.Release()) return retval;
        
        delete[] gpu_events; gpu_events = NULL;
        list[0].clear();
        list[1].clear();
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
            memcpy(array + offsets[0], ptr, sizeof(Value) * lens[0]);
            // in_event finish
            EventFinish(0, offsets[0], lens[0]);
            if (lens[1] != 0)
            {
                memcpy(array + offsets[1], ptr + lens[0], sizeof(Value) * lens[1]);
                // in_event finish
                EventFinish(0,offsets[1], lens[1]);
            }
               
        } else if (allocated == DEVICE)
        {
            /*MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                array.GetPointer(util::DEVICE) + offsets[0],
                ptr, lens[0]);
            // Add in_event
            if (lens[1] != 0)
            {
                MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                    array.GetPointer(util::DEVICE),
                    ptr + lens[0], lens[1]);
                // Add in_event
            }*/
        } 
        return retval;
    }

    cudaError_t Pop(SizeT min_length, SizeT max_length, Value* ptr, cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lens   [2] = {0, 0};
        
        if (retval = ReduceSize(min_length, max_length, offset, lens)) return retval;

        if (allocated == HOST)
        {
            memcpy(ptr, array + offsets[0], sizeof(Value) * lens[0]);
            // out_event finish
            EventFinish(1, offsets[0], lens[0]);
            if (lens[1] != 0)
            {
                memcpy(ptr + lens[0], array + offsets[1], sizeof(Value) * lens[1]);
                // out_event finish
                EventFinish(1, offsets[1], lens[1]);
            }
        } else if (allocated == DEVICE)
        {
        } 
    }

    cudaError_t AddSize(SizeT length, SizeT *offsets, SizeT* lens, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;

        // in critical sectioin
        //lock_guard<mutex> lock(queue_mutex);
        while (wait_resize != 0)
            this_thread::sleep_for(chrono::microseconds(10));
        if (!in_critical) queue_mutex.lock();        

        if (length + size > capacity) 
        { // queue full
            if (AUTO_RESIZE)
            {
                if (retval = EnsureCapacity(length + size, true)) 
                {
                    if (!in_critical) queue_mutex.unlock();
                    return retval;
                }
            } else {
                if (length > capacity)
                { // too large for the queue
                    retval = util::GRError(cudaErrorLaunchOutOfResource, 
                        (name + " oversize ").c_str(), __FILE__, __LINE__);
                    if (!in_critical) queue_mutex.unlock();
                    return retval;
                } else {
                    queue_mutex.unlock();
                    bool got_space = false;
                    while (!got_space)
                    {
                        if (length + size < capacity)
                        {
                            queue_mutex.lock();
                            if (length + size < capacity)
                            {
                                got_space = true;
                            } else {
                                queue_mutex.unlock();
                            }
                        }
                        if (!got_space) {
                            this_thread::sleep_for(chrono::microseconds(10));
                        }
                    }
                }
            }
        }

        if (head_a + length > capacity)
        { // splict
            offsets[0] = head_a;
            lens   [0] = capacity - head_a;
            EventStart(0, offsets[0], lens[0], true);
            offsets[1] = 0;
            lens   [1] = length - lens[0];
            EventStart(0, offsets[1], lens[1], true);
            head_a     = lens[1];
        } else { // no splict
            offsets[0] = head_a;
            lens   [0] = length;
            EventStart(0, offsets[0], lens[0], true);
            offsets[1] = 0;
            lens   [1] = 0;
            head_a += length;
            if (head_a == capacity) head_a = 0;
        }
        size += length;

        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    cudaError_t ReduceSize(SizeT min_length, SizeT max_length, SizeT *offsets, SizeT* lens, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        SizeT length = 0;
        // in critial section
        //lock_guard<mutex> lock(queue_mutex);
        while (wait_resize != 0)
            this_thread::sleep_for(chrono::microseconds(10));
        if (!in_critical) queue_mutex.lock();
        
        if (size < min_length)
        { // too small
            queue_mutex.unlock();
            bool got_content = false;
            while (!got_content)
            {
                if (size >= min_length)
                {
                    queue_mutex.lock();
                    if (size >= min_length)
                    {
                        got_content = true;
                    } else {
                        queue_mutex.unlock();
                    }
                }
                if (!got_content) {
                    this_thread::sleep_for(chrono::microseconds(10));
                }
            }
        }

        length = size > max_size ? size : max_size;
        if (tail_a + length > capacity)
        { // splict
            offsets[0] = tail_a;
            lens   [0] = capacity - tail_a;
            EventStart(1, offsets[0], lens[0], true);
            offsets[1] = 0;
            lens   [1] = length - lens[0];
            EventStart(1, offsets[1], lens[1], true);
            tail_a     = lens[1];
        } else {
            offsets[0] = tail_a;
            lens   [0] = length;
            EventStart(1, offsets[0], lens[0], true);
            offsets[1] = 0;
            lens   [1] = 0;
            tail_a += length;
            if (tail_a == capacity) tail_a = 0;
        }
        size -= length;

        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    cudaError_t EnsureCapacity(SizeT capacity_, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;

        if (!in_critical) queue_mutex.lock();
        if (capacity_ > capacity)
        {
            wait_resize = 1;
            while ((!events[0].empty()) || (!events[1].empty()))
            {
                this_thread::sleep_for(chrono::microseconds(10));
            }

            Array1D<SizeT, Value> temp_array;
            if (tail_a + size > capacity)
            { // content corss end point
                if (retval = temp_array.Allocate(head_a, allocated))
                {
                    if (!in_critical) queue_mutex.unlock();
                    return retval;
                }
                if (allocated == HOST) 
                {
                    memcpy(temp_array, array, sizeof(Value) * head_a);
                } else {
                }
            }

            if (retval = array.EnsureSize(capacity_, true))
            {
                if (!in_critical) queue_mutex.unlock();
                return retval;
            }

            if (tail_a + size > capacity)
            {
                if (tail_a + size > capacity_)
                { // splict new array
                    if (allocated == HOST)
                    {
                        memcpy(array + capacity, temp_array, sizeof(Value) * (capacity_-capacity));
                        memcpy(array, temp_array + (capacity_ - capacity), sizeof(Value) * (head_a - (capacity_ - capacity)));
                    } else (allocated == DEVICE)
                    {
                    } 
                } else {
                    if (allocated == HOST)
                    {
                        memcpy(array + capacity, temp_array, sizeof(Value) * head_a);
                    } else (allocated == DEVICE)
                    {
                    }
                }
            }

            capacity = capacity_;
            head_a = (tail_a + size) % capacity;
            wait_resize = 0;
        }
        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    void EventStart( int direction, SizeT offset, SizeT length, bool in_critical = false)
    {
        if (!in_critical) queue_mutex.lock();
        events[direction].push_back(CqEvent(offset, length));
        if (!in_critical) queue_mutex.unlock();
    }

    void EventSet(   int direction, SizeT offset, SizeT length)
    {
    }

    void EventFinish(int dierction, SizeT offset, SizeT length, bool in_critical = false)
    {
        if (!in_critical) queue_mutex.lock();
        list<CqEvent>::iterator it;
        for (it  = events[direction].begin(); 
             it != events[direction].end(); it ++)
        {
            if ((offset == (*it).offset) && (length == (*it).length)) // matched event
            {
                (*it).status = 2;
                break;
            }
        }

        while (!events[direction].empty())
        {
            it = events[direction].front();
            if ((*it).status == 2) // finished event
            {
                if (direction == 0)
                { // in event
                    if (offset == head_b)
                    {
                        head_b += (*it).length;
                        if (head_b >= capacity) head_b -= capacity;
                        (*it).status = 3;
                    } 
                } else { // out event
                    if (offset == tail_b)
                    {
                        tail_b += (*it).length;
                        if (tail_b >= capacity) tail_b -= capacity;
                        (*it).status = 3;
                        size -= (*it).length;
                    }
                }
                events[direction].pop_front();
            } else {
                break;
            }
        }
        if (!in_critical) queue_mutex.unlock();
    }
} // end of struct CircularQueue

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

