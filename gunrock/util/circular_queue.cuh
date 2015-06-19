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

#include <list>
#include <mutex>
#include <chrono>
#include <thread>
#include <string>
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

template <
    typename SizeT,
    typename VertexId,
    typename Value   = VertexId,
    bool AUTO_RESIZE = true>
struct CircularQueue
{
public:

    class CqEvent{
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
    std::string  name;      // name of the queue
    SizeT        capacity;  // capacity of the queue
    SizeT        size_occu; // occuplied size
    SizeT        size_soli; // size of the fixed part
    unsigned int allocated; // where the data is allocated, HOST or DEVICE
    Array1D<SizeT, VertexId>  array; // the main data
    Array1D<SizeT, VertexId>* vertex_associates; // VertexId type associate values
    Array1D<SizeT, Value   >* value__associates; // Value type associate values
    SizeT        num_vertex_associates;
    SizeT        num_value__associates;
    SizeT        head_a, head_b, tail_a, tail_b; // head and tail offsets
    std::list<CqEvent > events[2]; // 0 for in events, 1 for out events
    std::list<cudaEvent_t> empty_gpu_events;
    cudaEvent_t *gpu_events;
    SizeT        num_events;
    std::mutex   queue_mutex;
    int          wait_resize;

public:
    CircularQueue() :
        name      (""  ),
        capacity  (0   ),
        size_occu (0   ),
        size_soli (0   ),
        allocated (NONE),
        vertex_associates(NULL),
        value__associates(NULL),
        num_vertex_associates(0),
        num_value__associates(0),
        head_a    (0   ),
        head_b    (0   ),
        tail_a    (0   ),
        tail_b    (0   ),
        gpu_events(NULL),
        num_events(0   ),
        wait_resize(0  )
    {
        SetName("cq");
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
        SizeT        num_events = 10,
        SizeT        num_vertex_associates = 0,
        SizeT        num_value__associates = 0)
    {
        cudaError_t retval = cudaSuccess;

        this->capacity = capacity;
        if (retval = array.Allocate(capacity, target)) return retval;
        if (num_vertex_associates != 0)
        {
            this -> num_vertex_associates = num_vertex_associates;
            vertex_associates = new Array1D<SizeT, VertexId>[num_vertex_associates];
            for (int i=0; i<num_vertex_associates; i++)
            {
                vertex_associates[i].SetName(this->name + "_vertex[]");
                if (retval = vertex_associates[i].Allocate(capacity, target)) return retval;
            } 
        }
        if (num_value__associates != 0)
        {
            this -> num_value__associates = num_value__associates;
            value__associates = new Array1D<SizeT, Value>[num_value__associates];
            for (int i=0; i<num_value__associates; i++)
            {
                value_assocites[i].SetName(this->name + "_value[]");
                if (retval = value__associates[i].Allocate(capacity, target)) return retval;
            } 
        }
        allocated = target;
        head_a    = 0; head_b = 0;
        tail_a    = 0; tail_b = 0;
        size_occu = 0; size_soli = 0;
        wait_resize = 0;
       
        if (target == DEVICE)
        { 
            gpu_events = new cudaEvent_t[num_events];
            this -> num_events = num_events;
            for (SizeT i=0; i<num_events; i++)
            {
                if (retval = cudaEventCreateWithFlags(gpu_events + i, cudaEventDisableTiming)) return retval;
                empty_gpu_events.push_back(gpu_events[i]);
            }
        }

        events[0].clear();
        events[1].clear();

        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        if (allocated == DEVICE)
        {
            for (SizeT i=0; i<num_events; i++)
            {
                if (retval = cudaEventDestroy(gpu_events[i])) return retval;
            }
            delete[] gpu_events; gpu_events = NULL;
            empty_gpu_events.clear();
        }
        events[0].clear();
        events[1].clear();

        if (retval = array.Release()) return retval;
        if (vertex_associates != NULL)
        {
            for (int i=0; i<num_vertex_associates; i++)
            {
                if (retval = vertex_associates[i].Release()) return retval;
            }
            delete[] vertex_associates; vertex_associates = NULL;
        }
        if (value__associates != NULL)
        {
            for (int i=0; i<num_value__associates; i++)
            {
                if (retval = value__associates[i].Release()) return retval;
            }
            delete[] value__associates; value__associates = NULL;
        }

        return retval;
    }

    void GetSize(SizeT &size_occu, SizeT &size_soli)
    {
        size_soli = this->size_soli;
        size_occu = this->size_occu;
    }

    SizeT GetCapacity()
    {
        return capacity;
    }

    cudaError_t Combined_Return(cudaError_t retval, bool in_critical)
    {
        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    void ShowDebugInfo(
        std::string function_name,
        int         direction,
        SizeT       start,
        SizeT       end,
        SizeT       dsize,
        Value       value = -1)
    {
        printf("%s\t %s\t %d\t %d\t ~ %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\n",
            function_name.c_str(), direction == 0? "->" : "<-",
            value, start, end, dsize, size_occu, size_soli,
            head_a, head_b, tail_a, tail_b);
        //fflush(stdout);
    }

    cudaError_t Push(
        SizeT         length, 
        VertexId     *array, 
        cudaStream_t  stream = 0,
        SizeT         num_vertex_associates = 0, 
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lens   [2] = {0, 0};
        SizeT sum        = 0;
        if (retval = AddSize(length, offsets, lens)) return retval;

        for (int i=0; i<2; i++)
        {
            if (lens[i] == 0) continue;
            ShowDebugInfo("Push", 0, offsets[i], offsets[i]+lens[i], lens[i], ptr);
            if (retval = this->array.Move_In(
                allocated, allocated, array, lens[i], sum, offsets[i], stream)) return retval;
            for (SizeT j=0; j<num_vertex_associates; j++)
            {
                if (retval = this->vertex_associates[j].Move_In(
                    allocated, allocated, vertex_associates[j], sum, offsets[i], stream))
                    return retval;
            }
            for (SizeT j=0; j<num_value__associates; j++)
            {
                if (retval = this->value__associates[j].Move_In(
                    allocated, allocated, value__associates[j], sum, offsets[i], stream))
                    return retval;
            }

            // in_event finish
            if (allocated == HOST) EventFinish(0, offsets[i], lens[i]);
            else if (allocated == DEVICE)
                EventSet(0, offsets[i], lens[i], stream);
            sum += lens[i];
        } 
        return retval;
    }

    cudaError_t Pop(
        SizeT         min_length, 
        SizeT         max_length, 
        VertexId     *ptr, 
        SizeT        &len, 
        cudaStream_t  stream = 0,
        SizeT         num_vertex_associates = 0,
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lens   [2] = {0, 0};
        SizeT sum        = 0;
        
        if (retval = ReduceSize(min_length, max_length, offsets, lens)) return retval;

        for (int i=0; i<2; i++)
        {
            if (lens[i] == 0) continue;
            ShowDebugInfo("Pop", 1, offsets[i], offsets[i] + lens[i], lens[i]);
            if (retval = this->array.Move_Out(
                allocated, allocated, array, lens[i], sum, offsets[i], stream)) return retval;
            for (SizeT j=0; j<num_vertex_associates; j++)
            {
                if (retval = this->vertex_associates[j].Move_Out(
                    allocated, allocated, vertex_associates[j], sum, offsets[i], stream))
                    return retval;
            }
            for (SizeT j=0; j<num_value__associates; j++)
            {
                if (retval = this->value__associates[j].Move_Out(
                    allocated, allocated, value__associates[j], sum, offsets[i], stream))
                    return retval;
            }
            if (allocated == HOST) EventFinish(1, offsets[i], lens[i]);
            else if (allocated == DEVICE)
                EventSet(1, offsets[i], lens[i], stream);
            sum += lens[i];
        }
        len = sum;

        return retval; 
    }

    cudaError_t AddSize(SizeT length, SizeT *offsets, SizeT* lens, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;

        // in critical sectioin
        while (wait_resize != 0)
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        if (!in_critical) queue_mutex.lock();
        bool past_wait = false;
        while (!past_wait)
        {
            if (wait_resize == 0) {past_wait = true; break;}
            else {
                queue_mutex.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                queue_mutex.lock();
            }
        }
       
        if (allocated == DEVICE)
        {
            if (retval = EventCheck(1, true)) 
                return Combined_Return(retval, in_critical);
        }
         
        if (length + size_occu > capacity) 
        { // queue full
            if (AUTO_RESIZE)
            {
                if (retval = EnsureCapacity(length + size_occu, true)) 
                    return Combined_Return(retval, in_critical);
            } else {
                if (length > capacity)
                { // too large for the queue
                    retval = util::GRError(cudaErrorLaunchOutOfResources, 
                        (name + " oversize ").c_str(), __FILE__, __LINE__);
                    return Combined_Return(retval, in_critical);
                } else {
                    queue_mutex.unlock();
                    bool got_space = false;
                    while (!got_space)
                    {
                        if (length + size_occu < capacity)
                        {
                            queue_mutex.lock();
                            if (length + size_occu < capacity)
                            {
                                got_space = true;
                            } else {
                                queue_mutex.unlock();
                            }
                        }
                        if (!got_space) {
                            std::this_thread::sleep_for(std::chrono::microseconds(10));
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
        size_occu += length;

        ShowDebugInfo("AddSize", 0, offsets[0], head_a, length);
        return Combined_Return(retval, in_critical);
    }

    cudaError_t ReduceSize(SizeT min_length, SizeT max_length, SizeT *offsets, SizeT* lens, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        SizeT length = 0;
        // in critial section
        while (wait_resize != 0)
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        if (!in_critical) queue_mutex.lock();

        if (allocated == DEVICE)
        {
            if (retval = EventCheck(0, true))
                return Combined_Return(retval, in_critical);
        }
        
        if (size_soli < min_length)
        { // too small
            queue_mutex.unlock();
            bool got_content = false;
            while (!got_content)
            {
                if (size_soli >= min_length)
                {
                    queue_mutex.lock();
                    if (size_soli >= min_length)
                    {
                        got_content = true;
                    } else {
                        queue_mutex.unlock();
                    }
                }
                if (!got_content) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

        length = size_soli < max_length ? size_soli : max_length;
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
        size_soli -= length;

        ShowDebugInfo("RedSize", 1, offsets[0], tail_a, length);
        return Combined_Return(retval, in_critical);
    }

    cudaError_t EnsureCapacity(SizeT capacity_, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;

        if (!in_critical) queue_mutex.lock();
        printf("capacity -> %d\n", capacity_);
        if (capacity_ > capacity)
        {
            wait_resize = 1;
            while ((!events[0].empty()) || (!events[1].empty()))
            {
                queue_mutex.unlock(); 
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                queue_mutex.lock();
                for (int i=0; i<2; i++)
                if (retval = EventCheck(i, true))
                {
                    queue_mutex.unlock();
                    return retval;
                }
            }

            if (retval = array.EnsureSize(capacity_, true)) 
                return Combined_Return(retval, in_critical);
            for (SizeT i=0; i<num_vertex_associates; i++)
            {
                if (retval = vertex_associates[i].EnsureSize(capacity_, true))
                    return Combined_Return(retval, in_critical);
            }
            for (SizeT i=0; i<num_value__associates; i++)
            {
                if (retval = value__associates[i].EnsureSize(capacity_, true))
                    return Combined_Return(retval, in_critical);
            }

            if (tail_a + size_occu > capacity)
            {
                if (tail_a + size_occu > capacity_)
                { // Content cross original and new end point
                    SizeT first_length = capacity_ - capacity;
                    Array1D<SizeT, VertexId> temp_vertex;
                    Array1D<SizeT, Value   > temp_value ;

                    if (retval = temp_vertex.Allocate(head_a - first_length, allocated))
                        return Combined_Return(retval, in_critical);
                    if (num_value__associates != 0)
                    if (retval = temp_value .Allocate(head_a - first_length, allocated))
                        return Combined_Return(retval, in_critical);

                    if (retval = array.Move_Out(allocated, allocated,
                        array       .GetPointer(allocated), first_length, 0, capacity))
                        return Combined_Return(retval, in_critical);
                    if (retval = array.Move_Out(allocated, allocated,
                        temp_vertex .GetPointer(allocated), head_a - first_length, first_length, 0))
                        return Combined_Return(retval, in_critical);
                    if (retval = array.Move_In (allocated, allocated,
                        temp_vertex .GetPointer(allocated), head_a - first_length, 0, 0))
                        return Combined_Return(retval, in_critical);

                    for (SizeT i=0; i<num_vertex_asspciates; i++)
                    {
                        if (retval = vertex_associates[i].Move_Out(allocated, allocated,
                            vertex_assocaites[i].GetPointer(allocated), first_length, 0, capacity))
                            return Combined_Return(retval, in_critical);
                        if (retval = vertex_associates[i].Move_Out(allocated, allocated,
                            temp_vertex .GetPointer(allocated), head_a - first_length, first_length, 0))
                            return Combined_Return(retval, in_critical);
                        if (retval = vertex_associates[i].Move_In (allocated, allocated,
                            temp_vertex .GetPointer(allocated), head_a - first_length, 0, 0))
                            return Combined_Return(retval, in_critical);
                    }
                    for (SizeT i=0; i<num_value__asspciates; i++)
                    {
                        if (retval = value__associates[i].Move_Out(allocated, allocated,
                            value__assocaites[i].GetPointer(allocated), first_length, 0, capacity))
                            return Combined_Return(retval, in_critical);
                        if (retval = value__associates[i].Move_Out(allocated, allocated,
                            temp_value  .GetPointer(allocated), head_a - first_length, first_length, 0))
                            return Combined_Return(retval, in_critical);
                        if (retval = value__associates[i].Move_In (allocated, allocated,
                            temp_value .GetPointer(allocated), head_a - first_length, 0, 0))
                            return Combined_Return(retval, in_critical);
                    }
                   
                    if (retval = temp_vertex.Release(allocated))
                        return Combined_Return(retval, in_critical);
                    if (num_value__associates != 0)
                    if (retval = temp_value .Release(allocated))
                        return Combined_Return(retval, in_critical);
                } else { // Content cross original end point, but not new end point
                    if (retval = array.Move_Out(allocated, allocated, 
                        array.GetPointer(allocated), head_a, 0, capacity)) 
                        return Combined_Return(retval, in_critical);
                    for (SizeT i=0; i<num_vertex_associates; i++)
                    {
                        if (retval = vertex_associates[i].Move_Out(allocated, allocated,
                            vertex_associates[i].GetPointer(allocated), head_a, 0, capacity))
                            return Combined_Return(retval, in_critical);
                    }
                    for (SizeT i=0; i<num_value__associates; i++)
                    {
                        if (retval = value__associates[i].Move_Out(allocated, allocated,
                            value__associates[i].GetPointer(allocated), head_a, 0, capacity))
                            return Combined_Return(retval, in_critical);
                    }
                }
            }

            capacity = capacity_;
            head_a = (tail_a + size_occu) % capacity;
            head_b = head_a;
            temp_array.Release();
            printf("EnsureCapacity: capacity -> %d, head_a -> %d\n", capacity, head_a);
            //fflush(stdout);
            wait_resize = 0;
        }
        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    void EventStart( int direction, SizeT offset, SizeT length, bool in_critical = false)
    {
        if (!in_critical) queue_mutex.lock();
        printf("Event %d,%d,%d starts\n", direction, offset, length);//fflush(stdout);
        events[direction].push_back(CqEvent(offset, length));
        if (!in_critical) queue_mutex.unlock();
    }

    cudaError_t EventSet(   int direction, SizeT offset, SizeT length, cudaStream_t stream = 0, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        if (!in_critical) queue_mutex.lock();
        
        if (empty_gpu_events.empty())
        {
            retval = util::GRError(cudaErrorLaunchOutOfResources,
                (name + " gpu_events oversize ").c_str(), __FILE__, __LINE__);
            if (!in_critical) queue_mutex.unlock();
            return retval;    
        }
        cudaEvent_t event = empty_gpu_events.front();
        empty_gpu_events.pop_front();
        if (retval = cudaEventRecord(event, stream))
        {
            if (!in_critical) queue_mutex.unlock();
            return retval;
        }

        typename std::list<CqEvent>::iterator it = events[direction].begin();
        for (it  = events[direction].begin(); 
             it != events[direction].end(); it ++)
        {
            if ((offset == (*it).offset) && (length == (*it).length)) // matched event
            {
                printf("Event %d,%d,%d sets\n", direction, offset, length);//fflush(stdout);
                (*it).event = event;
                (*it).status = 1;
                break;
            }
        }
        EventCheck(direction, true);
        if (!in_critical) queue_mutex.unlock();
        return retval;
    }

    void EventFinish(int direction, SizeT offset, SizeT length, bool in_critical = false)
    {
        if (!in_critical) queue_mutex.lock();
        typename std::list<CqEvent>::iterator it = events[direction].begin();
        for (it  = events[direction].begin(); 
             it != events[direction].end(); it ++)
        {
            if ((offset == (*it).offset) && (length == (*it).length)) // matched event
            {
                printf("Event %d,%d,%d finishes\n", direction, offset, length);//fflush(stdout);
                (*it).status = 2;
                break;
            }
        }
        SizeCheck(direction, true);
        ShowDebugInfo("EventF", direction, offset, -1, length);
        if (!in_critical) queue_mutex.unlock();
    }

    cudaError_t EventCheck(int direction, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        if (!in_critical) queue_mutex.lock();

        typename std::list<CqEvent>::iterator it = events[direction].begin();
        for (it  = events[direction].begin();
             it != events[direction].end(); it++)
        {
            if ((*it).status == 1)
            {
                retval = cudaEventQuery((*it).event);
                if (retval == cudaSuccess)
                {
                    (*it).status = 2;
                    printf("Event %d,%d,%d finishes\n", direction, (*it).offset, (*it).length);
                    empty_gpu_events.push_back((*it).event);
                } else if (retval != cudaErrorNotReady) {
                    if (!in_critical) queue_mutex.unlock();
                    return retval;
                }
            }
        }
        SizeCheck(direction, true);
        ShowDebugInfo("EventC", direction, -1, -1, -1);
        if (!in_critical) queue_mutex.unlock();
        return retval; 
    }

    void SizeCheck(int direction, bool in_critical = false)
    {
        if (!in_critical) queue_mutex.lock();
        typename std::list<CqEvent>::iterator it = events[direction].begin();
       
        while (!events[direction].empty())
        {
            it = events[direction].begin();
            //printf("Event %d, %d, %d, status = %d\n", direction, (*it).offset, (*it).length, (*it).status);fflush(stdout);
            if ((*it).status == 2) // finished event
            {
                if (direction == 0)
                { // in event
                    if ((*it).offset == head_b)
                    {
                        head_b += (*it).length;
                        if (head_b >= capacity) head_b -= capacity;
                        (*it).status = 3;
                        size_soli += (*it).length;
                    } 
                } else { // out event
                    if ((*it).offset == tail_b)
                    {
                        tail_b += (*it).length;
                        if (tail_b >= capacity) tail_b -= capacity;
                        (*it).status = 3;
                        size_occu -= (*it).length;
                    }
                }
                events[direction].pop_front();
            } else {
                break;
            }
        }

        if (!in_critical) queue_mutex.unlock();
    }
}; // end of struct CircularQueue

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

