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
#include <gunrock/util/test_utils.h>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace util {

#define CQ_DEBUG true

static const char* const event_strs[] = {"In", "Out", "Block", "All", "None"};

template <typename SizeT>
class CqEvent{
public:
    enum Status {
        New,
        Init,
        Assigned,
        Running,
        Finished,
        Cleared
    };

    enum Type {
        In    = 0,
        Out   = 1,
        Block = 2,
        All   = 3,
        None  = 4
    };

    Status      status;
    SizeT       offset;
    SizeT       length;
    Type        type;
    cudaEvent_t event ;
    bool        externel;
    bool        use_temp;

    CqEvent(
        Type  type_,
        SizeT offset_,
        SizeT length_) :
        status(New    ),
        offset(offset_),
        length(length_),
        externel(false),
        use_temp(false),
        type    (type_)
    {
    }

    CqEvent& operator=(const CqEvent& src)
    {
        status = src.status;
        offset = src.offset;
        length = src.length;
        event  = src.event ;
        type   = src.type  ;
        externel = src.externel;
        use_temp = src.use_temp;
        return *this;
    }

    static const char* Type2Str(Type type)
    {
        return event_strs[type];
    }
}; // end of CqEvent

template <
    typename VertexId,
    typename SizeT,
    typename Value   = VertexId,
    bool AUTO_RESIZE = true>
struct CircularQueue
{
    typedef CqEvent<SizeT> Event;
    typedef typename CqEvent<SizeT>::Type EventType;

private:
    std::string  name;      // name of the queue
    int          gpu_idx ;  // gpu index
    int          gpu_num ;
    int          input_count;
    int          output_count;
    int          input_set_flip;
    int          input_get_flip;
    int          output_set_flip;
    int          output_get_flip;
    int          target_input_count[2];
    SizeT        target_output_pos [2]; 
    SizeT        iteration_length  [2];
    SizeT        capacity;  // capacity of the queue
    SizeT        size_occu; // occuplied size
    SizeT        size_soli; // size of the fixed part
    unsigned int allocated; // where the data is allocated, HOST or DEVICE
    Array1D<SizeT, VertexId>  array; // the main data
    Array1D<SizeT, VertexId>  *vertex_associates; // VertexId type associate values
    Array1D<SizeT, Value   >  *value__associates; // Value type associate values
    SizeT        num_vertex_associates;
    SizeT        num_value__associates;
    SizeT        head_a, head_b, tail_a, tail_b; // head and tail offsets
    std::list<Event> events; // In, Out and Block event list
    std::list<cudaEvent_t> empty_gpu_events;
    cudaEvent_t *gpu_events;
    SizeT        num_events;
    std::mutex   queue_mutex;
    int          wait_resize;
    int          wait_temp  ;
    //SizeT        temp_capacity;
    Array1D<SizeT, VertexId> temp_array;
    Array1D<SizeT, VertexId> temp_vertex_associates;
    Array1D<SizeT, Value   > temp_value__associates;
    //char         mssg[512];
    int          lock_counter;
    bool         reduce_show;

public:
    long long    input_iteration, input_iteration_base;
    long long    output_iteration, output_iteration_base;
    long long    iteration_jump;
 
    CircularQueue() :
        name      (""  ),
        gpu_idx   (0   ),
        gpu_num   (0   ),
        input_count(0  ),
        output_count(0 ),
        input_set_flip (0  ),
        input_get_flip (0  ),
        output_set_flip(0  ),
        output_get_flip(0  ),
        input_iteration(0  ),
        input_iteration_base(0),
        output_iteration(0 ),
        output_iteration_base(0),
        iteration_jump  (1 ),
        //target_input_count(MaxValue<int>()),
        //target_output_pos(0),
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
        wait_resize(0  ),
        wait_temp  (0  ),
        lock_counter(0 ),
        //temp_capacity(0)
        reduce_show(true)
    {
        SetName("cq");
        target_input_count[0] = MaxValue<int>();
        target_input_count[1] = MaxValue<int>();
        target_output_pos [0] = MaxValue<SizeT>();
        target_output_pos [1] = MaxValue<SizeT>();
        iteration_length  [0] = 0;
        iteration_length  [1] = 0;
    }

    ~CircularQueue()
    {
        Release();
    }

    __inline__ void Lock(bool in_critical = false)
    {
        if (in_critical) return;
        queue_mutex.lock();
        lock_counter ++;
        if (lock_counter != 1)
        {
            char mssg[128];
            sprintf(mssg, "Error: lock_counter = %d", lock_counter);
            ShowDebugInfo_(__func__, mssg);
        //} else {
        //    ShowDebugInfo_("Locked");
        }
    }

    __inline__ void Unlock(bool in_critical = false)
    {
        if (in_critical) return;
        lock_counter --;
        if (lock_counter != 0)
        {
            char mssg[128];
            sprintf(mssg, "Error: lock_counter = %d", lock_counter);
            ShowDebugInfo_(__func__, mssg);
        //} else {
        //    ShowDebugInfo_("Unlocking");
        }
        queue_mutex.unlock();
    }

    void SetName(std::string name)
    {
        this->name = name;
        array                 .SetName(name+"_array"      );
        temp_array            .SetName(name+"_temp_array" );
        temp_vertex_associates.SetName(name+"_temp_vertex");
        temp_value__associates.SetName(name+"_temp_value" );
    }

    cudaError_t Init(
        SizeT        capacity,
        int          gpu_num  = 0,
        unsigned int target   = HOST,
        SizeT        num_events = 20,
        SizeT        num_vertex_associates = 0,
        SizeT        num_value__associates = 0,
        SizeT        temp_capacity = 0)
    {
        cudaError_t retval = cudaSuccess;

        this -> gpu_num = gpu_num;
        this->capacity = capacity;
        if (retval = array.Allocate(capacity, target)) return retval;
        if (num_vertex_associates != 0)
        {
            if (vertex_associates == NULL)
                vertex_associates = new Array1D<SizeT, VertexId>[num_vertex_associates];
            for (SizeT i=0; i<num_vertex_associates; i++)
            {
                vertex_associates[i].SetName(name + "_vertex[]");
                if (retval = vertex_associates[i].Allocate(capacity, target))
                return retval;
            }
        }
        if (num_value__associates != 0)
        {
            if (value__associates == NULL)
                value__associates = new Array1D<SizeT, Value   >[num_value__associates];
            for (SizeT i=0; i<num_value__associates; i++)
            {
                value__associates[i].SetName(name + "_value[]");
                if (retval = value__associates[i].Allocate(capacity, target))
                return retval;
            }
        }
        this -> num_vertex_associates = num_vertex_associates;
        this -> num_value__associates = num_value__associates;
        allocated = target;
        head_a    = 0; head_b = 0;
        tail_a    = 0; tail_b = 0;
        size_occu = 0; size_soli = 0;
        wait_resize = 0; wait_temp    = 0;
        input_count = 0; output_count = 0;
        target_input_count[0] = MaxValue<int>();
        target_input_count[1] = MaxValue<int>();
        target_output_pos [0] = MaxValue<SizeT>();
        target_output_pos [1] = MaxValue<SizeT>();
        iteration_length  [0] = 0;
        iteration_length  [1] = 0;
        input_set_flip  = 0;
        input_get_flip  = 0;
        output_set_flip = 0;
        output_get_flip = 0;
        input_iteration = 0;
        input_iteration_base = 0;
        output_iteration = 0;
        output_iteration_base = 0;

        if (temp_capacity != 0)
        {
            if (retval = temp_array            .Allocate(
                temp_capacity, target)) 
                return retval;
            if (retval = temp_vertex_associates.Allocate(
                temp_capacity * num_vertex_associates, target))
                return retval;
            if (retval = temp_value__associates.Allocate(
                temp_capacity * num_value__associates, target))
                return retval;
            //this->temp_capacity = temp_capacity;
        }
       
        if (target == DEVICE)
        { 
            if (retval = GRError(cudaGetDevice(&gpu_idx), 
                "cudaGetDevice failed", __FILE__, __LINE__)) return retval;
            if (gpu_events == NULL)
            {
                gpu_events = new cudaEvent_t[num_events];
                this -> num_events = num_events;
                for (SizeT i=0; i<num_events; i++)
                {
                    if (retval = GRError(cudaEventCreateWithFlags(
                        gpu_events + i, cudaEventDisableTiming), 
                        "cudaEventCreateWithFlags failed", __FILE__, __LINE__)) 
                        return retval;
                    empty_gpu_events.push_back(gpu_events[i]);
                }
            }
        }

        events.clear();
        //events[0].clear();
        //events[1].clear();

        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        if (allocated == DEVICE && gpu_events != NULL)
        {
            for (SizeT i=0; i<num_events; i++)
            {
                if (retval = cudaEventDestroy(gpu_events[i])) return retval;
            }
            delete[] gpu_events; gpu_events = NULL;
            empty_gpu_events.clear();
        }
        events.clear();
        //events[0].clear();
        //events[1].clear();

        if (vertex_associates != NULL)
        {
            for (SizeT i=0; i<num_vertex_associates; i++)
                if (retval = vertex_associates[i].Release()) return retval;
            delete[] vertex_associates; vertex_associates = NULL;
        }
        if (value__associates != NULL)
        {
            for (SizeT i=0; i<num_value__associates; i++)
                if (retval = value__associates[i].Release()) return retval;
            delete[] value__associates; value__associates = NULL;
        }

        if (retval = array.Release()                 ) return retval;
        //if (retval = vertex_associates.Release()     ) return retval;
        //if (retval = value__associates.Release()     ) return retval;
        if (retval = temp_array.Release()            ) return retval;
        if (retval = temp_vertex_associates.Release()) return retval;
        if (retval = temp_value__associates.Release()) return retval;

        return retval;
    }

    cudaError_t UpdateSize(bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        if (size_soli != size_occu)
        {
            Lock(in_critical);
            //if (retval = EventCheck(0, true)) 
            //    return Combined_Return(retval, in_critical);
            //if (retval = EventCheck(1, true))
            //    return Combined_Return(retval, in_critical);
            if (retval = EventCheck(EventType::All, true))
                return Combined_Return(retval, in_critical);
            return Combined_Return(retval, in_critical);
        }
        return retval;
    }

    cudaError_t GetSize(SizeT &size_occu, SizeT &size_soli, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = UpdateSize(in_critical)) return retval;
        size_soli = this->size_soli;
        size_occu = this->size_occu;
        return retval;
    }

    SizeT GetSoliSize(bool in_critical = false)
    {
        UpdateSize(in_critical);
        return size_soli;
    }

    SizeT GetOccuSize(bool in_critical = false)
    {
        //char mssg[128];
        UpdateSize(in_critical);
        int size_occu = this -> size_occu;
        //sprintf(mssg, "GetOccuSize, size_occu = %d", size_occu);
        //ShowDebugInfo_(mssg);
        return size_occu;
    }

    bool Empty(bool in_critical = false)
    {
        UpdateSize(in_critical);
        if (size_soli != 0) return false;
        if (size_occu != 0) return false; 
        return true;
    }

    SizeT GetCapacity()
    {
        return capacity;
    }

    SizeT GetTempCapacity()
    {
        return temp_array.GetSize();
    }

    cudaError_t SetInputTarget(
        int target_input_count,
        bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        Lock(in_critical);
        this -> target_input_count[input_set_flip] = target_input_count;
        if (CQ_DEBUG)
        {
            char mssg[128];
            sprintf(mssg,"target_input_count[%d] -> %d",
                input_set_flip, this -> target_input_count[input_set_flip]);
            ShowDebugInfo_(__func__, mssg, input_iteration);
        }
        //input_set_flip ^= 1;

        if (input_count == this->target_input_count[input_get_flip])
        {
            target_output_pos[output_set_flip] = head_a;
            if (CQ_DEBUG)
            {
                char mssg[128];
                sprintf(mssg, "target_output_pos[%d] -> %d",
                    output_set_flip, target_output_pos[output_set_flip]);
                ShowDebugInfo_(__func__, mssg, output_iteration);
            }
            input_count -= this->target_input_count[input_get_flip];
            this->target_input_count[input_get_flip] = MaxValue<int>();
            input_iteration += iteration_jump;
            input_iteration_base += 1;
            iteration_length [input_iteration_base%2] = 0;
            //output_set_flip ^= 1;
            //input_get_flip ^= 1;
        } else if (input_count > this -> target_input_count[input_get_flip])
        {
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "Error: target set too late,"
                    "input_count = %d, target_input_count[%d] = %d",
                    input_count, input_get_flip, 
                    this -> target_input_count[input_get_flip]);
                ShowDebugInfo_(__func__, mssg, input_iteration);
            }
            return GRError(cudaErrorNotReady,
                "target set too late", __FILE__, __LINE__);
        }
        return Combined_Return(retval, in_critical);
    }

    cudaError_t GetResetOutputCount(
        int &output_count, 
        bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        Lock(in_critical);
        output_count = this->output_count;
        this->output_count = 0;
        ShowDebugInfo_(__func__, "output_count -> 0");
        return Combined_Return(retval, in_critical);
    }

    cudaError_t Reset(bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        Lock(in_critical);

        head_a    = 0; head_b = 0;
        tail_a    = 0; tail_b = 0;
        size_occu = 0; size_soli = 0;
        wait_resize = 0; wait_temp = 0;
        input_count = 0; output_count = 0;
        target_input_count[0] = MaxValue<int>();
        target_input_count[1] = MaxValue<int>();
        target_output_pos[0] = MaxValue<SizeT>();
        target_output_pos[1] = MaxValue<SizeT>();
        iteration_length [0] = 0;
        iteration_length [1] = 0;
        input_set_flip = 0;
        input_get_flip = 0;
        output_set_flip = 0;
        output_get_flip = 0;
        input_iteration = 0;
        input_iteration_base = 0;
        output_iteration = 0;
        output_iteration_base = 0;
        events.clear();
        //events[0].clear();
        //events[1].clear();
        reduce_show = true;
        
        while (!empty_gpu_events.empty())
            empty_gpu_events.pop_front();
        for (int i=0; i<num_events; i++)
            empty_gpu_events.push_back(gpu_events[i]);
        ShowDebugInfo_(__func__, "input_count -> 0, output_count -> 0");
        return Combined_Return(retval, in_critical);
    }

    cudaError_t Combined_Return(
        cudaError_t retval = cudaSuccess, 
        bool        in_critical = true,
        bool        set_gpu     = false,
        int         org_gpu     = 0)
    {
        Unlock(in_critical);
        if (retval == cudaSuccess && set_gpu)
            retval = util::SetDevice(org_gpu);
        return retval;
    }

    void ShowDebugInfo(
        const char* function_name,
        const char* mssg_,
        EventType   type,
        SizeT       start,
        SizeT       end,
        SizeT       dsize,
        Value*      value = NULL,
        long long   iteration = -1)
    {
        if (!CQ_DEBUG) return;
        else {
            char mssg[512];
            sprintf(mssg, "%d ~ %d, %s. size_o = %d, size_s = %d,"
                "head_a = %d, head_b = %d, tail_a = %d, tail_b = %d, "
                "i_count = %d, o_count = %d",
                start, end, mssg_, size_occu, size_soli,
                head_a, head_b, tail_a, tail_b,
                input_count, output_count);
            ShowDebugInfo_(function_name, mssg, iteration, type, start, dsize);
        }
    }

    void ShowDebugInfo_(
        const char* function_name,
        const char* mssg,
        long long iteration = -1,
        EventType type = EventType::None,
        SizeT offset = 0,
        SizeT length = 0)
    {
        if (!CQ_DEBUG) return;
        else {
            char str[600];
            if (type == EventType::None)
                sprintf(str, "%s\t %s\t \t %s", name.c_str(), function_name, mssg);
            else sprintf(str, "%s\t %s\t %s,%lld,%lld\t %s",
                name.c_str(), function_name, Event::Type2Str(type),
                (long long)offset, (long long)length, mssg);
            //printf("%d\t \t \t %s\t %s\n", gpu_num, name.c_str(), mssg);
            //fflush(stdout);
            cpu_mt::PrintMessage(str, gpu_num, iteration);
        }
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
        SizeT lengths[2] = {0, 0};
        SizeT sum        = 0;
        long long iteration = input_iteration;
        if (retval = AddSize(length, offsets, lengths)) return retval;

        for (int i=0; i<2; i++)
        {
            if (lengths[i] == 0) continue;
            ShowDebugInfo(__func__, "", EventType::In, offsets[i], 
                offsets[i] + lengths[i], lengths[i], NULL, iteration);
            //if (lengths[i] != 0)
            //{
                if (retval = this->array.Move_In(
                    allocated, allocated, array, 
                    lengths[i], sum, offsets[i], stream)) 
                    return retval;
                for (SizeT j=0; j<num_vertex_associates; j++)
                {
                    if (retval = this->vertex_associates[j].Move_In(
                        allocated, allocated, vertex_associates[j], 
                        lengths[i], sum, offsets[i], stream))
                        return retval;
                }
                for (SizeT j=0; j<num_value__associates; j++)
                {
                    if (retval = this->value__associates[j].Move_In(
                        allocated, allocated, value__associates[j], 
                        lengths[i], sum, offsets[i], stream))
                        return retval;
                }
            //}

            // in_event finish
            //if (allocated == HOST) EventFinish(0, offsets[i], lengths[i]);
            //else if (allocated == DEVICE)
            //    EventSet(0, offsets[i], lengths[i], stream);
            sum += lengths[i];
        }
        if (allocated == HOST) EventFinish(EventType::In, offsets[0], length);
        else if (allocated == DEVICE)
            EventSet(EventType::In, offsets[0], length, stream);
         
        return retval;
    }

    cudaError_t Push_Addr(
        SizeT         length, 
        VertexId    *&array, 
        SizeT        &offset,
        SizeT         num_vertex_associates = 0, 
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL,
        bool          set_gpu = false)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0,0};
        SizeT lengths[2] = {0,0};
        //SizeT sum        = 0;
        //long long iteration = input_iteration;

        if (retval = AddSize(length, offsets, lengths, false, true, set_gpu)) 
            return retval;
        offset = offsets[0];

        if (lengths[1] == 0)
        { // single chunk
            array = this->array.GetPointer(allocated) + offsets[0];
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = this->vertex_associates[j]
                    .GetPointer(allocated) + offsets[0];
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = this->value__associates[j]
                    .GetPointer(allocated) + offsets[0];
        } else { // splict at the end
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "Using temp_array, %d,%d + %d,%d -> %d",
                    offsets[0], lengths[0], offsets[1], lengths[1], length);
                ShowDebugInfo_(__func__, mssg, -1, EventType::In, offsets[0], length);
            }
            if (retval = EnsureTempCapacity(length, length, length, set_gpu))
                return retval;
            array = temp_array.GetPointer(allocated);
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = temp_vertex_associates
                    .GetPointer(allocated) + j*length;
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = temp_value__associates
                    .GetPointer(allocated) + j*length;
        }
        return retval;
    }
 
    cudaError_t Block_Addr(
        SizeT         length, 
        VertexId    *&array, 
        SizeT        &offset,
        SizeT         num_vertex_associates = 0, 
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL,
        bool          set_gpu = false)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0,0};
        SizeT lengths[2] = {0,0};
        //SizeT sum        = 0;
        if (retval = BlockSize(
            length, offsets, lengths, set_gpu)) return retval;
        offset = offsets[0];

        if (lengths[1] == 0)
        { // single chunk
            array = this->array.GetPointer(allocated) + offsets[0];
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = this->vertex_associates[j]
                    .GetPointer(allocated) + offsets[0];
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = this->value__associates[j]
                    .GetPointer(allocated) + offsets[0];
        } else { // splict at the end
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "Using temp_array, %d,%d + %d,%d = %d",
                    offsets[0], lengths[0], offsets[1], lengths[1], length);
                ShowDebugInfo_(__func__, mssg, -1, EventType::Block, offsets[0], length);
            }
           
            if (retval = EnsureTempCapacity(length, length, length, set_gpu))
                return retval;
            array = temp_array.GetPointer(allocated);
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = temp_vertex_associates
                    .GetPointer(allocated) + j*length;
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = temp_value__associates
                    .GetPointer(allocated) + j*length;
        }
        return retval;
    }
 
    cudaError_t Pop(
        SizeT         min_length, 
        SizeT         max_length, 
        VertexId     *array, 
        SizeT        &length, 
        cudaStream_t  stream = 0,
        SizeT         num_vertex_associates = 0,
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lengths[2] = {0, 0};
        SizeT sum        = 0;
        
        if (retval = ReduceSize(min_length, max_length, offsets, lengths)) return retval;

        for (int i=0; i<2; i++)
        {
            if (lengths[i] == 0) continue;
            ShowDebugInfo("Pop", "", EventType::Out, 
                offsets[i], offsets[i] + lengths[i], lengths[i]);
            if (retval = this->array.Move_Out(
                allocated, allocated, array, 
                lengths[i], sum, offsets[i], stream)) 
                return retval;
            for (SizeT j=0; j<num_vertex_associates; j++)
            {
                if (retval = this->vertex_associates[j].Move_Out(
                    allocated, allocated, vertex_associates[j], 
                    lengths[i], sum, offsets[i], stream))
                    return retval;
            }
            for (SizeT j=0; j<num_value__associates; j++)
            {
                if (retval = this->value__associates[j].Move_Out(
                    allocated, allocated, value__associates[j], 
                    lengths[i], sum, offsets[i], stream))
                    return retval;
            }
            //if (allocated == HOST) EventFinish(1, offsets[i], lengths[i]);
            //else if (allocated == DEVICE)
            //    EventSet(1, offsets[i], lengths[i], stream);
            sum += lengths[i];
        }
        length = sum;
        if (allocated == HOST) EventFinish(EventType::Out, offsets[0], length);
        else if (allocated == DEVICE)
            EventSet(EventType::Out, offsets[0], length, stream);

        return retval; 
    }

    cudaError_t Pop_Addr(
        SizeT         min_length, 
        SizeT         max_length, 
        VertexId    *&array, 
        SizeT        &length, 
        SizeT        &offset,
        cudaStream_t  stream = 0,
        SizeT         num_vertex_associates = 0,
        SizeT         num_value__associates = 0,
        VertexId    **vertex_associates = NULL,
        Value       **value__associates = NULL,
        bool          set_gpu = false,
        //bool          allow_smaller = false,
        //int           target_input = util::MaxValue<int>())
        bool         *target_meet = NULL,
        int          *output_count = NULL)
    {
        cudaError_t retval = cudaSuccess;
        SizeT offsets[2] = {0, 0};
        SizeT lengths[2] = {0, 0};
        SizeT sum        = 0;
        //char  mssg[512];
        long long iteration = output_iteration;

        //printf("To Pop, min_length = %d, max_length = %d\n", min_length, max_length);fflush(stdout);
        if (retval = ReduceSize(min_length, max_length, offsets, lengths, 
            //false, false, allow_smaller, target_input)) return retval;
            false, true, target_meet, output_count)) return retval;

        offset = offsets[0];
        length = lengths[0] + lengths[1];
        if (CQ_DEBUG)
        {
            char mssg[128];
            sprintf(mssg, "Length = %d,%d, offset = %d", 
                lengths[0], lengths[1], offset);
            ShowDebugInfo_(__func__, mssg, iteration, EventType::Out, offset, length);
        }

        if (lengths[1] == 0)
        { // single chunk
            array = this->array.GetPointer(allocated) + offset;
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = this->vertex_associates[j]
                    .GetPointer(allocated) + offset;
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = this->value__associates[j]
                    .GetPointer(allocated) + offset; 
        } else {
            int org_gpu = 0;
            if (set_gpu && allocated == DEVICE)
            {    
                if (retval = GRError(cudaGetDevice(&org_gpu),
                    "cudaGetDevice failed", __FILE__, __LINE__))
                    return retval;
                if (retval = SetDevice(gpu_idx)) return retval;
            } 

            if (retval = EnsureTempCapacity(length, length, length, false))
                return retval;
           
            for (int i=0; i<2; i++)
            {
                if (lengths[i] == 0) continue;
                if (retval = this->array.Move_Out(
                    allocated, allocated, temp_array.GetPointer(allocated), 
                    lengths[i], offsets[i], sum, stream)) return retval;
                for (SizeT j=0; j<num_vertex_associates; j++)
                {
                    if (retval = this->vertex_associates[j].Move_Out(
                        allocated, allocated, temp_vertex_associates.GetPointer(allocated), 
                        lengths[i], offsets[i], j*length + sum, stream)) return retval;
                }
                for (SizeT j=0; j<num_value__associates; j++)
                {
                    if (retval = this->value__associates[j].Move_Out(
                        allocated, allocated, temp_value__associates.GetPointer(allocated),
                        lengths[i], offsets[i], j*length + sum, stream)) return retval;
                }
                sum += lengths[i];
            }
            if (set_gpu && allocated == DEVICE)
            {
                if (retval = SetDevice(org_gpu)) return retval;
            }
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "Using temp_array, %d,%d + %d,%d -> %d",
                    offsets[0], lengths[0], offsets[1], lengths[1], sum);
                ShowDebugInfo_(__func__, mssg, -1, EventType::Out, offsets[0], sum);
            }
            array = temp_array.GetPointer(allocated);
            for (SizeT j=0; j<num_vertex_associates; j++)
                vertex_associates[j] = temp_vertex_associates
                    .GetPointer(allocated) + j*length;
            for (SizeT j=0; j<num_value__associates; j++)
                value__associates[j] = temp_value__associates
                    .GetPointer(allocated) + j*length;
           
        }
        return retval; 
    }
 
    cudaError_t AddSize(
        SizeT  length, 
        SizeT *offsets, 
        SizeT *lengths, 
        bool   in_critical = false,
        bool   single_chunk = true,
        bool   set_gpu = false)
    {
        cudaError_t retval = cudaSuccess;
        long long iteration = input_iteration;

        // in critical sectioin
        while (wait_resize != 0)
            //std::this_thread::sleep_for(std::chrono::microseconds(10));
            std::this_thread::yield();
        Lock(in_critical);
        bool past_wait = false;
        while (!past_wait)
        {
            if (wait_resize == 0) {past_wait = true; break;}
            else {
                Unlock();
                //std::this_thread::sleep_for(std::chrono::microseconds(10));
                std::this_thread::yield();
                Lock();
            }
        }
       
        if (allocated == DEVICE)
        {
            if (retval = EventCheck(EventType::Out, true)) 
                return Combined_Return(retval, in_critical);
        }
         
        if (length + size_occu > capacity) 
        { // queue full
            if (AUTO_RESIZE)
            {
                if (retval = EnsureCapacity(length + size_occu, true, set_gpu)) 
                    return Combined_Return(retval, in_critical);
            } else {
                if (length > capacity)
                { // too large for the queue
                    retval = util::GRError(cudaErrorLaunchOutOfResources, 
                        (name + " oversize ").c_str(), __FILE__, __LINE__);
                    return Combined_Return(retval, in_critical);
                } else {
                    Unlock();
                    bool got_space = false;
                    while (!got_space)
                    {
                        if (length + size_occu < capacity)
                        {
                            Lock();
                            if (length + size_occu < capacity)
                            {
                                got_space = true;
                            } else {
                                Unlock();
                            }
                        }
                        if (!got_space) {
                            //std::this_thread::sleep_for(std::chrono::microseconds(10));
                            std::this_thread::yield();
                        }
                    }
                }
            }
        }

        input_count ++;
        if (head_a + length > capacity)
        { // splict
            while (wait_temp != 0)
            {
                Unlock();
                std::this_thread::yield();
                Lock();
            }
            wait_temp = 1;
            offsets[0] = head_a;
            lengths[0] = capacity - head_a;
            offsets[1] = 0;
            lengths[1] = length - lengths[0];
            if (CQ_DEBUG)
                ShowDebugInfo_(__func__, "Temp occu", -1,
                    EventType::In, offsets[0], length);
            if (single_chunk)
            { // only single event
                //EventStart(0, offsets[0], length    , true);
                EventStart(EventType::In, offsets[0], length, true);
            } else { // two events
                //EventStart(0, offsets[0], lengths[0], true);
                //EventStart(0, offsets[1], lengths[1], true);
                EventStart(EventType::In, offsets[0], lengths[0], true);
                EventStart(EventType::In, offsets[1], lengths[1], true);
            }
            head_a     = lengths[1];
        } else { // no splict
            offsets[0] = head_a;
            lengths[0] = length;
            EventStart(EventType::In, offsets[0], lengths[0], true);
            offsets[1] = 0;
            lengths[1] = 0;
            head_a += length;
            if (head_a >= capacity) head_a -= capacity;
        }
        size_occu += length;
        iteration = input_iteration;
        iteration_length [input_iteration_base%2] += length;
        if (input_count == target_input_count[input_get_flip])
        {
            target_output_pos[output_set_flip] = head_a;
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "target_input_count[%d] = %d meet, "
                    "set target_output_pos[%d] -> %d",
                    input_get_flip, target_input_count[input_get_flip], 
                    output_set_flip, target_output_pos[output_set_flip]);
                ShowDebugInfo_(__func__, mssg, iteration, EventType::In,
                    offsets[0], length);
            }
            input_count -= target_input_count[input_get_flip];
            target_input_count[input_get_flip] = MaxValue<int>();
            input_iteration += iteration_jump;
            input_iteration_base += 1;
            iteration_length [input_iteration_base%2] = 0;
            //input_get_flip ^=1;
            //output_set_flip ^= 1;
        }

        if (CQ_DEBUG)
            ShowDebugInfo(__func__, "Done", EventType::In, 
                offsets[0], head_a, length, NULL, iteration);
        return Combined_Return(retval, in_critical);
    }

    cudaError_t BlockSize(
        SizeT  length, 
        SizeT *offsets, 
        SizeT *lengths, 
        bool   in_critical = false,
        bool   single_chunk = true)
    {
        cudaError_t retval = cudaSuccess;
        long long iteration = input_iteration;
        bool past_wait = false;

        // in critical sectioin
        while (wait_resize != 0)
            //std::this_thread::sleep_for(std::chrono::microseconds(10));
            std::this_thread::yield();
        Lock(in_critical);
        while (!past_wait)
        {
            if (wait_resize == 0) {past_wait = true; break;}
            else {
                Unlock();
                std::this_thread::yield();
                //std::this_thread::sleep_for(std::chrono::microseconds(10));
                Lock();
            }
        }
       
        if (allocated == DEVICE)
        {
            if (retval = UpdateSize(true)) 
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
                    Unlock();
                    bool got_space = false;
                    while (!got_space)
                    {
                        if (length + size_occu < capacity)
                        {
                            Lock();
                            if (length + size_occu < capacity)
                            {
                                got_space = true;
                            } else {
                                Unlock();
                            }
                        }
                        if (!got_space) {
                            std::this_thread::yield();
                            //std::this_thread::sleep_for(std::chrono::microseconds(10));
                        }
                    }
                }
            }
        }

        input_count ++;
        output_count ++;
        if (head_a + length > capacity)
        { // splict
            while (wait_temp != 0)
            {
                Unlock();
                std::this_thread::yield();
                Lock();
            }
            wait_temp = 1;
            offsets[0] = head_a;
            lengths[0] = capacity - head_a;
            offsets[1] = 0;
            lengths[1] = length - lengths[0];
            if (CQ_DEBUG)
                ShowDebugInfo_(__func__, "Temp occu", -1, 
                    EventType::Block, offsets[0], length);
            if (single_chunk)
            { // only single event
                //EventStart(0, offsets[0], length    , true);
                //EventStart(1, offsets[0], length    , true);
                EventStart(EventType::Block, offsets[0], length, true);
            } else { // two events
                //EventStart(0, offsets[0], lengths[0], true);
                //EventStart(1, offsets[0], lengths[0], true);
                //EventStart(0, offsets[1], lengths[1], true);
                //EventStart(1, offsets[1], lengths[1], true);
                EventStart(EventType::Block, offsets[0], lengths[0], true);
                EventStart(EventType::Block, offsets[1], lengths[1], true);
            }
            head_a     = lengths[1];
            tail_a     = lengths[1];
        } else { // no splict
            offsets[0] = head_a;
            lengths[0] = length;
            //EventStart(0, offsets[0], lengths[0], true);
            //EventStart(1, offsets[0], lengths[0], true);
            EventStart(EventType::Block, offsets[0], lengths[0], true);
            offsets[1] = 0;
            lengths[1] = 0;
            head_a += length;
            tail_a += length;
            if (head_a >= capacity) head_a -= capacity;
            if (tail_a >= capacity) tail_a -= capacity;
        }
        size_occu += length;
        size_soli -= length;
        iteration = input_iteration;
        if (input_count == target_input_count[input_get_flip])
        {
            if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "target_input_count[%d] = %d meet, "
                    "set target_output_pos[%d] -> %d",
                    input_get_flip, target_input_count[input_get_flip], 
                    output_set_flip, target_output_pos[output_set_flip]);
                ShowDebugInfo_(__func__, mssg, input_iteration, 
                    EventType::Block, offsets[0], length);
            }
            target_output_pos[output_set_flip] = head_a;
            input_count -= target_input_count[input_get_flip];
            input_iteration += iteration_jump;
            input_iteration_base += 1;
            //input_get_flip ^= 1;
            //output_set_flip ^= 1;
        }

        if (CQ_DEBUG)
        {
            //ShowDebugInfo("AddSize", 0, offsets[0], head_a, length,
            //    NULL, iteration);
            //ShowDebugInfo("RedSize", 1, offsets[0], tail_a, length,
            //    NULL, iteration);
            ShowDebugInfo(__func__, "Done", EventType::Block, 
                offsets[0], head_a, length, NULL, iteration);
        }
        return Combined_Return(retval, in_critical);
    }

    cudaError_t ReduceSize(
        SizeT  min_length, 
        SizeT  max_length, 
        SizeT *offsets, 
        SizeT *lengths, 
        bool   in_critical = false,
        bool   single_chunk = true,
        //bool   allow_smaller = false,
        //int    target_input = util::MaxValue<int>())
        bool  *target_meet = NULL,
        int   *output_count_ = NULL)
    {
        cudaError_t retval = cudaSuccess;
        SizeT length = 0;
        bool  on_target = false;
        long long iteration = output_iteration;

        // in critial section
        if (wait_resize != 0)
            return cudaErrorNotReady;

        //while (wait_resize != 0)
        //    std::this_thread::sleep_for(std::chrono::microseconds(10));
        Lock(in_critical);

        if (allocated == DEVICE)
        {
            if (retval = EventCheck(EventType::In, true))
                return Combined_Return(retval, in_critical);
        }
      
        length = size_soli < max_length ? size_soli : max_length;
        SizeT t_o_pos = target_output_pos[output_get_flip];
        if //(input_count >= target_input_count[input_get_flip] && // target_output_pos set
           (((t_o_pos >= tail_a &&     // normal situation
              t_o_pos <= tail_a + length) || 
             (t_o_pos <  tail_a &&     // warp around
              t_o_pos <= tail_a + length - capacity)))
        {
            /*if (CQ_DEBUG)
            {
                char mssg[256];
                sprintf(mssg, "Testing target, target_output_pos = %d, "
                    "iteration_length[%lld] = %d",
                    t_o_pos, output_iteration_base%2,
                    iteration_length[output_iteration_base%2]);
                ShowDebugInfo_(__func__, mssg, output_iteration,
                    EventType::Out, tail_a, length);
            }*/
            if (t_o_pos >= tail_a &&
                t_o_pos <= tail_a + length)
            {
                if (length == capacity)
                    length = t_o_pos + capacity - tail_a;
                else length = t_o_pos - tail_a;
            } else if (t_o_pos < tail_a &&
                t_o_pos <= tail_a + length - capacity)
            {
                length = t_o_pos + capacity - tail_a;    
            }
            if (length == 0 && t_o_pos == tail_a 
                && iteration_length[output_iteration_base%2] != 0)
            { // special case, queue is occu-full, but input not finished yet
                on_target = false;
            } else on_target = true;
        }

        if ((reduce_show || on_target) && CQ_DEBUG)
        {
            char mssg[256];
            sprintf(mssg, //"input_count = %d, target_input_count[%d] = %d, "
                "target_output_pos[%d] = %d, capacity = %d, iteration_length[%lld] = %d", 
                /*input_count, input_get_flip,
                target_input_count[input_get_flip],*/ 
                output_get_flip, target_output_pos [output_get_flip],
                capacity, output_iteration_base%2, iteration_length[output_iteration_base%2]);
            ShowDebugInfo_(__func__, mssg, output_iteration,
                EventType::Out, tail_a, length);
        }
        //printf("size_soli = %d, size_occu = %d, input_count = %d, target_input = %d, min_length = %d, max_length = %d\n",
        //    size_soli, size_occu, input_count, target_input, min_length, max_length);
        //fflush(stdout);
        //if (allow_smaller && target_input <= input_count)
        //{
        //    sprintf(mssg, "ToReduce, min_length = %d, allow_smaller = %s, "
        //        "target_input = %d, input_count = %d, size_soli = %d, "
        //        "size_occu = %d", min_length, allow_smaller ? "true":"false",
        //        target_input, input_count, size_soli, size_occu);
        //    ShowDebugInfo_(mssg);
        //}
 
        if (size_soli < min_length && 
            //!(allow_smaller && target_input <= input_count && size_soli < min_length))
            !on_target)
        { // too small
            /*//queue_mutex.unlock();
            bool got_content = false;
            while (!got_content)
            {
                if (size_soli >= min_length)
                {
                    //queue_mutex.lock();
                    if (retval = EventCheck(0, true))
                        return Combined_Return(retval, in_critical);
                    if (size_soli >= min_length)
                    {
                        got_content = true;
                    } else {
                        //queue_mutex.unlock();
                    }
                }
                if (!got_content) {
                    queue_mutex.unlock();
                    printf("waiting for content, size_soli = %d, size_occu = %d, min_length = %d\n", size_soli, size_occu, min_length);fflush(stdout);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    queue_mutex.lock();
                }
            }*/
            reduce_show = false;
            retval = cudaErrorNotReady;
            return Combined_Return(retval, in_critical);
        }

        output_count ++;
        iteration = output_iteration; 
        //if (CQ_DEBUG && size_soli < min_length && allow_smaller && 
        //    target_input <= input_count)
        if (CQ_DEBUG && on_target)
        {
            char mssg[512];
            sprintf(mssg, "On target: size_soli = %d, size_occu = %d,"
                " t_input[%d] = %d, i_count = %d, o_count = %d, tail_a = %d,"
                " length = %d, capacity = %d, target_output_pos[%d] = %d",
                size_soli, size_occu, input_get_flip, 
                target_input_count[input_get_flip], 
                input_count, output_count,tail_a, length, 
                capacity, output_get_flip,
                target_output_pos[output_get_flip]);
            ShowDebugInfo_(__func__, mssg, output_iteration,
                EventType::Out, tail_a, length);
        }

        if (tail_a + length > capacity)
        { // splict
            while (wait_temp != 0)
            {
                Unlock();
                std::this_thread::yield();
                Lock();
            }
            wait_temp = 1;
            offsets[0] = tail_a;
            lengths[0] = capacity - tail_a;
            offsets[1] = 0;
            lengths[1] = length - lengths[0];
            if (CQ_DEBUG)
                ShowDebugInfo_(__func__, "Temp occu", -1, 
                    EventType::Out, offsets[0], length);
            if (single_chunk)
            { // single event
                //EventStart(1, offsets[0], length    , true);
                EventStart(EventType::Out, offsets[0], length    , true);
            } else { // two events
                //EventStart(1, offsets[0], lengths[0], true);
                //EventStart(1, offsets[1], lengths[1], true);
                EventStart(EventType::Out, offsets[0], lengths[0], true);
                EventStart(EventType::Out, offsets[1], lengths[1], true);
            }
            tail_a     = lengths[1];
        } else {
            offsets[0] = tail_a;
            lengths[0] = length;
            EventStart(EventType::Out, offsets[0], lengths[0], true);
            offsets[1] = 0;
            lengths[1] = 0;
            tail_a += length;
            if (tail_a == capacity) tail_a = 0;
        }
        size_soli -= length;
        iteration_length [output_iteration_base%2] -= length;

        if (on_target)
        {
            //input_count -= target_input_count[input_get_flip];
            //target_input_count[input_get_flip] = MaxValue<int>();
            target_output_pos[output_get_flip] = MaxValue<SizeT>();
            if (target_meet != NULL) target_meet[0] = true;
            if (output_count_ != NULL) output_count_[0] = output_count;
            output_count = 0;
            if (CQ_DEBUG)
            {
                char mssg[128];
                sprintf(mssg, //"target_input_count[%d] -> %d, "
                    "target_output_pos[%d] -> %d, output_count -> 0",
                    //input_get_flip, target_input_count[input_get_flip],
                    output_get_flip, target_output_pos[output_get_flip]);
                output_iteration += iteration_jump;
                output_iteration_base += 1;
                ShowDebugInfo_(__func__, mssg, output_iteration, 
                    EventType::Out, offsets[0], length);
            }
            //input_get_flip ^= 1;
            //output_get_flip ^= 1;
        } else {
            if (target_meet != NULL) target_meet[0] = false;
        }

        ShowDebugInfo(__func__, "Done", EventType::Out, offsets[0], 
                tail_a, length, NULL, iteration);
        return Combined_Return(retval, in_critical);
    }

    cudaError_t EnsureTempCapacity(
        SizeT temp_capacity,
        SizeT vertex_capacity,
        SizeT value__capacity,
        bool set_gpu = false)
    {
        cudaError_t retval = cudaSuccess;

         if (temp_capacity > temp_array.GetSize() ||
             vertex_capacity * num_vertex_associates > temp_vertex_associates.GetSize() || 
             value__capacity * num_value__associates > temp_value__associates.GetSize())
        {
            if (!AUTO_RESIZE)
            {
                retval = util::GRError(cudaErrorLaunchOutOfResources, 
                    (name + " remp_array oversize ").c_str(), __FILE__, __LINE__);
                return retval;
            } else {
                int org_gpu = 0;
                if (set_gpu && allocated == DEVICE)
                {
                    if (retval = GRError(cudaGetDevice(&org_gpu),
                        "cudaGetDevice failed", __FILE__, __LINE__))
                        return retval;
                    if (retval = SetDevice(gpu_idx)) return retval;
                }
                if (CQ_DEBUG)
                {
                    char mssg[128];
                    sprintf(mssg, "TempCapacity -> %d, allocated = %d", 
                        temp_capacity, allocated);
                    ShowDebugInfo_(__func__, mssg);
                }
                if (retval = temp_array            .EnsureSize(
                    temp_capacity                        , false, 0, allocated))
                    return retval;
                if (retval = temp_vertex_associates.EnsureSize(
                    vertex_capacity * num_vertex_associates, false, 0, allocated))
                    return retval;
                if (retval = temp_value__associates.EnsureSize(
                    value__capacity * num_value__associates, false, 0, allocated))
                    return retval;
                if (set_gpu && allocated == DEVICE)
                {
                    if (retval = SetDevice(org_gpu)) return retval;
                }
            }
        }
        return retval;
    }

    cudaError_t EnsureCapacity(
        SizeT capacity_, 
        bool  in_critical = false,
        bool  set_gpu     = false)
    {
        cudaError_t retval = cudaSuccess;
        int org_gpu;
        
        if (set_gpu && allocated == DEVICE)
        { // set to the correct device
            if (retval = GRError(cudaGetDevice(&org_gpu),
                "cudaGetDevice failed", __FILE__, __LINE__))
                return retval;
            if (retval = SetDevice(gpu_idx)) return retval;
        }

        Lock(in_critical);
        //ShowDebugInfo_("EnsureCapacity locked");
        if (CQ_DEBUG)
        {
            char mssg[128];
            sprintf(mssg, "Capacity (%d) -> %d", capacity, capacity_);
            ShowDebugInfo_(__func__, mssg);
        }
        
        capacity_ ++;
        if (capacity_ > capacity)
        {
            wait_resize = 1;
            while (!events.empty())//((!events[0].empty()) || (!events[1].empty()))
            {
                //ShowDebugInfo_("EnsureCapacity unlocking");
                Unlock();
                //std::this_thread::sleep_for(std::chrono::microseconds(1000));
                std::this_thread::yield();
                Lock();
                //ShowDebugInfo_("EnsureCapacity locked");
                if (retval = UpdateSize(true))
                    return Combined_Return(retval, in_critical);
                //for (int i=0; i<2; i++)
                //if (retval = EventCheck(i, true))
                //{
                //    Unlock();
                //    return retval;
                //}
            }
            if (CQ_DEBUG)
                ShowDebugInfo_(__func__, "Events clear");

            if (retval = array.EnsureSize(capacity_, true, 0, allocated)) 
                return Combined_Return(retval, in_critical);
            for (SizeT i=0; i<num_vertex_associates; i++)
            {
                if (retval = vertex_associates[i].EnsureSize(capacity_, true, 0, allocated))
                    return Combined_Return(retval, in_critical);
            }
            for (SizeT i=0; i<num_value__associates; i++)
            {
                if (retval = value__associates[i].EnsureSize(capacity_, true, 0, allocated))
                    return Combined_Return(retval, in_critical);
            }

            if (tail_a + size_occu > capacity)
            {            

                if (tail_a + size_occu > capacity_)
                { // Content cross original and new end point
                    SizeT lengths[2] = {0, 0};
                    lengths[0] = capacity_ - capacity;
                    lengths[1] = head_a - lengths[0];

                    if (retval = temp_array.EnsureSize(lengths[1], false, 0, allocated))
                        return Combined_Return(retval, in_critical);
                    if (num_value__associates != 0)
                    if (retval = temp_value__associates.
                        EnsureSize(lengths[1], false, 0, allocated))
                        return Combined_Return(retval, in_critical);

                    if (retval = array.Move_Out(allocated, allocated,
                        array       .GetPointer(allocated), lengths[0], 0, capacity))
                        return Combined_Return(retval, in_critical);
                    if (retval = array.Move_Out(allocated, allocated,
                        temp_array  .GetPointer(allocated), lengths[1], lengths[0], 0))
                        return Combined_Return(retval, in_critical);
                    if (retval = array.Move_In (allocated, allocated,
                        temp_array  .GetPointer(allocated), lengths[1], 0, 0))
                        return Combined_Return(retval, in_critical);

                    for (SizeT i=0; i<num_vertex_associates; i++)
                    {
                        if (retval = vertex_associates[i].Move_Out(allocated, allocated,
                            vertex_associates[i].GetPointer(allocated), 
                            lengths[0], 0, capacity))
                            return Combined_Return(retval, in_critical);
                        if (retval = vertex_associates[i].Move_Out(allocated, allocated,
                            temp_array .GetPointer(allocated), lengths[1], lengths[0], 0))
                            return Combined_Return(retval, in_critical);
                        if (retval = vertex_associates[i].Move_In (allocated, allocated,
                            temp_array .GetPointer(allocated), lengths[1], 0, 0))
                            return Combined_Return(retval, in_critical);
                    }
                    for (SizeT i=0; i<num_value__associates; i++)
                    {
                        if (retval = value__associates[i].Move_Out(allocated, allocated,
                            value__associates[i].GetPointer(allocated), 
                            lengths[0], 0, capacity))
                            return Combined_Return(retval, in_critical);
                        if (retval = value__associates[i].Move_Out(allocated, allocated,
                            temp_value__associates.GetPointer(allocated), 
                            lengths[1], lengths[0], 0))
                            return Combined_Return(retval, in_critical);
                        if (retval = value__associates[i].Move_In (allocated, allocated,
                            temp_value__associates.GetPointer(allocated), 
                            lengths[1], 0, 0))
                            return Combined_Return(retval, in_critical);
                    }
                   
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

                SizeT t_o_pos = target_output_pos[output_get_flip];
                if   (t_o_pos <= tail_a &&     // warp around
                      t_o_pos <= tail_a + size_occu - capacity)
                {
                    t_o_pos = (capacity + t_o_pos) % capacity_;
                    if (CQ_DEBUG)
                    {
                        char mssg[128];
                        sprintf(mssg, "target_output_pos[%d] (%d) -> %d",
                            output_get_flip, target_output_pos[output_get_flip], t_o_pos);
                        ShowDebugInfo_(__func__, mssg);
                    }
                    target_output_pos[output_get_flip] = t_o_pos;
                } else if (CQ_DEBUG)
                {
                    char mssg[128];
                    sprintf(mssg, "target_output_pos[%d] (%d)",
                        output_get_flip, target_output_pos[output_get_flip]);
                    ShowDebugInfo_(__func__, mssg);
                }
            }

            capacity = capacity_;
            head_a = (tail_a + size_occu) % capacity;
            head_b = head_a;
            //temp_array.Release();
            if (CQ_DEBUG)
            {
                char mssg[128];
                sprintf(mssg, "Capacity -> %d, head_a -> %d",
                    capacity, head_a);
                ShowDebugInfo_(__func__, mssg);
            }
            wait_resize = 0;
        }
        Unlock(in_critical);
        if (allocated == DEVICE && set_gpu)
        {
            if (retval = SetDevice(org_gpu))
                return retval;
        }
        return retval;
    }

    void EventStart(
        EventType type, 
        SizeT offset, 
        SizeT length, 
        bool in_critical = false)
    {
        Lock(in_critical);
        if (length == 0)
        {
            //if (CQ_DEBUG)
            //{
            //    sprintf(mssg,"Event %d,%d,%d starts -> skip, input_count = %d, "
            //        "output_count = %d",
            //        direction, offset, length,
            //        input_count, output_count);
            //    ShowDebugInfo_(mssg);
            //}
        } else {
            if (CQ_DEBUG)
            {
                char mssg[128];
                sprintf(mssg, "Starts, input_count = %d, "
                    "output_count = %d", 
                    input_count, output_count);
                ShowDebugInfo_(__func__, mssg, -1, type, offset, length);
            }
            //events[direction].push_back(CqEvent(offset, length));
            events.push_back(Event(type, offset, length));
        }
        Unlock(in_critical);
    }

    cudaError_t EventSet(
        EventType  type, 
        SizeT offset, 
        SizeT length, 
        cudaStream_t stream = 0, 
        bool in_critical = false,
        bool set_gpu     = false,
        cudaEvent_t* src_event = NULL,
        cudaEvent_t* des_event = NULL)
    {
        cudaError_t retval = cudaSuccess;
        if (allocated != DEVICE) return retval;
        SizeT offsets[2] = {0, 0};
        SizeT lengths[2] = {0, 0};
        int org_gpu;
        

        if (length == 0)
        {
            if (CQ_DEBUG)
            {
                //Lock(in_critical);
                //sprintf(mssg, "Event %d,%d,%d sets -> skip, input_count = %d,"
                //    " output_count = %d", 
                //    direction, offset, length,
                //    input_count, output_count);
                //ShowDebugInfo_(mssg);
                //Unlock(in_critical);
            }
            return retval;
        }

        if (set_gpu && allocated == DEVICE)
        { // set to the correct device
            if (retval = GRError(cudaGetDevice(&org_gpu),
                "cudaGetDevice failed", __FILE__, __LINE__))
                return retval;
            if (retval = SetDevice(gpu_idx)) return retval;
        }

        if (offset + length > capacity)
        { // single chunk crossing the end, and in event
            SizeT sum        = 0;
            offsets[0] = offset; offsets[1] = 0;
            lengths[0] = capacity - offset; lengths[1] = length - lengths[0];

            //if (direction == 0)
            if (type == EventType::In)
            {
                if (length > 0 && src_event != NULL)
                {
                    if (retval = GRError(cudaStreamWaitEvent(stream, src_event[0], 0),
                        "cudaStreamWaitEvent failed", __FILE__, __LINE__))
                        return retval;
                    src_event = NULL;
                }

                for (int i=0; i<2; i++)
                {
                    if (lengths[i] == 0) continue;
                    if (retval = array.Move_In(
                        allocated, allocated, temp_array.GetPointer(allocated), 
                        lengths[i], sum, offsets[i], stream)) return retval;
                    for (SizeT j=0; j<num_vertex_associates; j++)
                    {
                        if (retval = vertex_associates[j].Move_In(
                            allocated, allocated, temp_vertex_associates.GetPointer(allocated),
                            lengths[i], j*length + sum, offsets[i], stream)) return retval; 
                    }
                    for (SizeT j=0; j<num_value__associates; j++)
                    {
                        if (retval = value__associates[j].Move_In(
                            allocated, allocated, temp_value__associates.GetPointer(allocated),
                            lengths[i], j*length + sum, offsets[i], stream)) return retval;
                    }
                    sum += lengths[i];
                }
                if (CQ_DEBUG)
                {
                    char mssg[256];
                    sprintf(mssg, "Copy from temp, %d -> %d,%d + %d,%d",
                        sum, offsets[0], lengths[0], offsets[1], lengths[1]);
                    ShowDebugInfo_(__func__, mssg, -1, type, offset, length);
                }
            }
        } //else {
            offsets[0] = offset; offsets[1] = 0;
            lengths[0] = length; lengths[1] = 0;
        //}

        if (CQ_DEBUG)
            ShowDebugInfo_(__func__, "Locking", -1, type, offset, length);
        Lock(in_critical);
        if (CQ_DEBUG)
            ShowDebugInfo_(__func__, "Locked", -1, type, offset, length);

        int i=0;
        //for (int i=0; i<2; i++)
        //{
            cudaEvent_t event = NULL;
            if (src_event == NULL)
            {
                //if (lengths[i] == 0 && i!=0) continue;
                //if (lengths[i] != 0)
            //{
                if (empty_gpu_events.empty())
                {
                    retval = util::GRError(cudaErrorLaunchOutOfResources,
                        (name + " gpu_events oversize ").c_str(), __FILE__, __LINE__);
                    Unlock(in_critical);
                    return retval;    
                }
                event = empty_gpu_events.front();
                empty_gpu_events.pop_front();
                if (retval = cudaEventRecord(event, stream))
                {
                    Unlock(in_critical);
                    return retval;
                }
            //}
            } else {
                event = src_event[0];
            }

            typename std::list<Event>::iterator it = events.begin();
            //for (it  = events[direction].begin(); 
            //     it != events[direction].end(); it ++)
            for (it = events.begin(); it != events.end(); it++)
            {
                if ((type       == (*it).type  ) &&
                    (offsets[i] == (*it).offset) && 
                    (lengths[i] == (*it).length))// &&
                    //((*it).status == CqEvent::New)) // matched event
                {
                    if (CQ_DEBUG)
                    {
                        char mssg[128];
                        sprintf(mssg, "Set, input_count = %d,"
                            " output_count = %d", 
                            input_count, output_count);
                        ShowDebugInfo_(__func__, mssg, -1,
                            type, offsets[i], lengths[i]);
                    }
                    (*it).event = event;
                    (*it).status = Event::Assigned;
                    if (src_event != NULL) (*it).externel = true;
                    else (*it).externel = false;
                    if (des_event != NULL) des_event[0] = event;
                    if (offset+length > capacity)
                        (*it).use_temp = true;
                    else (*it).use_temp = false;
                    break;
                } //else {
                //    sprintf(mssg, "EventSet looking for %d,%d,%d, having %d,%d,%d",
                //        direction, offsets[i], lengths[i],
                //        direction, (*it).offset, (*it).length);
                //    ShowDebugInfo_(mssg);
                //}
            }

            //if (it == events[direction].end())
            if (it == events.end())
            {
                if (CQ_DEBUG)
                    ShowDebugInfo_(__func__, "Error: Can not be found", -1,
                        type, offsets[i], lengths[i]);
            }
        //}
        //EventCheck(direction, true);
        if (CQ_DEBUG)
            ShowDebugInfo_(__func__, "Unlocking", -1, type, offset, length);
        Unlock(in_critical);
        if (allocated == DEVICE && set_gpu)
        {
            if (retval = SetDevice(org_gpu))
                return retval;
        }
        return retval;
    }

    cudaError_t EventFinish(
        //int   direction, 
        EventType type,
        SizeT offset, 
        SizeT length, 
        bool  in_critical = false,
        cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;

        SizeT offsets[2] = {0, 0};
        SizeT lengths[2] = {0, 0};

        if (length == 0)
        {
            //if (CQ_DEBUG)
            //{
            //    Lock(in_critical);
            //    sprintf(mssg, "Event %d,%d,%d done -> skip. input_count = %d,"
            //        " output_count = %d", 
            //        direction, offset, length,
            //        input_count, output_count);
            //    ShowDebugInfo_(mssg);
            //    Unlock(in_critical);
            //}
            return retval;
        }

        if (offset + length > capacity)
        { // single chunk crossing the end, and in event
            SizeT sum        = 0;
            offsets[0] = offset; offsets[1] = 0;
            lengths[0] = capacity - offset; lengths[1] = length - lengths[0];

            if (type == EventType::In)
            {
                for (int i=0; i<2; i++)
                {
                    if (lengths[i] == 0) continue;
                    if (retval = array.Move_In(
                        allocated, allocated, temp_array.GetPointer(allocated), 
                        lengths[i], sum, offsets[i], stream)) return retval;
                    for (SizeT j=0; j<num_vertex_associates; j++)
                    {
                        if (retval = vertex_associates[j].Move_In(
                            allocated, allocated, temp_vertex_associates.GetPointer(allocated),
                            lengths[i], j*length + sum, offsets[i], stream)) return retval; 
                    }
                    for (SizeT j=0; j<num_value__associates; j++)
                    {
                        if (retval = value__associates[j].Move_In(
                            allocated, allocated, temp_value__associates.GetPointer(allocated),
                            lengths[i], j*length + sum, offsets[i], stream)) return retval;
                    }
                    sum += lengths[i];
                }
            }
            if (allocated == DEVICE && stream != 0)
            {
                if (retval = GRError(cudaStreamSynchronize(stream),
                    name + "cudaStreamSynchronize failed", __FILE__, __LINE__)) return retval;
            }
        } //else {
            offsets[0] = offset; lengths[0] = length;
            offsets[1] = 0; lengths[1] = 0;
        //}

        Lock(in_critical);
        int i=0;
        //for (int i=0; i<2; i++)
        //{
            //typename std::list<CqEvent>::iterator it = events[direction].begin();
            typename std::list<Event>::iterator it = events.begin();
            //for (it  = events[direction].begin(); 
            //     it != events[direction].end(); it ++)
            for (it = events.begin(); it != events.end(); it++)
            {
                if ((type       == (*it).type  ) &&
                    (offsets[i] == (*it).offset) && 
                    (lengths[i] == (*it).length) &&
                    ((*it).status != Event::Finished))
                    //(((*it).status == CqEvent::Assigned) || 
                    // ((*it).status == CqEvent::New))) // matched event
                {
                    //if (direction == 0) input_count ++;
                    if (CQ_DEBUG)
                    {
                        char mssg[128];
                        sprintf(mssg, "Done. input_count = %d,"
                            " output_count = %d", 
                            input_count, output_count);
                        ShowDebugInfo_(__func__, mssg, -1, 
                            type, offsets[i], lengths[i]);
                    }
                    (*it).status = Event::Finished;
                    if (allocated == DEVICE && (*it).event != NULL 
                        && !(*it).externel)
                    {
                        empty_gpu_events.push_back((*it).event);
                    }
                    break;
                }
            }
            if (CQ_DEBUG && it == events.end())
            {
                ShowDebugInfo_(__func__, "Error: Can not be found", 
                    -1, type, offsets[i], lengths[i]);
            }
        //}
        SizeCheck(type, true);
        if (CQ_DEBUG)
        {
            ShowDebugInfo(__func__, "Done", type, offset, -1, length);
        }
        Unlock(in_critical);
        return retval;
    }

    cudaError_t EventCheck(EventType type, bool in_critical = false)
    {
        cudaError_t retval = cudaSuccess;
        Lock(in_critical);

        //int size = events[direction].size();
        int size = events.size();
        int e_size = empty_gpu_events.size();
        //int size_ = events[direction^1].size();
        if (size == 0)
        {
            Unlock(in_critical);
            return retval;
        }

        //typename std::list<CqEvent>::iterator it = events[direction].begin();
        typename std::list<Event>::iterator it = events.begin();
        //sprintf(mssg, "EventCheck direction = %d, size = %d, empty_size = %d",
        //    direction, size, e_size);
        //ShowDebugInfo_(mssg);
        //for (it  = events[direction].begin();
        //     it != events[direction].end(); it++)
        for (it = events.begin(); it != events.end(); it++)
        {
            if ((*it).status == Event::Assigned &&
                (type == EventType::All || type == (*it).type))
            {
                if (((*it).length == 0 && (*it).event == NULL) || 
                    (allocated != DEVICE))
                    retval = cudaSuccess;
                else retval = cudaEventQuery((*it).event);
                if (retval == cudaSuccess)
                {
                    (*it).status = Event::Finished;
                    //if (direction == 0) input_count ++;
                    if (CQ_DEBUG)
                    {
                        char mssg[256];
                        sprintf(mssg, "Finish: "
                            "input_count = %d, output_count = %d", 
                            input_count, output_count);
                        ShowDebugInfo_(__func__, mssg, -1, 
                            (*it).type, (*it).offset, (*it).length);
                    }
                    if (allocated == DEVICE && (*it).event != NULL &&
                        !(*it).externel)
                        empty_gpu_events.push_back((*it).event);
                    if ((*it).use_temp)// && type == EventType::Out)
                    {
                        ShowDebugInfo_(__func__, "Temp un-occu", -1,
                            (*it).type, (*it).offset, (*it).length);
                        wait_temp = 0;
                    }
                } else if (retval != cudaErrorNotReady) {
                    if (CQ_DEBUG)
                        ShowDebugInfo_(__func__, "Error", -1,
                            (*it).type, (*it).offset, (*it).length);
                    Unlock(in_critical);
                    return retval;
                } else retval = cudaSuccess;
            }
        }
        SizeCheck(type, true);
        //ShowDebugInfo("EventC", direction, -1, -1, -1);
        Unlock(in_critical);
        return retval; 
    }

    void SizeCheck(EventType type, bool in_critical = false)
    {
        Lock(in_critical);
        //ShowDebugInfo_("SizeCheck begin.");
        //typename std::list<CqEvent>::iterator it = events[direction].begin();
        typename std::list<Event>::iterator it = events.begin();
        //CqEvent *event = NULL;

        //while (!events[direction].empty())
        while (it != events.end())
        {
            //event = &(events[direction].front());
            //it = events[direction].begin();
            //printf("Event %d, %d, %d, status = %d\n", direction, (*it).offset, (*it).length, (*it).status);fflush(stdout);
            //if (event -> status == CqEvent::Finished) // finished event
            if ((*it).status == Event::Finished &&
                ((type == EventType::All) || (type == (*it).type)))
            {
                bool to_remove = false;
                if ((*it).type == EventType::In)
                { // in event
                    if ((*it).offset == head_b)
                    {
                        head_b += (*it).length;
                        if (head_b >= capacity) head_b -= capacity;
                        (*it).status = Event::Cleared;
                        size_soli += (*it).length;
                        to_remove = true;
                    } else {
                        //sprintf(mssg, "offset = %d, head_b = %d",
                        //    event -> offset, head_b);
                        //ShowDebugInfo_(mssg);
                    } 
                } else if ((*it).type == EventType::Out)
                { // out event
                    if ((*it).offset == tail_b)
                    {
                        tail_b += (*it).length;
                        if (tail_b >= capacity) tail_b -= capacity;
                        (*it).status = Event::Cleared;
                        size_occu -= (*it).length;
                        to_remove = true;
                    } else {
                        //sprintf(mssg, "offset = %d, tail_b = %d", 
                        //    event -> offset, tail_b);
                        //ShowDebugInfo_(mssg);
                    }
                } else if ((*it).type == EventType::Block)
                { // block event
                    if ((*it).offset == head_b)
                    {
                        head_b += (*it).length;
                        if (head_b >= capacity) head_b -= capacity;
                        tail_b += (*it).length;
                        if (tail_b >= capacity) tail_b -= capacity;
                        (*it).status = Event::Cleared;
                        size_occu -= (*it).length;
                        to_remove = true;
                    } else {
                    }
                }
                
                if (to_remove)
                {
                    if (CQ_DEBUG)
                    {
                        char mssg[256];
                        sprintf(mssg, "Cleared, head_b -> %d, tail_b -> %d, "
                            "size_occu -> %d, size_soli -> %d",
                            head_b, tail_b, size_occu, size_soli);
                        ShowDebugInfo_(__func__, mssg, -1,
                            (*it).type, (*it).offset, (*it).length);
                    }
                    //events[direction].pop_front();
                    it = events.erase(it);
                } else it++;
            } else {
                //sprintf(mssg, "event %d,%d,%d not finished",
                //    direction, event -> offset, event -> length);
                //ShowDebugInfo_(mssg);
                //break;
                it++;
            }
        }

        Unlock(in_critical);
    }

    
}; // end of struct CircularQueue

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

