// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_loop.cuh
 *
 * @brief Base Iteration Loop
 */

#pragma once

#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/app/enactor_helper.cuh>

namespace gunrock {
namespace app {

// TODO: embed into subq and fullq thread
/*template <
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES,
    typename Enactor,
    typename Functor,
    typename Iteration>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>  GraphSlice;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_gpus             =   problem     -> num_gpus;
    int           thread_num           =   thread_data -> thread_num;
    DataSlice    *data_slice           =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSlice   *graph_slice          =   problem     -> graph_slices       [thread_num] ;
    GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attribute   = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);  
    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);  
    util::CtaWorkProgressLifetime
                 *work_progress        = &(enactor     -> work_progress      [thread_num * num_gpus]);
    ContextPtr   *context              =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    SizeT         Total_Length         =   0;
    cudaError_t   tretval              =   cudaSuccess;
    int           grid_size            =   0;
    std::string   mssg                 =   "";
    int           pre_stage            =   0;
    size_t        offset               =   0;
    int           iteration            =   0;
    int           selector             =   0;
    util::DoubleBuffer<SizeT, VertexId, Value>
                 *frontier_queue_      =   NULL;
    FrontierAttribute<SizeT>
                 *frontier_attribute_  =   NULL;
    EnactorStats *enactor_stats_       =   NULL;
    util::CtaWorkProgressLifetime
                 *work_progress_       =   NULL;
    util::Array1D<SizeT, SizeT>
                 *scanned_edges_       =   NULL;
    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;
    bool          over_sized;

    printf("Iteration entered\n");fflush(stdout);
    while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
    {
        Total_Length             = 0;
        data_slice->wait_counter = 0;
        tretval                  = cudaSuccess;
        if (num_gpus>1 && enactor_stats[0].iteration>0)
        {
            frontier_attribute[0].queue_reset  = true;
            frontier_attribute[0].queue_offset = 0;
            for (i=1; i<num_gpus; i++)
            {
                frontier_attribute[i].selector     = frontier_attribute[0].selector;
                frontier_attribute[i].advance_type = frontier_attribute[0].advance_type;
                frontier_attribute[i].queue_offset = 0;
                frontier_attribute[i].queue_reset  = true;
                frontier_attribute[i].queue_index  = frontier_attribute[0].queue_index;
                frontier_attribute[i].current_label= frontier_attribute[0].current_label;
                enactor_stats     [i].iteration    = enactor_stats     [0].iteration;
            }
        } else {
            frontier_attribute[0].queue_offset = 0;
            frontier_attribute[0].queue_reset  = true;
        }
        for (peer=0; peer<num_gpus; peer++)
        {
            stages [peer         ] = 0   ; 
            stages [peer+num_gpus] = 0   ;
            to_show[peer         ] = true; 
            to_show[peer+num_gpus] = true;
            for (i=0; i<data_slice->num_stages; i++)
                data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
        }
        
        while (data_slice->wait_counter < num_gpus*2
           && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
        {
       }

        if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
        {
            for (peer_=0;peer_<num_gpus*2;peer_++)
                data_slice->wait_marker[peer_]=0;
            wait_count=0;
            while (wait_count<num_gpus*2-1 &&
                !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
            {
                for (peer_=0;peer_<num_gpus*2;peer_++)
                {
                    if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                        continue;
                    tretval = cudaStreamQuery(streams[peer_]);
                    if (tretval == cudaSuccess)
                    {
                        data_slice->wait_marker[peer_]=1;
                        wait_count++;
                        continue;
                    } else if (tretval != cudaErrorNotReady)
                    {
                        enactor_stats[peer_%num_gpus].retval = tretval;
                        break;
                    }
                }
            }

            if (Enactor::DEBUG) {printf("%d\t %lld\t \t Subqueue finished. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
            grid_size = Total_Length/256+1;
            if (grid_size > 512) grid_size = 512;

            if (Enactor::SIZE_CHECK)
            {
                if (enactor_stats[0]. retval = 
                    Check_Size<true, SizeT, VertexId> ("total_queue", Total_Length, &data_slice->frontier_queues[0].keys[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;
                if (Problem::USE_DOUBLE_BUFFER)
                    if (enactor_stats[0].retval = 
                        Check_Size<true, SizeT, Value> ("total_queue", Total_Length, &data_slice->frontier_queues[0].values[frontier_attribute[0].selector], over_sized, thread_num, iteration, num_gpus, true)) break;

                offset=frontier_attribute[0].queue_length;
                for (peer_=1;peer_<num_gpus;peer_++)
                if (frontier_attribute[peer_].queue_length !=0) {
                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                        data_slice->frontier_queues[0    ].keys[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                        data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                        frontier_attribute[peer_].queue_length);
                    if (Problem::USE_DOUBLE_BUFFER)
                        util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                            data_slice->frontier_queues[0       ].values[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_   ].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                            frontier_attribute[peer_].queue_length);
                    offset+=frontier_attribute[peer_].queue_length;
                }
            }
            frontier_attribute[0].queue_length = Total_Length;
            if (!Enactor::SIZE_CHECK) frontier_attribute[0].selector = 0;
            frontier_queue_ = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpus == 1)?0:num_gpus]);
       }         
        Iteration::Iteration_Change(enactor_stats->iteration);
    }
}*/
    
template <
    typename AdvanceKernelPolicy, 
    typename FilterKernelPolicy, 
    typename Enactor
    /*bool     _HAS_SUBQ,
    bool     _HAS_FULLQ,
    bool     _BACKWARD,
    bool     _FORWARD,
    bool     _UPDATE_PREDECESSORS*/>
struct IterationBase
{
public:
    enum Status {
        New,
        Init,
        Running
    };

    typedef typename Enactor::SizeT      SizeT     ;   
    typedef typename Enactor::Value      Value     ;   
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef typename Enactor::GraphSlice GraphSlice;
    typedef typename Enactor::FrontierT  FrontierT ;
    typedef typename Enactor::FrontierA  FrontierA ;
    typedef typename Enactor::WorkProgress WorkProgress;
    typedef typename Enactor::EnactorStats EnactorStats;

    template <typename Type>
    using Array = typename Enactor::Array<Type>;

    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    //static const bool HAS_SUBQ   = _HAS_SUBQ;
    //static const bool HAS_FULLQ  = _HAS_FULLQ;
    //static const bool BACKWARD   = _BACKWARD;
    //static const bool FORWARD    = _FORWARD;
    //static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;

    bool              has_subq;
    bool              has_fullq;
    bool              backward;
    bool              forward;
    bool              update_predecessors;

    Status            status;
    int               num_gpus;
    int               thread_num;
    int               gpu_num;
    int               stream_num;
    int               num_streams;
    int               num_vertex_associates;
    int               num_value__associates;
    SizeT             num_elements;
    SizeT             request_length;
    FrontierT        *frontier_queue;
    Array<SizeT>     *scanned_edge;
    FrontierA        *frontier_attribute;
    EnactorStats     *enactor_stats;
    util::Array1D<SizeT, DataSlice> *data_slice;
    util::Array1D<SizeT, DataSlice> *data_slices;
    GraphSlice       *graph_slice;
    WorkProgress     *work_progress;
    ContextPtr        context;
    cudaStream_t      stream;
    SizeT            *t_out_length;

    int               grid_size;
    int               block_size;
    size_t            shared_size;
    VertexId         *d_keys_in;
    VertexId         *d_keys_out;
    size_t            array_size;
    char             *d_array;

    SizeT            *d_offsets;
    VertexId         *d_indices;
    VertexId         *d_in_key_queue;
    SizeT             max_in;
    SizeT             max_out;
    gunrock::oprtr::advance::TYPE advance_type;
    bool              express;

    IterationBase(
        int  _num_gpus, 
        int  _num_streams,
        bool _has_subq,
        bool _has_fullq,
        bool _backward,
        bool _forward,
        bool _update_predecessors) :
        status            (Status::New),
        has_subq          (_has_subq),
        has_fullq         (_has_fullq),
        backward          (_backward),
        forward           (_forward),
        update_predecessors(_update_predecessors),
        num_gpus          (_num_gpus),
        thread_num        (0   ),
        gpu_num           (0   ),
        stream_num        (0   ),
        num_streams       (_num_streams),
        num_elements      (0   ),
        num_vertex_associates(0),
        num_value__associates(0),
        request_length    (0   ),
        frontier_queue    (NULL),
        scanned_edge      (NULL),
        frontier_attribute(NULL),
        enactor_stats     (NULL),
        data_slice        (NULL),
        data_slices       (NULL),
        graph_slice       (NULL),
        work_progress     (NULL),
        stream            (0   ),
        grid_size         (0   ),
        block_size        (0   ),
        shared_size       (0   ),
        d_keys_in         (NULL),
        d_keys_out        (NULL),
        array_size        (0   ),
        d_array           (NULL),
        d_offsets         (NULL),
        d_indices         (NULL),
        d_in_key_queue    (NULL),
        max_in            (0   ),
        max_out           (0   ),
        express           (false)
    {
        t_out_length = new SizeT[num_streams];
    }

    virtual ~IterationBase()
    {
        frontier_queue     = NULL;
        scanned_edge       = NULL;
        frontier_attribute = NULL;
        enactor_stats      = NULL;
        data_slice         = NULL;
        data_slices        = NULL;
        graph_slice        = NULL;
        work_progress      = NULL;
        delete[] t_out_length; t_out_length = NULL;
        d_keys_in          = NULL;
        d_keys_out         = NULL;
        d_array            = NULL;
        d_offsets          = NULL;
        d_indices          = NULL;
        d_in_key_queue     = NULL;
    }

    virtual cudaError_t SubQueue_Gather () {return cudaSuccess;}

    virtual cudaError_t Compute_OutputLength() {return cudaSuccess;}

    virtual cudaError_t SubQueue_Core   () {return cudaSuccess;}

    virtual cudaError_t FullQueue_Gather() {return cudaSuccess;}

    virtual cudaError_t FullQueue_Core  () {return cudaSuccess;}

    virtual cudaError_t Expand_Incoming () {return cudaSuccess;}

    virtual bool        Stop_Condition  ()
    {
        return All_Done(enactor_stats,frontier_attribute,data_slice,num_gpus);
    }

    virtual cudaError_t Iteration_Change (long long &iterations)
    {
        iterations++;
        return cudaSuccess;
    }

    virtual cudaError_t Iteration_Update_Preds()
    {
        cudaError_t retval = cudaSuccess;

        if (num_elements == 0) return retval;
        //int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && update_predecessors && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                d_keys_in,
                data_slice[0] ->preds         .GetPointer(util::DEVICE),
                data_slice[0] ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                d_keys_in,
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice[0] ->temp_preds     .GetPointer(util::DEVICE),
                data_slice[0] ->preds          .GetPointer(util::DEVICE));//,
        }
        return retval;
    }

    virtual cudaError_t Check_Queue_Size()
    {
        cudaError_t retval     = cudaSuccess;
        bool        over_sized = false;
        int         selector   = frontier_attribute->selector;
        int         iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
        {
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, stream_num,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);
            fflush(stdout);
        }

        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", request_length, &frontier_queue->keys  [selector^1], 
            over_sized, thread_num, iteration, stream_num, false)) 
            return retval;
        if (retval = Check_Size<true, SizeT, VertexId > (
            "queue3", request_length, &frontier_queue->keys  [selector  ],
            over_sized, thread_num, iteration, stream_num, true )) 
            return retval;
        if (Problem::USE_DOUBLE_BUFFER)
        {
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", request_length, &frontier_queue->values[selector^1],
                over_sized, thread_num, iteration, stream_num, false)) 
                return retval;
            if (retval = Check_Size<true, SizeT, Value> (
                "queue3", request_length, &frontier_queue->values[selector  ], 
                over_sized, thread_num, iteration, stream_num, true )) 
                return retval;
        }
        return retval; 
    }

    virtual cudaError_t Make_Output()
    {
        // TODO: modify to fit new interfaces
        cudaError_t retval          = cudaSuccess;
        bool        over_sized      = false;
        bool        keys_over_sized = false;
        int         stream_num      = 0;
        int         t=0, i=0;
        size_t      offset          = 0;
        int         selector        = frontier_attribute->selector;
        int         block_size      = 256;
        int         grid_size       = num_elements / block_size;
        DataSlice*  data_slice      = data_slice + 0;

        if (num_gpus < 2) return retval;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size=512;

        for (stream_num = 0; stream_num < num_streams; stream_num++)
        {
            t_out_length[stream_num] = 0;
            data_slice->out_length[peer_] = 0;
        }
        if (num_elements ==0) return;
 
        over_sized = false;
        for (stream_num = 0; stream_num < num_streams; stream_num++)
        {
            if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
                "keys_marker", num_elements, &data_slice->keys_marker[peer_], 
                over_sized, thread_num, iteration, )) break;
            if (over_sized) data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
        }
        if (enactor_stats->retval) return;
        if (over_sized) data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        
        for (t=0; t<2; t++)
        {
            if (t==0 && !FORWARD) continue;
            if (t==1 && !BACKWARD) continue;

            if (BACKWARD && t==1) 
                Assign_Marker_Backward<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->backward_offset   .GetPointer(util::DEVICE),
                    graph_slice   ->backward_partition.GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Assign_Marker<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->partition_table   .GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));

            for (peer_=0;peer_<num_gpus;peer_++)
            {
                Scan<mgpu::MgpuScanTypeInc>(
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    num_elements,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    context[0]);
            }

            if (num_elements>0) for (peer_=0; peer_<num_gpus;peer_++)
            {
                cudaMemcpyAsync(&(t_out_length[peer_]),
                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                        + (num_elements -1),
                    sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            } else {
                for (peer_=0;peer_<num_gpus;peer_++)
                    t_out_length[peer_]=0;
            }
            if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

            keys_over_sized = true;
            for (peer_=0; peer_<num_gpus;peer_++)
            {
                if (enactor_stats->retval = 
                    Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId> (
                        "keys_out", 
                        data_slice->out_length[peer_] + t_out_length[peer_], 
                        peer_!=0 ? &data_slice->keys_out[peer_] : 
                                   &data_slice->frontier_queues[0].keys[selector^1], 
                        keys_over_sized, thread_num, enactor_stats[0].iteration, peer_), 
                        data_slice->out_length[peer_]==0? false: true) break;
                if (keys_over_sized) 
                    data_slice->keys_outs[peer_] = peer_==0 ? 
                        data_slice->frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) : 
                        data_slice->keys_out[peer_].GetPointer(util::DEVICE);
                if (peer_ == 0) continue;

                over_sized = false;
                for (i=0;i<NUM_VERTEX_ASSOCIATES;i++)
                {
                    if (enactor_stats[0].retval = 
                        Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId>(
                            "vertex_associate_outs", 
                            data_slice->out_length[peer_] + t_out_length[peer_], 
                            &data_slice->vertex_associate_out[peer_][i], 
                            over_sized, thread_num, enactor_stats->iteration, peer_),
                            data_slice->out_length[peer_]==0? false: true) break;
                    if (over_sized) data_slice->vertex_associate_outs[peer_][i] = data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

                over_sized = false;
                for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
                {
                    if (enactor_stats->retval = 
                        Check_Size<Enactor::SIZE_CHECK, SizeT, Value   >(
                            "value__associate_outs", 
                            data_slice->out_length[peer_] + t_out_length[peer_], 
                            &data_slice->value__associate_out[peer_][i], 
                            over_sized, thread_num, enactor_stats->iteration, peer_,
                            data_slice->out_length[peer_]==0? false: true)) break;
                    if (over_sized) data_slice->value__associate_outs[peer_][i] = data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized) data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            }
            if (enactor_stats->retval) break;
            if (keys_over_sized) data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

            offset = 0;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_markers         .GetPointer(util::HOST),
                      sizeof(SizeT*   ) * num_gpus);
            offset += sizeof(SizeT*   ) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> keys_outs            .GetPointer(util::HOST),
                      sizeof(VertexId*) * num_gpus);
            offset += sizeof(VertexId*) * num_gpus ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                      sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
            offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            memcpy(&(data_slice -> make_out_array[offset]),
                     data_slice -> value__associate_orgs.GetPointer(util::HOST),
                      sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
            offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                         data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            }
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                        data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            }
            memcpy(&(data_slice->make_out_array[offset]),
                     data_slice->out_length.GetPointer(util::HOST),
                      sizeof(SizeT) * num_gpus);
            offset += sizeof(SizeT) * num_gpus;
            data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, stream);

            if (BACKWARD && t==1) 
                Make_Out_Backward<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> backward_offset     .GetPointer(util::DEVICE),
                    graph_slice   -> backward_partition  .GetPointer(util::DEVICE),
                    graph_slice   -> backward_convertion .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Make_Out<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> partition_table     .GetPointer(util::DEVICE),
                    graph_slice   -> convertion_table    .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            for (peer_ = 0; peer_<num_gpus; peer_++)
                data_slice->out_length[peer_] += t_out_length[peer_];
        }
        if (enactor_stats->retval) return;                    
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
        frontier_attribute->selector^=1;
        if (t_out_length!=NULL) {delete[] t_out_length; t_out_length=NULL;}
   
        return retval;
    }

    void PrintMessage(const char* message, long long iteration = -1)
    {
        if (Enactor::DEBUG)
            util::cpu_mt::PrintMessage(message, thread_num,
            iteration, stream_num);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
