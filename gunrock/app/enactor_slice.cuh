// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_slice.cuh
 *
 * @brief Per GPU Enactor Slice
 */

#pragma once

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {

template <typename Enactor>
struct EnactorSlice
{
    typedef typename Enactor::SizeT           SizeT        ;
    typedef typename Enactor::VertexId        VertexId     ;
    typedef typename Enactor::Value           Value        ;
    typedef typename Enactor::CircularQueue   CircularQueue;
    template <typename Type>
    using Array = typename Enactor::Array<Type>;
    typedef typename Enactor::FrontierT       FrontierT    ;
    typedef typename Enactor::FrontierA       FrontierA    ;
    typedef typename Enactor::PRequest        PRequest     ;
    typedef typename Enactor::EnactorStats    EnactorStats ;
    typedef typename Enactor::ExpandIncomingHandle ExpandIncomingHandle;
    typedef typename Enactor::MakeOutHandle   MakeOutHandle;

    int num_gpus;
    int gpu_num;
    int gpu_idx;
    int num_input_streams;
    int num_outpu_streams;
    int num_subq__streams;
    int num_fullq_stream ;
    int num_split_streams;
    
    Array<VertexId*   >   vertex_associate_orgs   ; // Device pointers to original VertexId type associate values
    Array<Value*      >   value__associate_orgs   ; // Device pointers to original Value type associate values
    std::list<cudaEvent_t> empty_events;

    Array<cudaStream_t>   input_streams           ; // GPU streams
    int                   input_target_count      ;
    CircularQueue         input_queues         [2];
    Array<ExpandIncomingHandle> input_e_handles   ; // compressed data structure for expand_incoming kernel
    void                 *input_iteration_loops   ;
    void                 *input_thread_slice      ;
    SizeT                 input_min_length        ;
    SizeT                 input_max_length        ;

    Array<cudaStream_t>   outpu_streams           ; // GPU streams
    CircularQueue         outpu_queue             ;
    std::list<PRequest>   outpu_request_queue     ;
    std::mutex            outpu_request_mutex     ;
    void                 *outpu_iteration_loops   ;
    void                 *outpu_thread_slice      ;

    Array<cudaStream_t>   subq__streams           ; // GPU streams
    Array<ContextPtr  >   subq__contexts          ;
    CircularQueue         subq__queue             ;
    Array<int         >   subq__stages            ; // current stages of each streams
    Array<bool        >   subq__to_shows          ; // whether to show debug information for the streams
    Array<cudaEvent_t*>   subq__events         [4]; // GPU stream events arrays
    Array<bool*       >   subq__event_sets     [4]; // Whether the GPU stream events are set
    Array<int         >   subq__wait_markers      ; //
    int                   subq__wait_counter      ; 
    Array<SizeT       >  *subq__scanned_edges     ;
    Array<FrontierT   >   subq__frontiers         ; // frontier queues
    Array<FrontierA   >   subq__frontier_attributes;
    Array<EnactorStats>   subq__enactor_statses   ;
    Array<util::CtaWorkProgressLifetime> 
                          subq__work_progresses   ;
    void                 *subq__iteration_loops   ;
    SizeT                 subq__target_count   [2];
    void                 *subq__thread_slice      ;
    SizeT                 subq__min_length        ;
    SizeT                 subq__max_length        ;
    Array<SizeT>          subq__s_lengths         ;
    Array<SizeT>          subq__s_offsets         ;

    Array<cudaStream_t>   fullq_stream            ; // GPU streams
    Array<ContextPtr  >   fullq_context           ;
    CircularQueue         fullq_queue             ;
    Array<int>            fullq_stage             ;
    Array<cudaEvent_t*>   fullq_event          [4]; // GPU stream events arrays
    Array<bool*       >   fullq_event_set      [4];
    Array<bool        >   fullq_to_show           ; // whether to show debug information for the streams
    //Array<int         >  *fullq_wait_marker       ;
    Array<SizeT       >  *fullq_scanned_edge      ;
    Array<FrontierT   >   fullq_frontier          ; // frontier queues
    Array<FrontierA   >   fullq_frontier_attribute;
    Array<EnactorStats>   fullq_enactor_stats     ;
    Array<util::CtaWorkProgressLifetime>
                          fullq_work_progress     ;
    void                 *fullq_iteration_loop    ;
    SizeT                 fullq_target_count   [2];
    void                 *fullq_thread_slice      ;

    Array<cudaStream_t>   split_streams           ; // GPU streams
    Array<ContextPtr  >   split_contexts          ;
    Array<SizeT       >   split_lengths           ; // Number of outgoing vertices to peers  
    Array<SizeT       >  *split_markers           ; // Markers to separate vertices to peer GPUs
    Array<SizeT*      >   split_markerss          ;
    Array<MakeOutHandle>  split_m_handles         ; // compressed data structure for make_out kernel
    Array<cudaEvent_t >   split_events            ;
    void                 *split_iteration_loop    ;
    cudaEvent_t           split_wait_event        ;

    EnactorSlice() :
        num_gpus           (0   ),
        gpu_num            (0   ),
        gpu_idx            (0   ),
        num_input_streams  (0   ),
        num_outpu_streams  (0   ),
        num_subq__streams  (0   ),
        num_fullq_stream   (0   ),
        num_split_streams  (0   ),
        //input_e_arrays     (NULL),
        subq__scanned_edges(NULL),
        fullq_scanned_edge (NULL),
        split_markers      (NULL)
        //split_m_arrays     (NULL)
    {
        printf("EnactorSlice() begin.\n");fflush(stdout);

        vertex_associate_orgs .SetName("vertex_associate_orgs");
        value__associate_orgs .SetName("value__associate_orgs");
        input_streams         .SetName("input_streams"        );
        input_queues[0]       .SetName("input_queues[0]"      );
        input_queues[1]       .SetName("input_queues[1]"      );
        input_e_handles       .SetName("input_e_handles"      );
        outpu_streams         .SetName("outpu_streams"        );
        outpu_queue           .SetName("outpu_queue"          );

        subq__streams         .SetName("subq__streams"        );
        subq__contexts        .SetName("subq__contexts"       );
        subq__queue           .SetName("subq__queue"          );
        subq__stages          .SetName("subq__stages"         );
        subq__to_shows        .SetName("subq__to_shows"       );
        for (int i=0; i<4; i++)
        {
            subq__events[i]    .SetName("subq__events[]"      );
            subq__event_sets[i].SetName("subq__event_sets[]"  );
        }
        subq__wait_markers    .SetName("subq__wait_markers"   );
        //subq__scanned_edges   .SetName("subq__scanned_edges"  );
        subq__frontiers       .SetName("subq__frontiers"      );
        subq__frontier_attributes.SetName("sub__frontier_attribures");
        subq__enactor_statses .SetName("subq__encator_statses");
        subq__work_progresses .SetName("subq__work_progresses");

        fullq_stream          .SetName("fullq_stream"         );
        fullq_context         .SetName("fullq_context"        );
        fullq_queue           .SetName("fullq_queue"          );
        fullq_stage           .SetName("fullq_stage"          );
        fullq_to_show         .SetName("fullq_to_show"        );
        for (int i=0; i<4; i++)
        {
            fullq_event [i]    .SetName("fullq_event[]"       );
        }
        //fullq_wait_marker     .SetName("fullq_wait_marker"    );
        //fullq_scanned_edge    .SetName("fullq_scanned_edges"  );
        fullq_frontier        .SetName("fullq_frontier"       );
        fullq_frontier_attribute.SetName("fullq_frontier_attribute");
        fullq_enactor_stats   .SetName("fullq_encator_stats"  );
        fullq_work_progress   .SetName("fullq_work_progress"  );

        split_streams         .SetName("split_streams"        );
        split_contexts        .SetName("split_contexts"       );
        split_lengths         .SetName("split_lengths"        );
        split_m_handles       .SetName("split_m_handles"      );
        printf("EnactorSlice() end.\n");fflush(stdout);
    }

    virtual ~EnactorSlice()
    {
        printf("~EnactorSlice() begin.\n");fflush(stdout);
        Release();
        printf("~EnactorSlice() end.\n");fflush(stdout);
    }

    cudaError_t Init(
        int num_gpus          = 1,
        int gpu_num           = 0,
        int gpu_idx           = 0,
        int num_input_streams = 0,
        int num_outpu_streams = 0,
        int num_subq__streams = 0,
        int num_fullq_streams = 0,
        int num_split_streams = 0)
    {
        cudaError_t retval = cudaSuccess;
        printf("EnactorSlice::Init begin. gpu_num = %d\n", gpu_num);fflush(stdout);

        this->num_gpus = num_gpus;
        this->gpu_num  = gpu_num;
        this->gpu_idx  = gpu_idx;
        if (retval = util::SetDevice(gpu_idx)) return retval;

        if (num_gpus > 1)
        {
            if (retval = vertex_associate_orgs.Allocate(Enactor::NUM_VERTEX_ASSOCIATES)) 
                return retval;
            if (retval = value__associate_orgs.Allocate(Enactor::NUM_VALUE__ASSOCIATES))
                return retval;

            this->num_input_streams = num_input_streams;
            input_target_count = num_gpus - 1;
            if (num_input_streams != 0)
            {
                if (retval = input_streams  .Allocate(num_input_streams)) return retval;
                if (retval = input_e_handles.Init(num_input_streams,
                    util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
                for (int stream=0; stream<num_input_streams; stream++)
                {
                    if (retval = util::GRError(cudaStreamCreate(input_streams + stream),
                        "cudaStreamCreate failed", __FILE__, __LINE__)) return retval;
                }
            }
                
            this->num_outpu_streams = num_outpu_streams;
            if (num_outpu_streams != 0)
            {
                if (retval = outpu_streams.Allocate(num_outpu_streams)) return retval;
                for (int stream=0; stream<num_outpu_streams; stream++)
                {    
                    if (retval = util::GRError(cudaStreamCreate(outpu_streams + stream),
                        "cudaStreamCreate failed", __FILE__, __LINE__)) return retval;
                }
            }
        }

        this->num_subq__streams = num_subq__streams;
        if (num_subq__streams != 0)
        {
            subq__min_length = 1;
            subq__max_length = 32 * 1024 * 1024;
            if (retval = subq__streams     .Allocate(num_subq__streams)) return retval;
            if (retval = subq__contexts    .Allocate(num_subq__streams)) return retval;
            if (retval = subq__stages      .Allocate(num_subq__streams)) return retval;
            if (retval = subq__to_shows    .Allocate(num_subq__streams)) return retval;
            if (retval = subq__wait_markers.Allocate(num_subq__streams)) return retval;
            if (retval = subq__frontiers   .Allocate(num_subq__streams)) return retval;
            if (retval = subq__s_lengths   .Allocate(num_subq__streams)) return retval;
            if (retval = subq__s_offsets   .Allocate(num_subq__streams)) return retval;
            if (retval = subq__frontier_attributes.Init(num_subq__streams, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            if (retval = subq__enactor_statses    .Init(num_subq__streams, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            if (retval = subq__work_progresses    .Init(num_subq__streams, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            subq__scanned_edges = new Array<SizeT>[num_subq__streams];
            for (int stream=0; stream<num_subq__streams; stream++)
            {
                if (retval = util::GRError(cudaStreamCreate(subq__streams + stream),
                    "cudaStreamCreate failed", __FILE__, __LINE__)) return retval;
                subq__contexts[stream] = mgpu::CreateCudaDeviceAttachStream(gpu_idx, subq__streams[stream]);
                subq__scanned_edges[stream].SetName("subq__scanned_edges[]");
                subq__work_progresses[stream].Init();
                if (retval = subq__frontier_attributes[stream].output_length.Init(1, util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            }
            for (int i=0; i<4; i++)
            {
                if (retval = subq__events    [i].Allocate(num_subq__streams)) return retval;
                if (retval = subq__event_sets[i].Allocate(num_subq__streams)) return retval;
                for (int stream=0; stream<num_subq__streams; stream++)
                {
                    subq__events    [i][stream] = new cudaEvent_t[Enactor::NUM_STAGES];
                    subq__event_sets[i][stream] = new bool       [Enactor::NUM_STAGES];
                    for (int stage=0; stage<Enactor::NUM_STAGES; stage++)
                    {
                        if (retval = util::GRError(
                            cudaEventCreate(subq__events[i][stream] + stage),
                            "cudaEventCreate failed", __FILE__, __LINE__)) 
                            return retval;
                        //printf("subq__events[%d][%d][%d] created %d\n", i, stream, stage, subq__events[i][stream][stage]);
                    }
                }
            }
        }
         
        this->num_fullq_stream = num_fullq_stream;
        if (num_fullq_stream != 0)
        {   
            if (retval = fullq_stream      .Allocate(num_fullq_stream)) return retval;
            if (retval = fullq_context     .Allocate(num_fullq_stream)) return retval;
            if (retval = fullq_stage       .Allocate(num_fullq_stream)) return retval;
            if (retval = fullq_to_show     .Allocate(num_fullq_stream)) return retval;
            if (retval = fullq_frontier    .Allocate(num_fullq_stream)) return retval;
            //if (retval = fullq_s_length    .Allocate(num_fullq_stream)) return retval;
            //if (retval = fullq_s_offset    .Allocate(num_fullq_stream)) return retval;
            if (retval = fullq_frontier_attribute.Init(num_fullq_stream, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            if (retval = fullq_enactor_stats     .Init(num_fullq_stream, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            if (retval = fullq_work_progress     .Init(num_fullq_stream, 
                util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            fullq_scanned_edge = new Array<SizeT>[num_fullq_stream];
            for (int stream=0; stream<num_fullq_stream; stream++)
            {
                if (retval = util::GRError(cudaStreamCreate(fullq_stream + stream),
                    "cudaStreamCreate failed", __FILE__, __LINE__)) return retval;
                fullq_context[stream] = mgpu::CreateCudaDeviceAttachStream(gpu_idx, fullq_stream[stream]);
                fullq_scanned_edge[stream].SetName("fullq_scanned_edge[]");
                fullq_work_progress[stream].Init();
                if (retval = fullq_frontier_attribute[stream].output_length.Allocate(1, 
                    util::HOST | util::DEVICE)) return retval;
            }
            for (int i=0; i<4; i++)
            {
                if (retval = fullq_event    [i].Allocate(num_fullq_stream)) return retval;
                //if (retval = fullq_event_set[i].Allocate(num_fullq_streams)) return retval;
                for (int stream=0; stream<num_fullq_stream; stream++)
                {
                    fullq_event    [i][stream] = new cudaEvent_t[Enactor::NUM_STAGES];
                    //fullq_event_set[i][stream] = new bool       [Enactor::NUM_STAGES];
                    for (int stage=0; stage<Enactor::NUM_STAGES; stage++)
                    {
                        if (retval = util::GRError(cudaEventCreate(fullq_event[i][stream] + stage)),
                            "cudaEventCreate failed", __FILE__, __LINE__) return retval;
                    }
                }
            }
        }

        if (num_gpus > 1)
        {
            this->num_split_streams = num_split_streams;
            if (num_split_streams != 0)
            {
                if (retval = split_streams .Allocate(num_split_streams)) return retval;
                if (retval = split_contexts.Allocate(num_split_streams)) return retval;
                split_markers  = new Array<SizeT>[num_split_streams];
                if (retval = split_m_handles.Init(num_split_streams,
                    util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
                for (int stream=0; stream<num_split_streams; stream++)
                {
                    if (retval = util::GRError(cudaStreamCreate(split_streams + stream),
                        "cudaStreamCreate failed", __FILE__, __LINE__)) return retval;
                    split_contexts[stream] = mgpu::CreateCudaDeviceAttachStream(gpu_idx, split_streams[stream]);
                    split_markers [stream].SetName("split_marker[]");
                }
                if (retval = util::GRError(cudaEventCreate(&split_wait_event),
                    "cudaEventCreate failed", __FILE__, __LINE__))
                    return retval;
                if (retval = split_lengths.Init(num_split_streams,
                    util::HOST | util::DEVICE, true, cudaHostAllocMapped | cudaHostAllocPortable)) return retval;
            }
        }

        printf("EnactorSlice::Init end. gpu_num = %d\n", gpu_num);fflush(stdout);
        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        printf("EnactorSlice::Release begin. gpu_num = %d\n", gpu_num);fflush(stdout);
        
        if (retval = util::SetDevice(gpu_idx)) return retval;
        if (retval = vertex_associate_orgs.Release()) return retval;
        if (retval = value__associate_orgs.Release()) return retval;

        if (num_input_streams != 0)
        {
            for (int stream=0; stream<num_input_streams; stream++)
            {
                if (retval = util::GRError(cudaStreamDestroy(input_streams[stream]),
                    "cudaStreamDestroy failed", __FILE__, __LINE__)) return retval;
            }   
            if (retval = input_streams  .Release()) return retval;
            if (retval = input_queues[0].Release()) return retval;
            if (retval = input_queues[1].Release()) return retval;
            if (retval = input_e_handles.Release()) return retval;
            num_input_streams = 0;
        }

        if (num_outpu_streams != 0)
        { 
            for (int stream=0; stream<num_outpu_streams; stream++)
            {
                if (retval = util::GRError(cudaStreamDestroy(outpu_streams[stream]),
                    "cudaStreamDestroy failed", __FILE__, __LINE__)) return retval;
            }
            if (retval = outpu_streams.Release()) return retval;
            if (retval = outpu_queue  .Release()) return retval;
            num_outpu_streams = 0;
        }

        if (num_subq__streams != 0)
        {
            for (int stream=0; stream<num_subq__streams; stream++)
            {
                if (retval = util::GRError(cudaStreamDestroy(subq__streams[stream]),
                    "cudaStreamDestroy failed", __FILE__, __LINE__)) return retval;
                if (retval = subq__frontier_attributes[stream].Release())
                    return retval;
                if (retval = subq__frontiers          [stream].Release())
                    return retval;
                if (retval = subq__scanned_edges      [stream].Release())
                    return retval;
                if (retval = subq__enactor_statses    [stream].Release())
                    return retval;
                if (retval = subq__work_progresses    [stream].Release())
                    return retval;
            }
            for (int i=0; i<4; i++)
            {
                for (int stream=0; stream<num_subq__streams; stream++)
                {
                    for (int stage=0; stage<Enactor::NUM_STAGES; stage++)
                    {
                        if (retval = util::GRError(cudaEventDestroy(subq__events[i][stream][stage]),
                            "cudaEventDestroy failed", __FILE__, __LINE__)) return retval;
                    }
                    delete[] subq__events    [i][stream]; subq__events    [i][stream] = NULL;
                    delete[] subq__event_sets[i][stream]; subq__event_sets[i][stream] = NULL;
                }
                if (retval = subq__events    [i].Release()) return retval;
                if (retval = subq__event_sets[i].Release()) return retval;
            }
            if (retval = subq__streams            .Release()) return retval;
            if (retval = subq__contexts           .Release()) return retval;
            if (retval = subq__queue              .Release()) return retval;
            if (retval = subq__stages             .Release()) return retval;
            if (retval = subq__to_shows           .Release()) return retval;
            if (retval = subq__wait_markers       .Release()) return retval;
            if (retval = subq__frontiers          .Release()) return retval;
            if (retval = subq__frontier_attributes.Release()) return retval;
            if (retval = subq__enactor_statses    .Release()) return retval;
            if (retval = subq__work_progresses    .Release()) return retval;
            if (retval = subq__s_lengths          .Release()) return retval;
            if (retval = subq__s_offsets          .Release()) return retval;
            delete[] subq__scanned_edges; subq__scanned_edges = NULL;
            num_subq__streams = 0;
        }

        if (num_fullq_stream != 0)
        {
            for (int stream=0; stream<num_fullq_stream; stream++)
            {
                if (retval = util::GRError(cudaStreamDestroy(fullq_stream[stream]),
                    "cudaStreamDestroy failed", __FILE__, __LINE__)) return retval;
                if (retval = fullq_frontier_attribute[stream].Release())
                    return retval;
                if (retval = fullq_frontier          [stream].Release())
                    return retval;
                if (retval = fullq_scanned_edge      [stream].Release())
                    return retval;
                if (retval = fullq_enactor_stats     [stream].Release())
                    return retval;
                if (retval = fullq_work_progress     [stream].Release())
                    return retval;
            }
            for (int i=0; i<4; i++)
            {
                for (int stream=0; stream<num_fullq_stream; stream++)
                {
                    for (int stage=0; stage<Enactor::NUM_STAGES; stage++)
                    {
                        if (retval = util::GRError(cudaEventDestroy(fullq_event[i][stream][stage]),
                            "cudaEventDestroy failed", __FILE__, __LINE__)) return retval;
                    }
                    delete[] fullq_event    [i][stream]; fullq_event    [i][stream] = NULL;
                }
                if (retval = fullq_event     [i].Release()) return retval;
            }
            if (retval = fullq_stream             .Release()) return retval;
            if (retval = fullq_context            .Release()) return retval;
            if (retval = fullq_queue              .Release()) return retval;
            if (retval = fullq_stage              .Release()) return retval;
            if (retval = fullq_to_show            .Release()) return retval;
            if (retval = fullq_frontier           .Release()) return retval;
            if (retval = fullq_frontier_attribute .Release()) return retval;
            if (retval = fullq_enactor_stats      .Release()) return retval;
            if (retval = fullq_work_progress      .Release()) return retval;
            delete[] fullq_scanned_edge; fullq_scanned_edge = NULL;
            num_fullq_stream = 0;
        }

        if (num_split_streams != 0)
        {
            for (int stream=0; stream<num_split_streams; stream++)
            {
                if (retval = util::GRError(cudaStreamDestroy(split_streams[stream]),
                    "cudaStreamDestory failed", __FILE__, __LINE__)) return retval;
                if (retval = split_markers [stream].Release()) return retval;
            }
            if (retval = split_streams .Release()) return retval;
            if (retval = split_contexts.Release()) return retval;
            if (retval = split_lengths .Release()) return retval;
            if (retval = split_m_handles.Release()) return retval;
            delete[] split_markers ; split_markers  = NULL;
            num_split_streams = 0;
        }

        printf("EnactorSlice::Release end. gpu_num = %d\n", gpu_num);fflush(stdout);
        return retval;
    }

    cudaError_t Reset(
        FrontierType frontier_type,
        GraphSlice<SizeT, VertexId, Value>
                    *graph_slice,
        bool   use_double_buffer = false,
        SizeT *num_in_nodes  = NULL,
        SizeT *num_out_nodes = NULL,
        double subq__factor  = 1.0,
        double subq__factor0 = 1.0,
        double subq__factor1 = 1.0,
        double fullq_factor  = 1.0,
        double fullq_factor0 = 1.0,
        double fullq_factor1 = 1.0,
        double input_factor  = 1.0,
        double outpu_factor  = 1.0,
        double split_factor  = 1.0,
        double temp_factor   = 0.1)
    {
        cudaError_t retval = cudaSuccess;

        printf("EnactorSlice::Reset begin. gpu_num = %d\n", gpu_num);fflush(stdout);
        if (subq__factor  < 0) subq__factor  = 1.0;
        if (subq__factor0 < 0) subq__factor0 = 1.0;
        if (subq__factor1 < 0) subq__factor1 = 1.0;
        if (fullq_factor  < 0) fullq_factor  = 1.0;
        if (fullq_factor0 < 0) fullq_factor0 = 1.0;
        if (fullq_factor1 < 0) fullq_factor1 = 1.0;
        if (input_factor  < 0) input_factor  = 1.0;
        if (outpu_factor  < 0) outpu_factor  = 1.0;
        if (split_factor  < 0) split_factor  = 1.0;
        if (temp_factor   < 0) temp_factor   = 0.1;

        if (num_input_streams != 0)
        {
            SizeT total_in_nodes = 0;
            for (int gpu=0; gpu<num_gpus; gpu++)
                total_in_nodes += num_in_nodes[gpu];
            SizeT target_capacity = total_in_nodes * input_factor;
            for (int i=0; i<2; i++)
            {
                if (retval = input_queues[i].Init(target_capacity, util::DEVICE, 10,
                    Enactor::NUM_VERTEX_ASSOCIATES, Enactor::NUM_VERTEX_ASSOCIATES,
                    temp_factor * target_capacity)) return retval;
                if (retval = input_queues[i].Reset()) return retval;
            }
        }

        if (num_outpu_streams != 0)
        {
            SizeT total_out_nodes = 0;
            for (int gpu=0; gpu<num_gpus; gpu++)
                total_out_nodes += num_out_nodes[gpu];
            SizeT target_capacity = total_out_nodes * outpu_factor;
            if (retval = outpu_queue.Init(target_capacity, util::DEVICE, 10,
                0, 0,
                temp_factor * target_capacity)) return retval;
            if (retval = outpu_queue.Reset()) return retval;
        }

        if (num_subq__streams != 0)
        {
            subq__wait_counter = 0;
            subq__target_count[0] = 1; //util::MaxValue<SizeT>();
            subq__target_count[1] = util::MaxValue<SizeT>();

            SizeT target_capacity = graph_slice->nodes * subq__factor;
            if (retval = subq__queue.Init(target_capacity, util::DEVICE, 10,
                0, 0,
                temp_factor * target_capacity)) return retval;
            if (retval = subq__queue.Reset()) return retval;

            SizeT frontier_sizes[2] = {0, 0};
            SizeT max_elements = 0;
            for (int i=0; i<2; i++)
            {
                double queue_sizing = (i==0)? subq__factor0 : subq__factor1;
                queue_sizing *= subq__factor;
                switch (frontier_type) {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    frontier_sizes[0] = graph_slice->nodes * queue_sizing +2;
                    frontier_sizes[1] = frontier_sizes[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    frontier_sizes[0] = graph_slice->edges * queue_sizing +2; 
                    frontier_sizes[1] = frontier_sizes[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    frontier_sizes[0] = graph_slice->nodes * queue_sizing +2;
                    frontier_sizes[1] = graph_slice->edges * queue_sizing +2; 
                    break;
                }
                for (int stream=0; stream<num_subq__streams; stream++)
                {
                    printf("frontier_sizes[%d] = %d\n", i, frontier_sizes[i]);
                    fflush(stdout);
                    if (retval = subq__frontiers[stream].keys[i].Allocate(
                        frontier_sizes[i], util::DEVICE)) return retval;
                    if (use_double_buffer) {
                        if (retval = subq__frontiers[stream].values[i].Allocate(
                            frontier_sizes[i], util::DEVICE)) return retval;
                    }
                }
                if (frontier_sizes[0] > max_elements) max_elements = frontier_sizes[0];
                if (frontier_sizes[1] > max_elements) max_elements = frontier_sizes[1];
            }
            for (int stream=0; stream<num_subq__streams; stream++)
            {
                subq__to_shows[stream] = false;
                subq__stages[stream] = 0;
                if (retval = subq__scanned_edges[stream].Allocate(max_elements, util::DEVICE))
                    return retval;
                for (int i=0; i<4; i++)
                for (int stage = 0; stage < Enactor::NUM_STAGES; stage++)
                {
                    subq__event_sets[i][stream][stage] = false;
                }
                if (retval = subq__enactor_statses[stream].Reset()) return retval; 
            }
        }

        if (num_fullq_stream != 0)
        {
            fullq_target_count[0] = util::MaxValue<SizeT>();
            fullq_target_count[1] = util::MaxValue<SizeT>();

            SizeT target_capacity = graph_slice->nodes * fullq_factor;
            if (retval = fullq_queue.Init(target_capacity, util::DEVICE, 10,
                0, 0,
                temp_factor * target_capacity)) return retval;
            if (retval = fullq_queue.Reset()) return retval;

            SizeT frontier_sizes[2] = {0, 0};
            SizeT max_elements = 0;
            for (int i=0; i<2; i++)
            {
                double queue_sizing = (i==0)? fullq_factor0 : fullq_factor1;
                queue_sizing *= fullq_factor;
                switch (frontier_type) {
                case VERTEX_FRONTIERS :
                    // O(n) ping-pong global vertex frontiers
                    frontier_sizes[0] = graph_slice->nodes * queue_sizing +2;
                    frontier_sizes[1] = frontier_sizes[0];
                    break;

                case EDGE_FRONTIERS :
                    // O(m) ping-pong global edge frontiers
                    frontier_sizes[0] = graph_slice->edges * queue_sizing +2; 
                    frontier_sizes[1] = frontier_sizes[0];
                    break;

                case MIXED_FRONTIERS :
                    // O(n) global vertex frontier, O(m) global edge frontier
                    frontier_sizes[0] = graph_slice->nodes * queue_sizing +2;
                    frontier_sizes[1] = graph_slice->edges * queue_sizing +2; 
                    break;
                }
                for (int stream=0; stream<num_fullq_stream; stream++)
                {
                    if (retval = fullq_frontier[stream].keys[i].Allocate(
                        frontier_sizes[i], util::DEVICE)) return retval;
                    if (use_double_buffer) {
                        if (retval = fullq_frontier[stream].values[i].Allocate(
                            frontier_sizes[i], util::DEVICE)) return retval;
                    }
                }
                if (frontier_sizes[0] > max_elements) max_elements = frontier_sizes[0];
                if (frontier_sizes[1] > max_elements) max_elements = frontier_sizes[1];
            }
            for (int stream=0; stream<num_fullq_stream; stream++)
            {
                fullq_stage[stream] = 0;
                if (retval = fullq_scanned_edge[stream].Allocate(max_elements, util::DEVICE))
                    return retval;
            }
        }

        if (num_split_streams != 0)
        {
            SizeT target_capacity = graph_slice->nodes * split_factor;
            for (int stream=0; stream<num_split_streams; stream++)
            {
                if (retval = split_markers[stream].Allocate(target_capacity, util::DEVICE))
                    return retval;
            }
        }
        printf("EnactorSlice::Reset end. gpu_num = %d\n", gpu_num);fflush(stdout);
        return retval;
    }
}; // end of EnactorSlice

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
