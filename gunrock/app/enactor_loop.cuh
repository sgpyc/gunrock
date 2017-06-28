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

#include <gunrock/app/enactor_kernel.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/util/latency_utils.cuh>
//#include <gunrock/util/test_utils.h>
#include <gunrock/util/mpi_utils.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

enum Loop_Type
{
    Host_Send, // Dummy
    Host_Recv,
    Local_Send,
    Local_Recv,
    Remote_Send,
    Remote_Recv,
};

/*
 * @brief Iteration loop.
 *
 * @tparam Enactor
 * @tparam Functor
 * @tparam Iteration
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] thread_data
 */
template <
    typename Enactor,
    typename Functor,
    typename Iteration,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    //typedef typename Iteration::Enactor   Enactor   ;
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value>  GraphSliceT;
    typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_local_gpus       =   problem     -> num_local_gpus;
    int           num_total_gpus       =   problem     -> num_total_gpus;
    //int           thread_num           =   thread_data -> thread_num;
    int           local_rank           =   problem     -> mpi_rank;
    int           gpu_rank_local       =   thread_data -> thread_num;
    int           gpu_rank             =   gpu_rank_local
                                         + local_rank * num_local_gpus;
    DataSlice    *data_slice           =   problem     -> data_slices        [gpu_rank_local].GetPointer(util::HOST);
    DataSlice    *d_data_slice         =   problem     -> data_slices        [gpu_rank_local].GetPointer(util::DEVICE);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSliceT  *graph_slice          =   problem     -> graph_slices       [gpu_rank_local] ;
    //GraphSliceT  **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attributes  = &(enactor     -> frontier_attribute [gpu_rank_local * num_total_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);
    EnactorStats<SizeT>
                 *enactor_statses      = &(enactor     -> enactor_stats      [gpu_rank_local * num_total_gpus]);
    EnactorStats<SizeT>
                 *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);
    util::CtaWorkProgressLifetime<SizeT>
                 *work_progresses      = &(enactor     -> work_progress      [gpu_rank_local * num_total_gpus]);
    ContextPtr   *contexts             =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_shows             =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    //SizeT         Total_Length         =   0;
    //SizeT         received_length      =   0;
    //cudaError_t   tretval              =   cudaSuccess;
    //int           grid_size            =   0;
    std::string   mssg                 =   "";
    //Loop_Stage    pre_stage            =   Pre_SendRecv;
    //Loop_Stage    current_stage        =   Pre_SendRecv;
    //Loop_Stage    next_stage           =   Pre_SendRecv;
    //Loop_Type     loop_type            =   Host_Recv;
    //size_t        offset               =   0;
    //int           iteration            =   0;
    //int           selector             =   0;
    //Frontier     *frontier_queue_      =   NULL;
    //FrontierAttribute<SizeT>
    //             *frontier_attribute_  =   NULL;
    //EnactorStats<SizeT>
    //             *enactor_stats_       =   NULL;
    //util::CtaWorkProgressLifetime<SizeT>
    //             *work_progress_       =   NULL;
    //util::Array1D<SizeT, SizeT>
    //             *scanned_edges_       =   NULL;
    //int           peer_gpu_rank, peer_gpu_rank_;
    //int           gpu_rank_remote_, i, iteration_, wait_count, peer_rank;
    //bool          over_sized;
    SizeT         communicate_latency  =   enactor -> communicate_latency;
    float         communicate_multipy  =   enactor -> communicate_multipy;
    SizeT         expand_latency       =   enactor -> expand_latency;
    SizeT         subqueue_latency     =   enactor -> subqueue_latency;
    SizeT         fullqueue_latency    =   enactor -> fullqueue_latency;
    SizeT         makeout_latency      =   enactor -> makeout_latency;

#ifdef ENABLE_PERFORMANCE_PROFILING
    util::CpuTimer      cpu_timer;
    std::vector<double> &iter_full_queue_time =
        enactor -> iter_full_queue_time        [gpu_rank_local].back();
    std::vector<double> &iter_sub_queue_time =
        enactor -> iter_sub_queue_time         [gpu_rank_local].back();
    std::vector<double> &iter_total_time =
        enactor -> iter_total_time             [gpu_rank_local].back();
    std::vector<SizeT>  &iter_full_queue_nodes_queued =
        enactor -> iter_full_queue_nodes_queued[gpu_rank_local].back();
    std::vector<SizeT>  &iter_full_queue_edges_queued =
        enactor -> iter_full_queue_edges_queued[gpu_rank_local].back();

    cpu_timer.Start();
    double iter_start_time = cpu_timer.MillisSinceStart();
    double iter_stop_time = 0;
    double subqueue_finish_time = 0;

    SizeT  h_edges_queued       [num_total_gpus];
    SizeT  h_nodes_queued       [num_total_gpus];
    SizeT  previous_edges_queued[num_total_gpus];
    SizeT  previous_nodes_queued[num_total_gpus];
    SizeT  h_full_queue_edges_queued = 0;
    SizeT  h_full_queue_nodes_queued = 0;
    SizeT  previous_full_queue_edges_queued = 0;
    SizeT  previous_full_queue_nodes_queued = 0;

    for (int peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus;
        peer_gpu_rank_++)
    {
        h_edges_queued       [peer_gpu_rank_] = 0;
        h_nodes_queued       [peer_gpu_rank_] = 0;
        previous_nodes_queued[peer_gpu_rank_] = 0;
        previous_edges_queued[peer_gpu_rank_] = 0;
    }
#endif

    if (enactor -> debug)
        util::PrintMsg("Iteration entered");

    while (!Iteration::Stop_Condition(
        s_enactor_stats, s_frontier_attribute, s_data_slice,
        num_local_gpus, num_total_gpus, gpu_rank_local))
    {
        SizeT Total_Length             = 0;
        SizeT received_length          = frontier_attributes[0].queue_length;
        data_slice->wait_counter = 0;
        cudaError_t tretval            = cudaSuccess;
        //auto &iteration          = enactor_stats[0].iteration;
        if (num_total_gpus > 1 && enactor_statses[0].iteration != 0)
        {
            frontier_attributes[0].queue_reset  = true;
            frontier_attributes[0].queue_offset = 0;
            for (int peer_gpu_rank_ = 1; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_++)
            {
                auto &frontier_attribute = frontier_attributes[peer_gpu_rank_];
                frontier_attribute.selector     = frontier_attributes[0].selector;
                frontier_attribute.advance_type = frontier_attributes[0].advance_type;
                frontier_attribute.queue_offset = 0;
                frontier_attribute.queue_reset  = true;
                frontier_attribute.queue_index  = frontier_attributes[0].queue_index;
                frontier_attribute.current_label= frontier_attributes[0].current_label;
                enactor_statses[peer_gpu_rank_].iteration = enactor_statses[0].iteration;
            }
        } else {
            frontier_attributes[0].queue_offset = 0;
            frontier_attributes[0].queue_reset  = true;
        }

        if (num_total_gpus > 1)
        {
            if (enactor -> problem -> unified_receive)
            {
                //printf("%d, %d : start_received_length = %d\n",
                //    thread_num, enactor_stats[0].iteration, received_length);
                data_slice -> in_length_out[0] = received_length;
                data_slice -> in_length_out.Move(
                    util::HOST, util::DEVICE, 1, 0, streams[0]);
                if (enactor_statses[0].retval = util::GRError(
                    cudaStreamSynchronize(streams[0]),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                break;
            }

            if (enactor_statses[0].iteration != 0)
            {
                auto t = enactor_statses[0].iteration % 2;
                //util::PrintMsg("Iteration = " + std::to_string(enactor_statses[0].iteration) +
                //    ", t = " + std::to_string(t));
                //int t = 0;
                for (int peer_gpu_rank = 0; peer_gpu_rank < num_total_gpus;
                    peer_gpu_rank++)
                {
                    int peer_rank = peer_gpu_rank / num_local_gpus;
                    int peer_gpu_rank_remote = peer_gpu_rank % num_local_gpus;
                    if (peer_rank == local_rank) continue;
                    int send_tag_base = util::Get_Send_Tag(peer_gpu_rank_remote,
                        num_total_gpus, gpu_rank, t * 8);
                    int peer_gpu_rank_ = (peer_gpu_rank < gpu_rank) ?
                        peer_gpu_rank + 1 : peer_gpu_rank;

                    util::Mpi_Isend_Bulk(&(data_slice -> out_length[peer_gpu_rank_]),
                        1, peer_rank, send_tag_base,
                        MPI_COMM_WORLD, data_slice -> send_requests[peer_gpu_rank_]);
                    //util::PrintMsg(std::string(" -> rank ") + std::to_string(peer_rank) +
                    //    ", tag " + std::to_string(send_tag_base) +
                    //    ", out_length[" + std::to_string(peer_gpu_rank_) + "] = " + 
                    //    std::to_string(data_slice -> out_length[peer_gpu_rank_]));

                    int recv_tag_base = util::Get_Recv_Tag(gpu_rank_local,
                        num_total_gpus, peer_gpu_rank, t * 8);
                    util::Mpi_Irecv_Bulk(&(data_slice -> in_length[t][peer_gpu_rank_]),
                        1, peer_rank, recv_tag_base,
                        MPI_COMM_WORLD, data_slice -> recv_requests[peer_gpu_rank_]);
                    data_slice -> in_iteration [t][peer_gpu_rank_] = enactor_statses[0].iteration;
                }
            }
        } else data_slice -> in_length_out[0] = received_length;

        for (int peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_++)
        {
            stages  [peer_gpu_rank_                 ] = Loop_Stage::Pre_SendRecv;
            stages  [peer_gpu_rank_ + num_total_gpus] = Loop_Stage::Pre_SendRecv;
            to_shows[peer_gpu_rank_                 ] = true;
            to_shows[peer_gpu_rank_ + num_total_gpus] = true;
            for (int stage = 0; stage < data_slice->num_stages; stage++)
                data_slice->events_set[enactor_statses[0].iteration%4][peer_gpu_rank_][stage]
                    = false;
        }
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labels",
        //    data_slice -> labels.GetPointer(util::DEVICE),
        //    graph_slice -> nodes, thread_num, iteration, -1, streams[0]);

        while (data_slice->wait_counter < num_total_gpus * 2
           && (!Iteration::Stop_Condition(
            s_enactor_stats, s_frontier_attribute, s_data_slice,
            num_local_gpus, num_total_gpus, gpu_rank_local)))
        {
            //util::cpu_mt::PrintCPUArray<int, int>("stages", stages,
            //    num_total_gpus * 2, thread_num, iteration);
            for (int peer_gpu_pipe = 0; peer_gpu_pipe < num_total_gpus * 2; peer_gpu_pipe++)
            {
                auto peer_gpu_rank_      = (peer_gpu_pipe % num_total_gpus);
                auto peer_gpu_rank       = peer_gpu_rank_ <= gpu_rank ?
                                             peer_gpu_rank_-1 : peer_gpu_rank_;
                auto peer_rank           = peer_gpu_rank / num_local_gpus;
                auto gpu_rank_remote_    = peer_gpu_rank <  gpu_rank ?
                                             gpu_rank  : gpu_rank + 1;
                auto iteration           = enactor_statses[peer_gpu_rank_].iteration;
                auto iteration_mod_4     = iteration % 4;
                auto iteration_mod_2     = iteration % 2;
                auto current_stage       = stages[peer_gpu_pipe];
                auto next_stage          = current_stage;
                auto &frontier_queue     = data_slice->frontier_queues[peer_gpu_rank_];
                auto &scanned_edges      = data_slice->scanned_edges  [peer_gpu_rank_];
                auto &frontier_attribute = frontier_attributes        [peer_gpu_rank_];
                auto &enactor_stats      = enactor_statses            [peer_gpu_rank_];
                auto &work_progress      = work_progresses            [peer_gpu_rank_];
                auto selector            = frontier_attribute.selector;
                auto &queue_length       = frontier_attribute.queue_length;
                auto &retval             = enactor_stats     .retval;
                auto &to_show            = to_shows[peer_gpu_pipe];
                auto &stream             = streams [peer_gpu_pipe];
                auto &context            = contexts[peer_gpu_rank_];
                auto  loop_type          = Host_Recv;
                //int send_tag_base        = gpu_rank      * 16 + iteration_mod_2 * 8;
                int recv_tag_base        = util::Get_Recv_Tag(gpu_rank_local, 
                    num_total_gpus, peer_gpu_rank, iteration_mod_2 * 8);

                if (peer_gpu_pipe == 0)
                    loop_type = Host_Recv;
                else if (peer_gpu_pipe > 0 && peer_gpu_pipe < num_total_gpus)
                {
                    if (local_rank == peer_rank)
                        loop_type = Local_Recv;
                    else loop_type = Remote_Recv;
                } else if (peer_gpu_pipe == num_total_gpus)
                    loop_type = Host_Send;
                else if (peer_gpu_pipe > num_total_gpus
                    && peer_gpu_pipe < num_total_gpus * 2)
                {
                    if (local_rank == peer_rank)
                        loop_type = Local_Send;
                    else loop_type = Remote_Send;
                }

                if (enactor -> debug && to_show)
                {
                    mssg=" "; mssg[0]='0' + data_slice->wait_counter;
                    ShowDebugInfo<Problem>(
                        gpu_rank_local, peer_gpu_pipe,
                        &frontier_attribute, &enactor_stats,
                        data_slice, graph_slice,
                        &work_progress, mssg, stream);
                }
                to_show = true;

                switch (current_stage)
                {
                case Pre_SendRecv:
                    switch (loop_type)
                    {
                    case Host_Send: // Dummy, not in use
                        next_stage = End; break;

                    case Host_Recv:
                        if (queue_length != 0)
                            next_stage = Recv;
                        else {
                            // empty local queue
                            //Set_Record(data_slice, iteration,
                            //    peer_gpu_pipe, 3, streams[peer_gpu_pipe]);
                            next_stage = End;
                        }
                        break;

                    case Local_Recv:
                        next_stage = Recv; break;

                    case Local_Send:
                        if (communicate_latency != 0)
                            util::latency::Insert_Latency(communicate_latency,
                            data_slice -> out_length[peer_gpu_rank_], stream,
                            data_slice -> latency_data.GetPointer(util::DEVICE));
                        next_stage = Send; break;

                    case Remote_Recv:
                        // Wait recv length
                        if (retval = util::Mpi_Test(
                            data_slice -> recv_requests[peer_gpu_rank_]))
                            break;
                        if (!data_slice -> recv_requests[peer_gpu_rank_].empty())
                        {
                            to_show = false; break;
                        }

                        //util::PrintMsg(std::string(" rank ") + std::to_string(peer_rank) +
                        //    ", tag " + std::to_string(recv_tag_base) +
                        //    " ->, in_length[" + std::to_string(peer_gpu_rank_) + "] = " + 
                        //    std::to_string(data_slice -> in_length[iteration_mod_2][peer_gpu_rank_]));

                        if (data_slice -> keys_out[peer_gpu_rank_]
                            .GetPointer(util::DEVICE) != NULL)
                        {
                            retval = data_slice -> temp_keys_in[peer_gpu_rank_].EnsureSize(
                                data_slice -> in_length[iteration_mod_2][peer_gpu_rank_]);
                            if (retval) break;
                            retval = util::Mpi_Irecv_Bulk(
                                data_slice -> temp_keys_in[peer_gpu_rank_].GetPointer(util::HOST),
                                data_slice -> in_length[iteration_mod_2][peer_gpu_rank_],
                                peer_rank, recv_tag_base + 1, MPI_COMM_WORLD,
                                data_slice -> recv_requests[peer_gpu_rank_]);
                            if (retval) break;
                        }

                        retval = data_slice -> temp_vertex_associate_in[peer_gpu_rank_].EnsureSize(
                            data_slice -> in_length[iteration_mod_2][peer_gpu_rank_] * NUM_VERTEX_ASSOCIATES);
                        if (retval) break;
                        retval = util::Mpi_Irecv_Bulk(
                            data_slice -> temp_vertex_associate_in[peer_gpu_rank_].GetPointer(util::HOST),
                            data_slice -> in_length[iteration_mod_2][peer_gpu_rank_] * NUM_VERTEX_ASSOCIATES,
                            peer_rank, recv_tag_base + 2, MPI_COMM_WORLD,
                            data_slice -> recv_requests[peer_gpu_rank_]);
                        if (retval) break;

                        retval = data_slice -> temp_value__associate_in[peer_gpu_rank_].EnsureSize(
                            data_slice -> in_length[iteration_mod_2][peer_gpu_rank_] * NUM_VALUE__ASSOCIATES);
                        if (retval) break;
                        retval = util::Mpi_Irecv_Bulk(
                            data_slice -> temp_value__associate_in[peer_gpu_rank_].GetPointer(util::HOST),
                            data_slice -> in_length[iteration_mod_2][peer_gpu_rank_] * NUM_VALUE__ASSOCIATES,
                            peer_rank, recv_tag_base + 3, MPI_COMM_WORLD,
                            data_slice -> recv_requests[peer_gpu_rank_]);
                        if (retval) break;
                        next_stage = Recv; break;

                    case Remote_Send:
                        // Wait pervious send to complete
                        if (retval = util::Mpi_Test(
                            data_slice -> send_requests[peer_gpu_rank_]))
                            break;
                        if (!data_slice -> send_requests[peer_gpu_rank_].empty())
                        {
                            to_show = false;
                            break;
                        }

                        if (communicate_latency != 0)
                            util::latency::Insert_Latency(communicate_latency,
                            data_slice -> out_length[peer_gpu_rank_], stream,
                            data_slice -> latency_data.GetPointer(util::DEVICE));

                        Pre_Send_Remote <Enactor, GraphSliceT, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            enactor, gpu_rank_local, peer_gpu_rank,
                            data_slice->out_length[peer_gpu_rank_],
                            &enactor_stats, s_data_slice,
                            stream, communicate_multipy);

                        Set_Record(data_slice, iteration,
                            peer_gpu_pipe, Pre_SendRecv, stream);
                        next_stage = Send; break;
                    }
                    break;

                case Recv: // Recv
                    if (loop_type == Host_Recv)
                    {
                        if (Iteration::HAS_SUBQ)
                            next_stage = Comp_OutLength;
                        else {
                            Set_Record(data_slice, iteration,
                                peer_gpu_pipe, SubQ_Core, stream);
                            next_stage = Copy;
                        }
                        break;
                    }

                    //wait and expand incoming
                    if (loop_type == Local_Recv)
                    {
                        if (!(s_data_slice[peer_gpu_rank % num_local_gpus] ->
                            events_set[iteration_mod_4][gpu_rank_remote_ + num_total_gpus][Send]))
                        {
                            to_show = false; break;
                        }
                    } else {
                        if (retval = util::Mpi_Test(
                            data_slice -> recv_requests[peer_gpu_rank_]))
                            break;
                        if (!data_slice -> recv_requests[peer_gpu_rank_].empty())
                        {
                            to_show = false; break;
                        }
                    }

                    queue_length =
                        data_slice -> in_length[iteration_mod_2][peer_gpu_rank_];
#ifdef ENABLE_PERFORMANCE_PROFILING
                    enactor_stats. iter_in_length.back().push_back(
                        data_slice -> in_length[iteration_mod_2][peer_gpu_rank_]);
#endif
                    if (loop_type == Local_Recv)
                    {
                        if (queue_length != 0)
                        {
                            if (retval = util::GRError(cudaStreamWaitEvent(stream,
                                s_data_slice[peer_gpu_rank % num_local_gpus] ->
                                events[iteration_mod_4][gpu_rank_remote_ + num_total_gpus][Send], 0),
                                "cudaStreamWaitEvent failed",
                                __FILE__, __LINE__))
                                break;
                        }
                        s_data_slice[peer_gpu_rank % num_local_gpus] ->
                            events_set[iteration_mod_4][gpu_rank_remote_ + num_total_gpus][Send]
                            = false;
                    }
                    data_slice -> in_length[iteration_mod_2][peer_gpu_rank_] = 0;

                    if (queue_length == 0)
                    {
                        //Set_Record(data_slice, iteration,
                        //    peer_gpu_pipe, 3, streams[peer_gpu_pipe]);
                        //printf(" %d\t %d\t %d\t Expand and subQ skipped\n",
                        //    thread_num, iteration, peer_gpu_pipe);
                        next_stage = End; break;
                    }

                    if (expand_latency != 0)
                        util::latency::Insert_Latency(expand_latency,
                        frontier_attribute.queue_length, stream,
                        data_slice -> latency_data.GetPointer(util::DEVICE));

                    if (loop_type == Remote_Recv)
                    {
                        // Memcpy -> GPU
                        if (data_slice -> keys_out[peer_gpu_rank_]
                            .GetPointer(util::DEVICE) != NULL)
                        if (retval = util::GRError(cudaMemcpyAsync(
                            data_slice -> keys_in[iteration_mod_2][peer_gpu_rank_]
                                .GetPointer(util::DEVICE),
                            data_slice -> temp_keys_in[peer_gpu_rank_]
                                .GetPointer(util::HOST),
                            queue_length * sizeof(VertexId),
                            cudaMemcpyHostToDevice, stream),
                            "cudaMemcpyAsync keys_in H2D failed", __FILE__, __LINE__))
                            break;

                        if (retval = util::GRError(cudaMemcpyAsync(
                            data_slice -> vertex_associate_in[iteration_mod_2][peer_gpu_rank_]
                                .GetPointer(util::DEVICE),
                            data_slice -> temp_vertex_associate_in[peer_gpu_rank_]
                                .GetPointer(util::HOST),
                            queue_length * sizeof(VertexId) * NUM_VERTEX_ASSOCIATES,
                            cudaMemcpyHostToDevice, stream),
                            "cudaMemcpyAsync vertex_associate_in H2D failed", __FILE__, __LINE__))
                            break;

                        if (retval = util::GRError(cudaMemcpyAsync(
                            data_slice -> value__associate_in[iteration_mod_2][peer_gpu_rank_]
                                .GetPointer(util::DEVICE),
                            data_slice -> temp_value__associate_in[peer_gpu_rank_]
                                .GetPointer(util::HOST),
                            queue_length * sizeof(Value) * NUM_VALUE__ASSOCIATES,
                            cudaMemcpyHostToDevice, stream),
                            "cudaMemcpyAsync temp_value__associate_in[" +
                            std::to_string(peer_gpu_rank_) + "] -> value__associate_in[" +
                            std::to_string(iteration_mod_2) + "][" +
                            std::to_string(peer_gpu_rank_) + "] H2D failed", __FILE__, __LINE__))
                            break;
                    }

                    Iteration::template Expand_Incoming
                        <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                        enactor, stream,
                        data_slice -> in_iteration       [iteration_mod_2][peer_gpu_rank_],
                        peer_gpu_rank_,
                        //data_slice -> in_length          [iteration%2],
                        received_length,
                        queue_length,
                        data_slice -> in_length_out,
                        data_slice -> keys_in            [iteration_mod_2][peer_gpu_rank_],
                        data_slice -> vertex_associate_in[iteration_mod_2][peer_gpu_rank_],
                        data_slice -> value__associate_in[iteration_mod_2][peer_gpu_rank_],
                        (enactor -> problem -> unified_receive) ?
                            data_slice -> frontier_queues[0].keys[frontier_attributes[0].selector]
                            : frontier_queue.keys[selector^1],
                        data_slice -> vertex_associate_orgs,
                        data_slice -> value__associate_orgs,
                        data_slice,
                        &enactor_stats);
                    //printf("%d, Expand, selector = %d, keys = %p\n",
                    //    thread_num, selector^1,
                    //    frontier_queue_ -> keys[selector^1].GetPointer(util::DEVICE));

                    frontier_attribute.selector^=1;
                    frontier_attribute.queue_index++;
                    if (!Iteration::HAS_SUBQ) {
                        if (enactor -> problem -> unified_receive)
                        {
                            //Set_Record(data_slice, iteration,
                            //    peer_gpu_pipe, 3, streams[peer_gpu_pipe]);
                            next_stage = End;
                        } else {
                            Set_Record(data_slice, iteration,
                                peer_gpu_pipe, SubQ_Core, stream);
                            next_stage = Copy;
                        }
                    } else {
                        Set_Record(data_slice, iteration,
                            peer_gpu_pipe, Recv, stream);
                        next_stage = Comp_OutLength;
                    }
                    break;

                case Send:
                    if (iteration == 0)
                    {  // first iteration, nothing to send
                        Set_Record(data_slice, iteration,
                            peer_gpu_pipe, Send, stream);
                        next_stage = End; break;
                    }

                    //Send to Neighbor
                    if (loop_type == Local_Send)
                        Send_Local <Enactor, GraphSliceT, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            enactor, gpu_rank_local, peer_gpu_rank,
                            data_slice->out_length[peer_gpu_rank_],
                            &enactor_stats, s_data_slice, stream,
                            communicate_multipy);

                    else if (loop_type == Remote_Send)
                        Send_Remote <Enactor, GraphSliceT, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            enactor, gpu_rank_local, peer_gpu_rank,
                            data_slice->out_length[peer_gpu_rank_],
                            &enactor_stats, s_data_slice, stream,
                            communicate_multipy);

                    Set_Record(data_slice, iteration,
                        peer_gpu_pipe, Send, stream);
                    next_stage = End;
                    break;

                case Comp_OutLength: //Comp Length
                    if (peer_gpu_rank_ != 0)
                    {
                        if (retval = Check_Record(
                            data_slice, iteration, peer_gpu_pipe,
                            Recv, Comp_OutLength, to_show)) break;
                        if (to_show == false) break;
                        queue_length = data_slice -> in_length_out[peer_gpu_rank_];
                    }

                    if (retval = Iteration::Compute_OutputLength(
                        enactor, &frontier_attribute,
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                        graph_slice    ->column_offsets  .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices     .GetPointer(util::DEVICE),
                        frontier_queue.keys[selector]    .GetPointer(util::DEVICE),
                        &scanned_edges,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context[0], stream,
                        gunrock::oprtr::advance::V2V, true, false, false)) break;

                    if (enactor -> size_check ||
                        (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_FORWARD &&
                         Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_BACKWARD))
                    {
                        Set_Record(data_slice, iteration,
                            peer_gpu_pipe, Comp_OutLength, stream);
                    }
                    next_stage = SubQ_Core;
                    break;

                case SubQ_Core: //SubQueue Core
                    if (enactor -> size_check ||
                        (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_FORWARD &&
                         Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_BACKWARD))
                    {
                        if (retval = Check_Record (
                            data_slice, iteration, peer_gpu_pipe,
                            Comp_OutLength, SubQ_Core, to_show)) break;
                        if (to_show == false) break;
                        /*if (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            == oprtr::advance::TWC_FORWARD ||
                            Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            == oprtr::advance::TWC_BACKWARD)
                        {
                            frontier_attribute_->output_length[0] *= 1.1;
                        }*/

                        if (enactor -> size_check)
                        Iteration::Check_Queue_Size(
                            enactor, gpu_rank_local, peer_gpu_rank_,
                            frontier_attribute.output_length[0] + 2,
                            &frontier_queue, &frontier_attribute,
                            &enactor_stats, graph_slice);
                        if (retval) break;
                    }
                    if (subqueue_latency != 0)
                        util::latency::Insert_Latency(subqueue_latency,
                        frontier_attribute.queue_length, stream,
                        data_slice -> latency_data.GetPointer(util::DEVICE));

                    Iteration::SubQueue_Core(
                        enactor, gpu_rank_local, peer_gpu_rank_,
                        &frontier_queue, &scanned_edges,
                        &frontier_attribute, &enactor_stats,
                        data_slice, d_data_slice,
                        graph_slice, &work_progress,
                        context, stream);
#ifdef ENABLE_PERFORMANCE_PROFILING
                    h_nodes_queued[peer_gpu_rank_] = enactor_stats.nodes_queued[0];
                    h_edges_queued[peer_gpu_rank_] = enactor_stats.edges_queued[0];
                    enactor_stats.nodes_queued.Move(
                        util::DEVICE, util::HOST, 1, 0, stream);
                    enactor_stats.edges_queued.Move(
                        util::DEVICE, util::HOST, 1, 0, stream);
#endif

                    if (num_total_gpus > 1)
                    {
                        Set_Record(data_slice, iteration,
                            peer_gpu_pipe, SubQ_Core, stream);
                        next_stage = Copy;
                    } else {
                        //Set_Record(data_slice, iteration, peer_gpu_pipe, 3, streams[peer_gpu_pipe]);
                        next_stage = End;
                    }
                    break;

                case Copy: //Copy
                    //if (Iteration::HAS_SUBQ || peer_ != 0)
                    {
                        if (retval = Check_Record(
                            data_slice, iteration, peer_gpu_pipe,
                            SubQ_Core, Copy, to_show))
                            break;
                        if (to_show == false) break;
                    }

                    //printf("size_check = %s\n", enactor -> size_check ? "true" : "false");
                    //fflush(stdout);
                    if (!Iteration::HAS_SUBQ && peer_gpu_rank_ > 0)
                    {
                        queue_length = data_slice -> in_length_out[peer_gpu_rank_];
                    }
                    if (!enactor -> size_check &&
                        (enactor -> debug || num_total_gpus > 1))
                    {
                        bool over_sized = false;
                        if (Iteration::HAS_SUBQ)
                        {
                            if (retval = Check_Size<SizeT, VertexId> (false, "queue3",
                                    frontier_attribute.output_length[0]+2,
                                    &frontier_queue.keys  [selector^1],
                                    over_sized, gpu_rank_local, iteration, peer_gpu_rank_, false))
                                break;
                        }
                        if (queue_length ==0) break;

                        if (retval = Check_Size<SizeT, VertexId> (false, "total_queue",
                                Total_Length + frontier_attribute.queue_length,
                                &data_slice->frontier_queues[num_total_gpus].keys[0],
                                over_sized, gpu_rank_local, iteration, peer_gpu_rank_, false))
                            break;

                        util::MemsetCopyVectorKernel<<<256,256, 0, stream>>>(
                            data_slice->frontier_queues[num_total_gpus].keys[0]
                                .GetPointer(util::DEVICE) + Total_Length,
                            frontier_queue.keys[selector].GetPointer(util::DEVICE),
                            queue_length);
                        if (problem -> use_double_buffer)
                            util::MemsetCopyVectorKernel<<<256,256,0,stream>>>(
                                data_slice->frontier_queues[num_total_gpus].values[0]
                                    .GetPointer(util::DEVICE) + Total_Length,
                                frontier_queue. values[selector].GetPointer(util::DEVICE),
                                queue_length);
                    }

                    Total_Length += queue_length;
                    //Set_Record(data_slice, iteration, peer_gpu_pipe, 3, streams[peer_gpu_pipe]);
                    next_stage = End;
                    break;

                case End: //End
                    data_slice -> wait_counter++;
                    to_show = false;
                    next_stage = Finished;
                    break;
                default:
                    //stages[peer_gpu_pipe]--;
                    to_show = false;
                } // end of switchs

                if (enactor -> debug && !retval)
                {
                    retval = util::GRError(
                        std::string("Stage ") + std::to_string(current_stage)
                        + " @ gpu " + std::to_string(gpu_rank)
                        + ", peer_ " + std::to_string(peer_gpu_pipe)
                        + "failed", __FILE__, __LINE__);
                    if (retval) break;
                }
                stages[peer_gpu_pipe] = next_stage;
                //stages[peer_gpu_pipe]++;
                if (retval) break;
            } // end of for peer_gpu_pipe
        } // end of while

        if (!Iteration::Stop_Condition(
            s_enactor_stats, s_frontier_attribute, s_data_slice,
            num_local_gpus, num_total_gpus, gpu_rank_local))
        {
            for (int peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus;
                peer_gpu_rank_++)
                data_slice->wait_marker[peer_gpu_rank_]=0;
            int wait_count = 0;
            while (wait_count < num_total_gpus &&
                !Iteration::Stop_Condition(
                s_enactor_stats, s_frontier_attribute, s_data_slice,
                num_local_gpus, num_total_gpus, gpu_rank_local))
            {
                for (int peer_gpu_rank_=0; peer_gpu_rank_<num_total_gpus;
                    peer_gpu_rank_++)
                {
                    if (peer_gpu_rank_== num_total_gpus ||
                        data_slice->wait_marker[peer_gpu_rank_]!=0)
                        continue;
                    cudaError_t tretval = cudaStreamQuery(streams[peer_gpu_rank_]);
                    if (tretval == cudaSuccess)
                    {
                        data_slice->wait_marker[peer_gpu_rank_]=1;
                        wait_count++;
                        continue;
                    } else if (tretval != cudaErrorNotReady)
                    {
                        enactor_statses[peer_gpu_rank_].retval = tretval;
                        break;
                    }
                }
            }

            if (enactor -> problem -> unified_receive)
            {
                Total_Length = data_slice -> in_length_out[0];
            } else if (num_total_gpus == 1)
                Total_Length = frontier_attributes[0].queue_length;
#ifdef ENABLE_PERFORMANCE_PROFILING
            subqueue_finish_time = cpu_timer.MillisSinceStart();
            iter_sub_queue_time.push_back(subqueue_finish_time - iter_start_time);
            if (Iteration::HAS_SUBQ)
            for (int peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_ ++)
            {
                enactor_stats        [peer_gpu_rank_].iter_nodes_queued.back().push_back(
                    h_nodes_queued   [peer_gpu_rank_]
                    + enactor_stats  [peer_gpu_rank_].nodes_queued[0]
                    - previous_nodes_queued[peer_gpu_rank_]);
                previous_nodes_queued[peer_gpu_rank_]
                    = h_nodes_queued [peer_gpu_rank_]
                    + enactor_stats  [peer_gpu_rank_].nodes_queued[0];
                enactor_stats        [peer_gpu_rank_].nodes_queued[0]
                    = h_nodes_queued [peer_gpu_rank_];

                enactor_stats        [peer_gpu_rank_].iter_edges_queued.back().push_back(
                    h_edges_queued   [peer_gpu_rank_]
                    + enactor_stats  [peer_gpu_rank_].edges_queued[0]
                    - previous_edges_queued[peer_gpu_rank_]);
                previous_edges_queued[peer_gpu_rank_]
                    = h_edges_queued [peer_gpu_rank_]
                    + enactor_stats  [peer_gpu_rank_].edges_queued[0];
                enactor_stats        [peer_gpu_rank_].edges_queued[0]
                    = h_edges_queued [peer_gpu_rank_];
            }
#endif
            if (enactor -> debug)
            {
                util::PrintMsg(std::string("GPU ") + std::to_string(gpu_rank)
                    + "\t " + std::to_string(enactor_statses[0].iteration)
                    + "\t \t Subqueue finished. Total_Length = "
                    + std::to_string(Total_Length));
            }

            int grid_size = Total_Length/256+1;
            if (grid_size > 512) grid_size = 512;

            if (enactor -> size_check && !enactor -> problem -> unified_receive)
            {
                bool over_sized = false;
                if (enactor_statses[0]. retval =
                    Check_Size</*true,*/ SizeT, VertexId> (
                        true, "total_queue", Total_Length,
                        &data_slice->frontier_queues[0].keys[frontier_attributes[0].selector],
                        over_sized, gpu_rank_local, enactor_statses[0].iteration, num_total_gpus, true))
                    break;
                if (problem -> use_double_buffer)
                    if (enactor_statses[0].retval =
                        Check_Size</*true,*/ SizeT, Value> (
                            true, "total_queue", Total_Length,
                            &data_slice->frontier_queues[0].values[frontier_attributes[0].selector],
                            over_sized, gpu_rank_local, enactor_statses[0].iteration, num_total_gpus, true))
                        break;

                auto offset = frontier_attributes[0].queue_length;
                for (int peer_gpu_rank_ = 1; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_++)
                if (frontier_attributes[peer_gpu_rank_].queue_length !=0)
                {
                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                        data_slice->frontier_queues[0    ]
                            .keys[frontier_attributes[0    ].selector]
                            .GetPointer(util::DEVICE) + offset,
                        data_slice->frontier_queues[peer_gpu_rank_]
                            .keys[frontier_attributes[peer_gpu_rank_].selector]
                            .GetPointer(util::DEVICE),
                        frontier_attributes[peer_gpu_rank_].queue_length);

                    if (problem -> use_double_buffer)
                        util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                            data_slice->frontier_queues[0    ]
                                .values[frontier_attributes[0    ].selector]
                                .GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_gpu_rank_]
                                .values[frontier_attributes[peer_gpu_rank_].selector]
                                .GetPointer(util::DEVICE),
                            frontier_attributes[peer_gpu_rank_].queue_length);
                    offset+=frontier_attributes[peer_gpu_rank_].queue_length;
                }
            }
            frontier_attributes[0].queue_length = Total_Length;
            if (! enactor -> size_check) frontier_attributes[0].selector = 0;
            auto frontier_queue_ = &(data_slice->frontier_queues
                [(enactor -> size_check || num_total_gpus == 1) ? 0 : num_total_gpus]);
            if (Iteration::HAS_FULLQ)
            {
                int peer_gpu_rank_  = 0;
                frontier_queue_     = &(data_slice->frontier_queues
                    [(enactor -> size_check || num_total_gpus==1) ? 0 : num_total_gpus]);
                auto &scanned_edges = data_slice->scanned_edges
                    [(enactor -> size_check || num_total_gpus==1) ? 0 : num_total_gpus];
                auto &frontier_attribute = frontier_attributes[peer_gpu_rank_];
                auto &enactor_stats      = enactor_statses[peer_gpu_rank_];
                auto &work_progress      = work_progresses[peer_gpu_rank_];
                auto &iteration          = enactor_stats.iteration;
                auto &stream             = streams        [peer_gpu_rank_];
                auto &context            = contexts       [peer_gpu_rank_];
                auto &retval             = enactor_stats.retval;

                frontier_attribute.queue_offset = 0;
                frontier_attribute.queue_reset  = true;
                if (!enactor -> size_check)
                    frontier_attribute.selector     = 0;

                Iteration::FullQueue_Gather(
                    enactor, gpu_rank_local, peer_gpu_rank_,
                    frontier_queue_, &scanned_edges,
                    &frontier_attribute, &enactor_stats,
                    data_slice, d_data_slice,
                    graph_slice, &work_progress,
                    context,stream);
                auto selector = frontier_attribute.selector;
                if (retval) break;

                if (frontier_attribute.queue_length !=0)
                {
                    if (enactor -> debug) {
                        mssg = "";
                        ShowDebugInfo<Problem>(
                            gpu_rank_local, peer_gpu_rank_,
                            &frontier_attribute, &enactor_stats,
                            data_slice, graph_slice,
                            &work_progress, mssg,
                            streams[peer_gpu_rank_]);
                    }

                    retval = Iteration::Compute_OutputLength(
                        enactor, &frontier_attribute,
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                        graph_slice    ->column_offsets  .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices     .GetPointer(util::DEVICE),
                        frontier_queue_->keys[selector]    .GetPointer(util::DEVICE),
                        &scanned_edges,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context[0], stream,
                        gunrock::oprtr::advance::V2V, true, false, false);
                    if (retval) break;

                    //frontier_attribute_->output_length.Move(
                    //    util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                    if (enactor -> size_check)
                    {
                        tretval = cudaStreamSynchronize(stream);
                        if (tretval != cudaSuccess)
                        {
                            retval = tretval;break;
                        }

                        Iteration::Check_Queue_Size(
                            enactor, gpu_rank_local, peer_gpu_rank_,
                            frontier_attribute.output_length[0] + 2,
                            frontier_queue_, &frontier_attribute,
                            &enactor_stats, graph_slice);
                        if (retval) break;
                    }

                    if (fullqueue_latency != 0)
                        util::latency::Insert_Latency(fullqueue_latency,
                        frontier_attribute.queue_length, stream,
                        data_slice -> latency_data.GetPointer(util::DEVICE));

                    Iteration::FullQueue_Core(
                        enactor, gpu_rank_local, peer_gpu_rank_,
                        frontier_queue_, &scanned_edges,
                        &frontier_attribute, &enactor_stats,
                        data_slice, d_data_slice,
                        graph_slice, &work_progress,
                        context, stream);
                    if (retval) break;
#ifdef ENABLE_PERFORMANCE_PROFILING
                    h_full_queue_nodes_queued = enactor_stats.nodes_queued[0];
                    h_full_queue_edges_queued = enactor_stats.edges_queued[0];
                    enactor_stats.edges_queued.Move(util::DEVICE, util::HOST, 1, 0, stream);
                    enactor_stats.nodes_queued.Move(util::DEVICE, util::HOST, 1, 0, stream);
#endif
                    //if (retval = util::GRError(
                    //    cudaStreamSynchronize(streams),
                    //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    //    break;
                    tretval = cudaErrorNotReady;
                    while (tretval == cudaErrorNotReady)
                    {
                        tretval = cudaStreamQuery(stream);
                        if (tretval == cudaErrorNotReady)
                            sleep(0);
                    }
                    if (retval = util::GRError(tretval,
                        "FullQueue_Core failed.", __FILE__, __LINE__))
                        break;

#ifdef ENABLE_PERFORMANCE_PROFILING
                    iter_full_queue_nodes_queued.push_back(
                        h_full_queue_nodes_queued + enactor_stats.nodes_queued[0]
                        - previous_full_queue_nodes_queued);
                    previous_full_queue_nodes_queued = h_full_queue_nodes_queued
                        + enactor_stats.nodes_queued[0];
                    enactor_stats. nodes_queued[0] = h_full_queue_nodes_queued;

                    iter_full_queue_edges_queued.push_back(
                        h_full_queue_edges_queued + enactor_stats.edges_queued[0]
                        - previous_full_queue_edges_queued);
                    previous_full_queue_edges_queued = h_full_queue_edges_queued
                        + enactor_stats.edges_queued[0];
                    enactor_stats.edges_queued[0] = h_full_queue_edges_queued;
#endif
                    if (!enactor -> size_check)
                    {
                        bool over_sized = false;
                        if (retval = Check_Size<SizeT, VertexId> (false, "queue3",
                                frontier_attribute.output_length[0]+2,
                                &frontier_queue_->keys[selector^1],
                                over_sized, gpu_rank_local, iteration, peer_gpu_rank_, false))
                            break;
                    }
                    selector = frontier_attributes[peer_gpu_rank_].selector;
                    Total_Length = frontier_attributes[peer_gpu_rank_].queue_length;
                } else {
                    Total_Length = 0;
                    for (peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_++)
                        data_slice->out_length[peer_gpu_rank_]=0;
#ifdef ENABLE_PERFORMANCE_PROFILING
                    iter_full_queue_nodes_queued.push_back(0);
                    iter_full_queue_edges_queued.push_back(0);
#endif
                }
#ifdef ENABLE_PERFORMANCE_PROFILING
                iter_full_queue_time.push_back(
                    cpu_timer.MillisSinceStart() - subqueue_finish_time);
#endif
                if (enactor -> debug)
                {
                    util::PrintMsg(std::to_string(gpu_rank_local)
                        + "\t " + std::to_string(iteration)
                        + "\t \t Fullqueue finished. Total_Length = "
                        + std::to_string(Total_Length));
                }
                frontier_queue_ = &(data_slice->frontier_queues
                    [enactor -> size_check ? 0 : num_total_gpus]);
                if (num_total_gpus == 1)
                    data_slice->out_length[0] = Total_Length;
            }

            if (num_total_gpus > 1)
            {
                for (int peer_gpu_pipe = num_total_gpus + 1;
                    peer_gpu_pipe < num_total_gpus * 2; peer_gpu_pipe++)
                    data_slice -> wait_marker[peer_gpu_pipe] = 0;
                wait_count = 0;
                while (wait_count < num_total_gpus-1 &&
                    !Iteration::Stop_Condition(
                    s_enactor_stats, s_frontier_attribute, s_data_slice,
                    num_local_gpus, num_total_gpus, gpu_rank_local))
                {
                    for (int peer_gpu_pipe = num_total_gpus + 1;
                        peer_gpu_pipe < num_total_gpus * 2; peer_gpu_pipe++)
                    {
                        if (peer_gpu_pipe == num_total_gpus ||
                            data_slice -> wait_marker[peer_gpu_pipe]!=0)
                            continue;
                        tretval = cudaStreamQuery(streams[peer_gpu_pipe]);
                        if (tretval == cudaSuccess)
                        {
                            data_slice->wait_marker[peer_gpu_pipe]=1;
                            wait_count++;
                            continue;
                        } else if (tretval != cudaErrorNotReady)
                        {
                            enactor_statses[peer_gpu_pipe % num_total_gpus].retval
                                = tretval;
                            break;
                        }
                    }
                }

                Iteration::Iteration_Update_Preds(
                    enactor, graph_slice, data_slice,
                    &frontier_attributes[0],
                    &data_slice->frontier_queues[enactor -> size_check ? 0 : num_total_gpus],
                    Total_Length, streams[0]);

                if (makeout_latency != 0)
                    util::latency::Insert_Latency(makeout_latency,
                    Total_Length, streams[0],
                    data_slice -> latency_data.GetPointer(util::DEVICE));

                Iteration::template Make_Output <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                    enactor,
                    gpu_rank_local,
                    Total_Length,
                    num_total_gpus,
                    &data_slice->frontier_queues[enactor -> size_check ? 0 : num_total_gpus],
                    &data_slice->scanned_edges[0],
                    &frontier_attributes[0],
                    enactor_statses,
                    &problem->data_slices[gpu_rank_local],
                    graph_slice,
                    &work_progresses[0],
                    contexts[0],
                    streams[0]);
            }
            else
            {
                data_slice->out_length[0]= Total_Length;
            }

            for (int peer_gpu_rank_ = 0; peer_gpu_rank_ < num_total_gpus; peer_gpu_rank_++)
            {
                frontier_attributes[peer_gpu_rank_].queue_length
                    = data_slice->out_length[peer_gpu_rank_];
#ifdef ENABLE_PERFORMANCE_PROFILING
                //if (peer_ == 0)
                    enactor_statses[peer_gpu_rank_].iter_out_length.back().push_back(
                        data_slice -> out_length[peer_gpu_rank_]);
#endif
            }
        }

#ifdef ENABLE_PERFORMANCE_PROFILING
        iter_stop_time = cpu_timer.MillisSinceStart();
        iter_total_time.push_back(iter_stop_time - iter_start_time);
        iter_start_time = iter_stop_time;
#endif
        Iteration::Iteration_Change(enactor_statses->iteration);
    } // end of while iteration
}

/*
 * @brief IterationBase data structure.
 *
 * @tparam AdvanceKernelPolicy
 * @tparam FilterKernelPolicy
 * @tparam Enactor
 * @tparam _HAS_SUBQ
 * @tparam _HAS_FULLQ
 * @tparam _BACKWARD
 * @tparam _FORWARD
 * @tparam _UPDATE_PREDECESSORS
 */
template <
    typename _AdvanceKernelPolicy,
    typename _FilterKernelPolicy,
    typename _Enactor,
    bool     _HAS_SUBQ,
    bool     _HAS_FULLQ,
    bool     _BACKWARD,
    bool     _FORWARD,
    bool     _UPDATE_PREDECESSORS>
struct IterationBase
{
public:
    typedef _Enactor                     Enactor   ;
    typedef _AdvanceKernelPolicy AdvanceKernelPolicy;
    typedef _FilterKernelPolicy  FilterKernelPolicy;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value>
                                         GraphSliceT;
    typedef util::DoubleBuffer<VertexId, SizeT, Value>
                                         Frontier;
    //static const bool INSTRUMENT = Enactor::INSTRUMENT;
    //static const bool DEBUG      = Enactor::DEBUG;
    //static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = _HAS_SUBQ;
    static const bool HAS_FULLQ  = _HAS_FULLQ;
    static const bool BACKWARD   = _BACKWARD;
    static const bool FORWARD    = _FORWARD;
    static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;

    /*
     * @brief SubQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief SubQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief Stop_Condition check function.
     *
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] num_local_gpus Number of local GPUs
     * @param[in] num_total_gpus Total number of GPUs used.
     */
    static bool Stop_Condition(
        EnactorStats<SizeT>           *enactor_stats,
        FrontierAttribute<SizeT>      *frontier_attribute,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        int                            num_local_gpus,
        int                            num_total_gpus,
        int                            gpu_rank_local)
    {
        return All_Done(enactor_stats, frontier_attribute, data_slice,
            num_local_gpus, num_total_gpus);
    }

    /*
     * @brief Iteration_Change function.
     *
     * @param[in] iterations
     */
    static void Iteration_Change(long long &iterations)
    {
        iterations++;
    }

    /*
     * @brief Iteration_Update_Preds function.
     *
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] num_elements Number of elements.
     * @param[in] stream CUDA stream.
     */
    static void Iteration_Update_Preds(
        Enactor                       *enactor,
        GraphSliceT                   *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>      *frontier_attribute,
        Frontier                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {
        if (num_elements == 0) return;
        int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && UPDATE_PREDECESSORS && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                frontier_queue->keys[selector].GetPointer(util::DEVICE),
                data_slice    ->preds         .GetPointer(util::DEVICE),
                data_slice    ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                frontier_queue->keys[selector] .GetPointer(util::DEVICE),
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice    ->temp_preds     .GetPointer(util::DEVICE),
                data_slice    ->preds          .GetPointer(util::DEVICE));//,
        }
    }

    /*
     * @brief Check frontier queue size function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] request_length Request frontier queue length.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     */
    static void Check_Queue_Size(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        Frontier                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        GraphSliceT                   *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (enactor -> debug)
            util::PrintMsg(std::to_string(thread_num) + "\t " +
                std::to_string(iteration) + "\t " +
                std::to_string(peer_) + "\t queue_length = " +
                std::to_string(frontier_queue -> keys[selector^1].GetSize()) +
                ", output_length = " + std::to_string(request_length));

        if (enactor_stats->retval =
            Check_Size</*true,*/ SizeT, VertexId > (
                true, "queue3", request_length, &frontier_queue->keys  [selector^1],
                over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size</*true,*/ SizeT, VertexId > (
                true, "queue3", request_length, &frontier_queue->keys  [selector  ],
                over_sized, thread_num, iteration, peer_, true )) return;
        if (enactor -> problem -> use_double_buffer)
        {
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, Value> (
                    true, "queue3", request_length, &frontier_queue->values[selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, Value> (
                    true, "queue3", request_length, &frontier_queue->values[selector  ],
                    over_sized, thread_num, iteration, peer_, true )) return;
        }
    }

    /*
     * @brief Make_Output function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] thread_num Number of threads.
     * @param[in] num_elements
     * @param[in] num_total_gpus Number of GPUs used.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        Enactor                       *enactor,
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_total_gpus,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);
        if (num_total_gpus < 2) return;
        if (num_elements == 0)
        {
            for (int peer_ = 0; peer_ < num_total_gpus; peer_ ++)
            {
                data_slice -> out_length[peer_] = 0;
            }
            return;
        }
        bool over_sized = false, keys_over_sized = false;
        int selector = frontier_attribute->selector;
        //printf("%d Make_Output begin, num_elements = %d, size_check = %s\n",
        //    data_slice -> gpu_idx, num_elements,
        //    enactor->size_check ? "true" : "false");
        //fflush(stdout);
        SizeT size_multi = 0;
        if (FORWARD ) size_multi += 1;
        if (BACKWARD) size_multi += 1;

        int peer_ = 0;
        for (peer_ = 0; peer_ < num_total_gpus; peer_++)
        {
            if (enactor_stats -> retval =
                Check_Size<SizeT, VertexId> (
                    enactor -> size_check, "keys_out",
                    num_elements * size_multi,
                    (peer_ == 0) ?
                        &data_slice -> frontier_queues[0].keys[selector^1] :
                        &data_slice -> keys_out[peer_],
                    keys_over_sized, thread_num, enactor_stats[0].iteration,
                    peer_),
                    false)
                break;
            //if (keys_over_sized)
                data_slice->keys_outs[peer_] = (peer_==0) ?
                    data_slice -> frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) :
                    data_slice -> keys_out[peer_].GetPointer(util::DEVICE);
            if (peer_ == 0) continue;

            over_sized = false;
            //for (i = 0; i< NUM_VERTEX_ASSOCIATES; i++)
            //{
                if (enactor_stats[0].retval =
                    Check_Size <SizeT, VertexId>(
                        enactor -> size_check, "vertex_associate_outs",
                        num_elements * NUM_VERTEX_ASSOCIATES * size_multi,
                        &data_slice->vertex_associate_out[peer_],
                        over_sized, thread_num, enactor_stats->iteration, peer_),
                        false)
                    break;
                //if (over_sized)
                    data_slice->vertex_associate_outs[peer_] =
                        data_slice->vertex_associate_out[peer_].GetPointer(util::DEVICE);
            //}
            //if (enactor_stats->retval) break;
            //if (over_sized)
            //    data_slice->vertex_associate_outs[peer_].Move(
            //        util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

            over_sized = false;
            //for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
            //{
                if (enactor_stats->retval =
                    Check_Size<SizeT, Value   >(
                        enactor -> size_check, "value__associate_outs",
                        num_elements * NUM_VALUE__ASSOCIATES * size_multi,
                        &data_slice->value__associate_out[peer_],
                        over_sized, thread_num, enactor_stats->iteration, peer_,
                        false)) break;
                //if (over_sized)
                    data_slice->value__associate_outs[peer_] =
                        data_slice->value__associate_out[peer_].GetPointer(util::DEVICE);
            //}
            //if (enactor_stats->retval) break;
            //if (over_sized)
            //    data_slice->value__associate_outs[peer_].Move(
            //        util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            if (enactor -> problem -> skip_makeout_selection) break;
        }
        if (enactor_stats->retval) return;
        if (enactor -> problem -> skip_makeout_selection)
        {
            if (NUM_VALUE__ASSOCIATES == 0 && NUM_VERTEX_ASSOCIATES == 0)
            {
                util::MemsetCopyVectorKernel<<<120, 512, 0, stream>>>(
                    data_slice -> keys_out[1].GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    num_elements);
                for (int peer_=0; peer_<num_total_gpus; peer_++)
                    data_slice -> out_length[peer_] = num_elements;
                if (enactor_stats -> retval = util::GRError(
                    cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    return;
                return;
            } else {
                for (int peer_ = 2; peer_ < num_total_gpus; peer_++)
                {
                    data_slice -> keys_out[peer_].SetPointer(
                        data_slice -> keys_out[1].GetPointer(util::DEVICE),
                        data_slice -> keys_out[1].GetSize(), util::DEVICE);
                    data_slice -> keys_outs[peer_]
                        = data_slice -> keys_out[peer_].GetPointer(util::DEVICE);

                    data_slice -> vertex_associate_out[peer_].SetPointer(
                        data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
                        data_slice -> vertex_associate_out[1].GetSize(), util::DEVICE);
                    data_slice -> vertex_associate_outs[peer_]
                        = data_slice -> vertex_associate_out[peer_].GetPointer(util::DEVICE);

                    data_slice -> value__associate_out[peer_].SetPointer(
                        data_slice -> value__associate_out[1].GetPointer(util::DEVICE),
                        data_slice -> value__associate_out[1].GetSize(), util::DEVICE);
                    data_slice -> value__associate_outs[peer_]
                        = data_slice -> value__associate_out[peer_].GetPointer(util::DEVICE);
                }
            }
        }
        //printf("%d Make_Out 1\n", data_slice -> gpu_idx);
        //fflush(stdout);
        //if (keys_over_sized)
        data_slice -> keys_outs            .Move(util::HOST, util::DEVICE,
            num_total_gpus, 0, stream);
        data_slice -> vertex_associate_outs.Move(util::HOST, util::DEVICE,
            num_total_gpus, 0, stream);
        data_slice -> value__associate_outs.Move(util::HOST, util::DEVICE,
            num_total_gpus, 0, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PreMakeOut",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    num_elements, data_slice -> gpu_idx, enactor_stats -> iteration, -1, stream);
        int num_blocks = (num_elements >> (AdvanceKernelPolicy::LOG_THREADS)) + 1;
        if (num_blocks > 480) num_blocks = 480;
        //printf("%d Make_Out 2, num_blocks = %d, num_threads = %d\n",
        //    data_slice -> gpu_idx, num_blocks, AdvanceKernelPolicy::THREADS);
        //fflush(stdout);
        if (!enactor -> problem -> skip_makeout_selection)
        {
            for (int i=0; i< num_total_gpus; i++)
                data_slice -> out_length[i] = 1;
            data_slice -> out_length.Move(util::HOST, util::DEVICE,
                num_total_gpus, 0, stream);
            //printf("Make_Output direction = %s %s\n", FORWARD ? "FORWARD" : "", BACKWARD ? "BACKWARD" : "");

            /*printf("num_blocks = %d, num_threads = %d, stream = %p, "
                "num_elements = %d, num_total_gpus = %d, out_length = %p, (%d)"
                "keys_in = %p (%d), partition_table = %p (%d), convertion_table = %d (%d), "
                "vertex_associate_orgs = %p (%d), value__associate_orgs = %p (%d), "
                "keys_outs = %p (%d), vertex_associate_outs = %p (%d), value__associate_outs = %p (%d), "
                "keep_node_num = %s, num_vertex_associates = %d, num_value_associates = %d\n",
                num_blocks, AdvanceKernelPolicy::THREADS /2, stream,
                num_elements, num_total_gpus,
                data_slice -> out_length.GetPointer(util::DEVICE), data_slice -> out_length.GetSize(),
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                frontier_queue -> keys[frontier_attribute -> selector].GetSize(),
                graph_slice -> partition_table      .GetPointer(util::DEVICE),
                graph_slice -> partition_table      .GetSize(),
                graph_slice -> convertion_table     .GetPointer(util::DEVICE),
                graph_slice -> convertion_table     .GetSize(),
                data_slice  -> vertex_associate_orgs[0],
                data_slice  -> vertex_associate_orgs.GetSize(),
                data_slice  -> value__associate_orgs[0],
                data_slice  -> value__associate_orgs.GetSize(),
                data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                data_slice  -> keys_outs            .GetSize(),
                data_slice  -> vertex_associate_outs[1],
                data_slice  -> vertex_associate_outs.GetSize(),
                data_slice  -> value__associate_outs[1],
                data_slice  -> value__associate_outs.GetSize(),
                enactor -> problem -> keep_node_num ? "true" : "false",
                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES);*/

            if (FORWARD)
                Make_Output_Kernel < VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                    AdvanceKernelPolicy::CUDA_ARCH,
                    AdvanceKernelPolicy::LOG_THREADS-1>
                    <<< num_blocks, AdvanceKernelPolicy::THREADS / 2, 0, stream >>> (
                    num_elements,
                    num_total_gpus,
                    data_slice -> out_length.GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    graph_slice -> partition_table      .GetPointer(util::DEVICE),
                    graph_slice -> convertion_table     .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_outs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_outs.GetPointer(util::DEVICE),
                    enactor -> problem -> keep_node_num);

            if (BACKWARD)
                Make_Output_Backward_Kernel < VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                    AdvanceKernelPolicy::CUDA_ARCH,
                    AdvanceKernelPolicy::LOG_THREADS-1>
                    <<< num_blocks, AdvanceKernelPolicy::THREADS / 2, 0, stream >>> (
                    num_elements,
                    num_total_gpus,
                    data_slice -> out_length.GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    graph_slice -> backward_offset      .GetPointer(util::DEVICE),
                    graph_slice -> backward_partition   .GetPointer(util::DEVICE),
                    graph_slice -> backward_convertion  .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_outs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_outs.GetPointer(util::DEVICE),
                    enactor -> problem -> keep_node_num);

            data_slice -> out_length.Move(util::DEVICE, util::HOST,
                num_total_gpus, 0, stream);
            frontier_attribute->selector^=1;
        } else {
            Make_Output_Kernel_SkipSelection < VertexId, SizeT, Value,
                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                AdvanceKernelPolicy::CUDA_ARCH,
                AdvanceKernelPolicy::LOG_THREADS>
                <<< num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>> (
                num_elements,
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                data_slice -> vertex_associate_orgs.GetPointer(util::DEVICE),
                data_slice -> value__associate_orgs.GetPointer(util::DEVICE),
                data_slice -> keys_out[1]          .GetPointer(util::DEVICE),
                data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
                data_slice -> value__associate_out[1].GetPointer(util::DEVICE));
            for (int peer_=0; peer_<num_total_gpus; peer_++)
                data_slice -> out_length[peer_] = num_elements;
        }
        if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
            "Make_Output failed", __FILE__, __LINE__))
            return;
        if (!enactor -> problem -> skip_makeout_selection)
        {
            for (int i=0; i< num_total_gpus; i++)
            {
                data_slice -> out_length[i] --;
                //printf("out_length[%d] = %d\n", i, data_slice -> out_length[i]);
            }
        }
        //for (int i=0; i<num_total_gpus; i++)
        //{
            //if (i == 0)
            //    printf("%d, selector = %d, keys = %p\n",
            //        data_slice -> gpu_idx, frontier_attribute -> selector^1,
            //        data_slice -> keys_outs[i]);
        //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PostMakeOut",
        //        data_slice -> keys_outs[i], data_slice -> out_length[i],
        //        data_slice -> gpu_idx, enactor_stats -> iteration, i, stream);
        //}

        //printf("%d Make_Out 3\n", data_slice -> gpu_idx);
        //fflush(stdout);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
