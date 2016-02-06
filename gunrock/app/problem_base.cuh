// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base structure for all the application types
 */

#pragma once

#include <vector>
#include <string>

// Graph construction utilities
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>

// Information stats utilities
#include <boost/filesystem.hpp>
#include <gunrock/util/sysinfo.h>
#include <gunrock/util/gitsha1.h>
#include <gunrock/util/json_spirit_writer_template.h>

// Gunrock test error utilities
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/track_utils.cuh>

// Graph partitioner utilities
#include <gunrock/partitioner/random/random_partitioner.cuh>
#include <gunrock/partitioner/cluster/cluster_partitioner.cuh>
#include <gunrock/partitioner/biasrandom/biasrandom_partitioner.cuh>
#include <gunrock/partitioner/metis/metis_partitioner.cuh>
#include <gunrock/partitioner/static/static_partitioner.cuh>
#include <vector>
#include <string>

#include <moderngpu.cuh>

// this is the "stringize macro macro" hack
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */
enum FrontierType
{
    VERTEX_FRONTIERS,       // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,         // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS         // O(n) global vertex frontier, O(m) global edge frontier
};

/**
 * @brief Graph slice structure which contains common graph structural data.
 *
 * @tparam SizeT    Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value    Type to use as vertex / edge associated values
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
struct GraphSlice
{
    int             num_gpus; // Number of GPUs
    int             index   ; // Slice index
    VertexId        nodes   ; // Number of nodes in slice
    SizeT           edges   ; // Number of edges in slice

    Csr<VertexId, Value, SizeT   > *graph             ; // Pointer to CSR format subgraph
    util::Array1D<SizeT, SizeT   > row_offsets        ; // CSR format row offset
    util::Array1D<SizeT, VertexId> column_indices     ; // CSR format column indices
    util::Array1D<SizeT, SizeT   > out_degrees        ;
    util::Array1D<SizeT, SizeT   > column_offsets     ; // CSR format column offset
    util::Array1D<SizeT, VertexId> row_indices        ; // CSR format row indices
    util::Array1D<SizeT, SizeT   > in_degrees         ;
    util::Array1D<SizeT, int     > partition_table    ; // Partition number for vertices, local is always 0
    util::Array1D<SizeT, VertexId> convertion_table   ; // IDs of vertices in their hosting partition
    util::Array1D<SizeT, VertexId> original_vertex    ; // Original IDs of vertices
    util::Array1D<SizeT, SizeT   > in_counter         ; // Incoming vertices counter from peers
    util::Array1D<SizeT, SizeT   > out_offset         ; // Outgoing vertices offsets
    util::Array1D<SizeT, SizeT   > out_counter        ; // Outgoing vertices counter
    util::Array1D<SizeT, SizeT   > backward_offset    ; // Backward offsets for partition and conversion tables
    util::Array1D<SizeT, int     > backward_partition ; // Remote peers having the same vertices
    util::Array1D<SizeT, VertexId> backward_convertion; // IDs of vertices in remote peers

    /**
     * @brief GraphSlice Constructor
     *
     * @param[in] index GPU index.
     */
    GraphSlice(int index) :
        index(index),
        graph(NULL),
        num_gpus(0),
        nodes(0),
        edges(0)
    {
        row_offsets        .SetName("row_offsets"        );
        column_indices     .SetName("column_indices"     );
        out_degrees        .SetName("out_degrees"        );
        column_offsets     .SetName("column_offsets"     );
        row_indices        .SetName("row_indices"        );
        in_degrees         .SetName("in_degrees"         );
        partition_table    .SetName("partition_table"    );
        convertion_table   .SetName("convertion_table"   );
        original_vertex    .SetName("original_vertex"    );
        in_counter         .SetName("in_counter"         );
        out_offset         .SetName("out_offset"         );
        out_counter        .SetName("out_counter"        );
        backward_offset    .SetName("backward_offset"    );
        backward_partition .SetName("backward_partition" );
        backward_convertion.SetName("backward_convertion");
    }  // end GraphSlice(int index)

    /**
     * @brief GraphSlice Destructor to free all device memories.
     */
    virtual ~GraphSlice()
    {
        // Set device (use slice index)
        util::SetDevice(index);

        // Release allocated host / device memory
        row_offsets        .Release();
        column_indices     .Release();
        out_degrees        .Release();
        column_offsets     .Release();
        row_indices        .Release();
        in_degrees         .Release();
        partition_table    .Release();
        convertion_table   .Release();
        original_vertex    .Release();
        in_counter         .Release();
        out_offset         .Release();
        out_counter        .Release();
        backward_offset    .Release();
        backward_partition .Release();
        backward_convertion.Release();
    } // end ~GraphSlice()

    /**
      * @brief Initialize graph slice
      *
      * @param[in] stream_from_host    Whether to stream data from host
      * @param[in] num_gpus            Number of GPUs
      * @param[in] graph               Pointer to the sub graph
      * @param[in] inverstgraph        Pointer to the invert graph
      * @param[in] partition_table     The partition table
      * @param[in] convertion_table    The conversion table
      * @param[in] original_vertex     The original vertex table
      * @param[in] in_counter          In_counters
      * @param[in] out_offset          Out_offsets
      * @param[in] out_counter         Out_counters
      * @param[in] backward_offsets    Backward_offsets
      * @param[in] backward_partition  The backward partition table
      * @param[in] backward_convertion The backward conversion table
      * \return cudaError_t            Object indicating the success of all CUDA function calls
      */
    cudaError_t Init(
        bool                       stream_from_host,
        int                        num_gpus,
        Csr<VertexId, Value, SizeT>* graph,
        Csr<VertexId, Value, SizeT>* inverstgraph,
        int*                       partition_table,
        VertexId*                  convertion_table,
        VertexId*                  original_vertex,
        SizeT*                     in_counter,
        SizeT*                     out_offset,
        SizeT*                     out_counter,
        SizeT*                     backward_offsets   = NULL,
        int*                       backward_partition = NULL,
        VertexId*                  backward_convertion = NULL)
    {
        cudaError_t retval     = cudaSuccess;

        // Set local variables / array pointers
        this->num_gpus         = num_gpus;
        this->graph            = graph;
        this->nodes            = graph->nodes;
        this->edges            = graph->edges;
        if (partition_table  != NULL) this->partition_table    .SetPointer(partition_table      , nodes     );
        if (convertion_table != NULL) this->convertion_table   .SetPointer(convertion_table     , nodes     );
        if (original_vertex  != NULL) this->original_vertex    .SetPointer(original_vertex      , nodes     );
        if (in_counter       != NULL) this->in_counter         .SetPointer(in_counter           , num_gpus + 1);
        if (out_offset       != NULL) this->out_offset         .SetPointer(out_offset           , num_gpus + 1);
        if (out_counter      != NULL) this->out_counter        .SetPointer(out_counter          , num_gpus + 1);
        this->row_offsets        .SetPointer(graph->row_offsets   , nodes + 1   );
        this->column_indices     .SetPointer(graph->column_indices, edges     );
        if (inverstgraph != NULL)
        {
            this->column_offsets .SetPointer(inverstgraph->row_offsets, nodes + 1);
            this->row_indices    .SetPointer(inverstgraph->column_indices   , edges  );
        }

        do
        {
            // Set device using slice index
            if (retval = util::SetDevice(index)) break;

            // Allocate and initialize row_offsets
            if (retval = this->row_offsets.Allocate(nodes + 1      , util::DEVICE)) break;
            if (retval = this->row_offsets.Move    (util::HOST   , util::DEVICE)) break;

            // Allocate and initialize column_indices
            if (retval = this->column_indices.Allocate(edges     , util::DEVICE)) break;
            if (retval = this->column_indices.Move    (util::HOST, util::DEVICE)) break;

            // Allocate out degrees for each node
            if (retval = this->out_degrees.Allocate(nodes        , util::DEVICE)) break;
            // count number of out-going degrees for each node
            util::MemsetMadVectorKernel <<< 128, 128>>>(
                this->out_degrees.GetPointer(util::DEVICE),
                this->row_offsets.GetPointer(util::DEVICE),
                this->row_offsets.GetPointer(util::DEVICE) + 1,
                (SizeT)-1, nodes);


            if (inverstgraph != NULL)
            {
                // Allocate and initialize column_offsets
                if (retval = this->column_offsets.Allocate(nodes + 1      , util::DEVICE)) break;
                if (retval = this->column_offsets.Move    (util::HOST   , util::DEVICE)) break;

                // Allocate and initialize row_indices
                if (retval = this->row_indices.Allocate(edges     , util::DEVICE)) break;
                if (retval = this->row_indices.Move    (util::HOST, util::DEVICE)) break;


                if (retval = this->in_degrees .Allocate(nodes,  util::DEVICE)) break;
                // count number of in-going degrees for each node
                util::MemsetMadVectorKernel <<< 128, 128>>>(
                    this->in_degrees    .GetPointer(util::DEVICE),
                    this->column_offsets.GetPointer(util::DEVICE),
                    this->column_offsets.GetPointer(util::DEVICE) + 1,
                    (SizeT)-1, nodes);
            }

            // For multi-GPU cases
            if (num_gpus > 1)
            {
                // Allocate and initialize convertion_table
                if (retval = this->partition_table.Allocate (nodes     , util::DEVICE)) break;
                if (partition_table  != NULL)
                    if (retval = this->partition_table.Move (util::HOST, util::DEVICE)) break;

                // Allocate and initialize convertion_table
                if (retval = this->convertion_table.Allocate(nodes     , util::DEVICE)) break;
                if (convertion_table != NULL)
                    if (retval = this->convertion_table.Move(util::HOST, util::DEVICE)) break;

                // Allocate and initialize original_vertex
                if (retval = this->original_vertex .Allocate(nodes     , util::DEVICE)) break;
                if (original_vertex  != NULL)
                    if (retval = this->original_vertex .Move(util::HOST, util::DEVICE)) break;

                // If need backward information proration
                if (backward_offsets != NULL)
                {
                    // Allocate and initialize backward_offset
                    this->backward_offset    .SetPointer(backward_offsets     , nodes + 1);
                    if (retval = this->backward_offset    .Allocate(nodes + 1, util::DEVICE)) break;
                    if (retval = this->backward_offset    .Move(util::HOST, util::DEVICE)) break;

                    // Allocate and initialize backward_partition
                    this->backward_partition .SetPointer(backward_partition   , backward_offsets[nodes]);
                    if (retval = this->backward_partition .Allocate(backward_offsets[nodes], util::DEVICE)) break;
                    if (retval = this->backward_partition .Move(util::HOST, util::DEVICE)) break;

                    // Allocate and initialize backward_convertion
                    this->backward_convertion.SetPointer(backward_convertion  , backward_offsets[nodes]);
                    if (retval = this->backward_convertion.Allocate(backward_offsets[nodes], util::DEVICE)) break;
                    if (retval = this->backward_convertion.Move(util::HOST, util::DEVICE)) break;
                }
            } // end if num_gpu>1
        }
        while (0);

        return retval;
    } // end of Init(...)

    /**
     * @brief overloaded = operator
     *
     * @param[in] other GraphSlice to copy from
     *
     * \return GraphSlice& a copy of local GraphSlice
     */
    GraphSlice& operator=(GraphSlice other)
    {
        num_gpus            = other.num_gpus           ;
        index               = other.index              ;
        nodes               = other.nodes              ;
        edges               = other.edges              ;
        graph               = other.graph              ;
        row_offsets         = other.row_offsets        ;
        column_indices      = other.column_indices     ;
        column_offsets      = other.column_offsets     ;
        row_indices         = other.row_indices        ;
        partition_table     = other.partition_table    ;
        convertion_table    = other.convertion_table   ;
        original_vertex     = other.original_vertex    ;
        in_counter          = other.in_counter         ;
        out_offset          = other.out_offset         ;
        out_counter         = other.out_counter        ;
        backward_offset     = other.backward_offset    ;
        backward_partition  = other.backward_partition ;
        backward_convertion = other.backward_convertion;
        return *this;
    } // end operator=()

}; // end GraphSlice

/**
 * @brief Base data slice structure which contains common data structural needed for primitives.
 *
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam Value               Type to use as vertex / edge associated values
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
struct DataSliceBase
{
    int                              gpu_idx                 ; // GPU index
    SizeT                            nodes                   ; // Number of vertices
    util::Array1D<SizeT, VertexId>   preds                   ; // predecessors of vertices
    util::Array1D<SizeT, VertexId>   temp_preds              ; // temporary storages for predecessors

    /**
     * @brief DataSliceBase default constructor
     */
    DataSliceBase() :
        gpu_idx (0),
        nodes   (0)
    {
        // Assign names to arrays
        preds                  .SetName("preds"                  );
        temp_preds             .SetName("temp_preds"             );
    } // end DataSliceBase()

    /**
     * @brief DataSliceBase default destructor to release host / device memory
     */
    virtual ~DataSliceBase()
    {
        Release();
    }
     
    cudaError_t Release()
    {
        cudaError_t retval;   
        // Set device by index
        if (retval = util::SetDevice(gpu_idx)) return retval;
        if (retval = preds         .Release()) return retval;
        if (retval = temp_preds    .Release()) return retval;
    } // end ~DataSliceBase()

    /**
     * @brief Initiate DataSliceBase
     *
     * @param[in] gpu_idx              GPU index
     * @param[in] graph                Pointer to the CSR formated sub-graph
     * \return                         Error occurred if any, otherwise cudaSuccess
     */
    cudaError_t Init(
        int    gpu_idx             ,
        Csr<VertexId, Value, SizeT>
              *graph               )
    {
        cudaError_t retval         = cudaSuccess;
        // Copy input values
        this->gpu_idx              = gpu_idx;
        this->nodes                = graph->nodes;

        return retval;
    } // end Init(..)

    /**
     * @brief Performs reset work needed for DataSliceBase. Must be called prior to each search
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
     cudaError_t Reset()
     {
        cudaError_t retval = cudaSuccess;
        return retval;
    } // end Reset(...)

}; // end DataSliceBase

/**
 * @brief Base test parameter structure
 */
struct TestParameter_Base
{
public:
    json_spirit::mObject        info; // JSON information storing running stats
    bool                     g_quiet; // Don't print anything unless specifically directed
    bool          g_quick           ; // Whether or not to skip CPU based computation
    bool          g_stream_from_host; // Whether or not to use stream data from host
    bool          g_undirected      ; // Whether or not to use undirected graph
    bool          instrumented      ; // Whether or not to collect instrumentation from kernels
    bool          debug             ; // Whether or not to use debug mode
    bool          size_check        ; // Whether or not to enable size_check
    bool          mark_predecessors ; // Whether or not to mark src-distance vs. parent vertices
    bool          enable_idempotence; // Whether or not to enable idempotent operation
    void         *graph             ; // Pointer to the input CSR graph
    long long    *src               ; // Source vertex IDs
    int           max_grid_size     ; // maximum grid size (0: leave it up to the enactor)
    int           num_gpus          ; // Number of GPUs for multi-GPU enactor to use
    double        max_queue_sizing  ; // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    double        max_in_sizing     ; // Maximum size scaling factor for data communication
    void         *context           ; // GPU context array used by MordernGPU
    std::string   partition_method  ; // Partition method
    int          *gpu_idx           ; // Array of GPU indices
    cudaStream_t *streams           ; // Array of GPU streams
    float         partition_factor  ; // Partition factor
    int           partition_seed    ; // Partition seed
    int           iterations        ; // Number of repeats
    int           traversal_mode    ; // Load-balanced or Dynamic cooperative

    /**
     * @brief TestParameter_Base constructor
     */
    TestParameter_Base()
    {
        // Assign default values
        g_quiet            = false;
        g_quick            = false;
        g_stream_from_host = false;
        g_undirected       = false;
        instrumented       = false;
        debug              = false;
        size_check         = true;
        graph              = NULL;
        src                = NULL;
        max_grid_size      = 0;
        num_gpus           = 1;
        max_queue_sizing   = 1.0;
        max_in_sizing      = 1.0;
        context            = NULL;
        partition_method   = "random";
        gpu_idx            = NULL;
        streams            = NULL;
        partition_factor   = -1;
        partition_seed     = -1;
        iterations         = 1;
        traversal_mode     = -1;
    }  // end TestParameter_Base()

    /**
     * @brief TestParameter_Base destructor
     */
    ~TestParameter_Base()
    {
        // Clear pointers
        graph   = NULL;
        context = NULL;
        gpu_idx = NULL;
        streams = NULL;
    } // end ~TestParameter_Base()

    /**
     * @brief Utility function to print parameters for debugging
     */
    void PrintParameters()
    {
        using std::cout;
        using std::endl;

        cout << endl << "______________________________"       << endl;
        cout << "==> Test Parameters:  "                       << endl;
        cout << "g_quiet:            \t" << g_quiet            << endl;
        cout << "g_quick:            \t" << g_quick            << endl;
        cout << "g_stream_from_host: \t" << g_stream_from_host << endl;
        cout << "g_undirected:       \t" << g_undirected       << endl;
        cout << "instrumented:       \t" << instrumented       << endl;
        cout << "debug:              \t" << debug              << endl;
        cout << "size_check:         \t" << size_check         << endl;
        cout << "mark_predecessors:  \t" << mark_predecessors  << endl;
        cout << "enable_idempotence: \t" << enable_idempotence << endl;
        cout << "src:                \t" << src                << endl;
        cout << "max_grid_size:      \t" << max_grid_size      << endl;
        cout << "num_gpus:           \t" << num_gpus           << endl;
        cout << "max_queue_sizing:   \t" << max_queue_sizing   << endl;
        cout << "max_in_sizing:      \t" << max_in_sizing      << endl;
        cout << "partition_method:   \t" << partition_method   << endl;
        cout << "partition_factor:   \t" << partition_factor   << endl;
        cout << "partition_seed:     \t" << partition_seed     << endl;
        cout << "iterations:         \t" << iterations         << endl;
        cout << "traversal_mode:     \t" << traversal_mode     << endl;
        cout << "------------------------------" << endl       << endl;
    }

    /**
     * @brief Utility function to print collected info
     */
    void PrintInfo()
    {
        json_spirit::write_stream(json_spirit::mValue(info), std::cout,
                                  json_spirit::pretty_print);
    }

    /**
     * @brief Initialization process for TestParameter_Base
     *
     * @param[in] args Command line arguments
     */
    void Init(util::CommandLineArgs &args)
    {
        bool disable_size_check = true;

        // Get settings from command line arguments
        instrumented       = args.CheckCmdLineFlag("instrumented");
        disable_size_check = args.CheckCmdLineFlag("disable-size-check");
        size_check         = !disable_size_check;
        debug              = args.CheckCmdLineFlag("v");
        g_quick            = args.CheckCmdLineFlag("quick");
        g_undirected       = args.CheckCmdLineFlag("undirected");
        args.GetCmdLineArgument("queue-sizing"    , max_queue_sizing);
        args.GetCmdLineArgument("in-sizing"       , max_in_sizing   );
        args.GetCmdLineArgument("grid-size"       , max_grid_size   );
        args.GetCmdLineArgument("partition-factor", partition_factor);
        args.GetCmdLineArgument("partition-seed"  , partition_seed  );
        args.GetCmdLineArgument("iteration-num"   , iterations      );
        if (args.CheckCmdLineFlag  ("partition-method"))
            args.GetCmdLineArgument("partition-method", partition_method);
    }  // end Init(..)
};

/**
 * @brief Base problem structure.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer
 * @tparam _MARK_PREDCESSORS    Whether or not to mark predecessors for vertices
 * @tparam _ENABLE_IDEMPOTENCE  Whether or not to use idempotent
 * @tparam _USE_DOUBLE_BUFFER   Whether or not to use double buffer for frontier queues
 * @tparam _ENABLE_BACKWARD     Whether or not to use backward propagation
 * @tparam _KEEP_ORDER          Whether or not to keep vertices order after partitioning
 * @tparam _KEEP_NODE_NUM       Whether or not to keep vertex IDs after partitioning
 */
template <
    typename    _VertexId,
    typename    _SizeT,
    typename    _Value,
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE,
    bool        _USE_DOUBLE_BUFFER,
    bool        _ENABLE_BACKWARD = false,
    bool        _KEEP_ORDER      = false,
    bool        _KEEP_NODE_NUM   = false >
struct ProblemBase
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;
    typedef _Value              Value;
    static const bool           MARK_PREDECESSORS  = _MARK_PREDECESSORS ;
    static const bool           ENABLE_IDEMPOTENCE = _ENABLE_IDEMPOTENCE;
    static const bool           USE_DOUBLE_BUFFER  = _USE_DOUBLE_BUFFER ;
    static const bool           ENABLE_BACKWARD    = _ENABLE_BACKWARD   ;

    /**
     * Load instruction cache-modifier const defines.
     */
    static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER                    = util::io::ld::cg;             // Load instruction cache-modifier for reading incoming frontier vertex-ids. Valid on SM2.0 or newer
    static const util::io::ld::CacheModifier COLUMN_READ_MODIFIER                   = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR column-indices.
    static const util::io::ld::CacheModifier EDGE_VALUES_READ_MODIFIER              = util::io::ld::NONE;           // Load instruction cache-modifier for reading edge values.
    static const util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER       = util::io::ld::cg;             // Load instruction cache-modifier for reading CSR row-offsets (8-byte aligned)
    static const util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER     = util::io::ld::NONE;           // Load instruction cache-modifier for reading CSR row-offsets (4-byte aligned)
    static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER                   = util::io::st::cg;             // Store instruction cache-modifier for writing outgoing frontier vertex-ids. Valid on SM2.0 or newer

    // Members
    int                 num_gpus              ; // Number of GPUs to be sliced over
    int                 *gpu_idx              ; // GPU indices
    SizeT               nodes                 ; // Number of vertices in the graph
    SizeT               edges                 ; // Number of edges in the graph
    GraphSlice<SizeT, VertexId, Value>
    **graph_slices        ; // Set of graph slices (one for each GPU)
    Csr<VertexId, Value, SizeT> *sub_graphs     ; // Subgraphs for multi-GPU implementation
    Csr<VertexId, Value, SizeT> *org_graph      ; // Original graph
    PartitionerBase<VertexId, SizeT, Value, _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
    *partitioner          ; // Partitioner
    int                 **partition_tables    ; // Partition tables indicating which GPU the vertices are hosted
    VertexId            **convertion_tables   ; // Conversions tables indicating vertex IDs on local / remote GPUs
    VertexId            **original_vertexes   ; // Vertex IDs in the original graph
    SizeT               **in_counter          ; // Number of in vertices
    SizeT               **out_offsets         ; // Out offsets for data communication
    SizeT               **out_counter         ; // Number of out vertices
    SizeT               **backward_offsets    ; // Offsets for backward propagation
    int                 **backward_partitions ; // Partition tables for backward propagation
    VertexId            **backward_convertions; // Conversion tables for backward propagation

    // Methods

    /**
     * @brief ProblemBase default constructor
     */
    ProblemBase() :
        num_gpus            (0   ),
        gpu_idx             (NULL),
        nodes               (0   ),
        edges               (0   ),
        graph_slices        (NULL),
        sub_graphs          (NULL),
        org_graph           (NULL),
        partitioner         (NULL),
        partition_tables    (NULL),
        convertion_tables   (NULL),
        original_vertexes   (NULL),
        in_counter          (NULL),
        out_offsets         (NULL),
        out_counter         (NULL),
        backward_offsets    (NULL),
        backward_partitions (NULL),
        backward_convertions(NULL)
    {
    } // end ProblemBase()

    /**
     * @brief ProblemBase default destructor to free all graph slices allocated.
     */
    virtual ~ProblemBase()
    {
        // Cleanup graph slices on the heap
        for (int i = 0; i < num_gpus; ++i)
        {
            delete graph_slices[i]; graph_slices[i] = NULL;
        }
        if (num_gpus > 1)
        {
            delete partitioner; partitioner = NULL;
        }
        delete[] graph_slices; graph_slices = NULL;
        delete[] gpu_idx;      gpu_idx      = NULL;
    }  // end ~ProblemBase()

    /**
     * @brief Get the GPU index for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Index of the GPU that owns the neighbor list of the specified vertex
     */
    template <typename VertexId>
    int GpuIndex(VertexId vertex)
    {
        if (num_gpus <= 1)
        {

            // Special case for only one GPU, which may be set as with
            // an ordinal other than 0.
            return graph_slices[0]->index;
        }
        else
        {
            return partition_tables[0][vertex];
        }
    }

    /**
     * @brief Get the row offset for a specified vertex id.
     *
     * @tparam VertexId Type of signed integer to use as vertex id
     * @param[in] vertex Vertex Id to search
     * \return Row offset of the specified vertex. If a single GPU is used,
     * this will be the same as the vertex id.
     */
    template <typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
        if (num_gpus <= 1)
        {
            return vertex;
        }
        else
        {
            return convertion_tables[0][vertex];
        }
    }

    /**
     * @brief Initialize problem from host CSR graph.
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph            Pointer to the input CSR graph.
     * @param[in] inverse_graph    Pointer to the inversed input CSR graph.
     * @param[in] num_gpus         Number of GPUs
     * @param[in] gpu_idx          Array of GPU indices
     * @param[in] partition_method Partition methods
     * @param[in] queue_sizing     Queue sizing
     * @param[in] partition_factor Partition factor
     * @param[in] partition_seed   Partition seed
     *
     * \return cudaError_t object indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        bool        stream_from_host,
        Csr<VertexId, Value, SizeT> *graph,
        Csr<VertexId, Value, SizeT> *inverse_graph = NULL,
        int         num_gpus          = 1,
        int         *gpu_idx          = NULL,
        std::string partition_method  = "random",
        float       partition_factor  = -1,
        int         partition_seed    = -1)
    {
        cudaError_t retval      = cudaSuccess;
        this->org_graph         = graph;
        this->nodes             = graph->nodes;
        this->edges             = graph->edges;
        this->num_gpus          = num_gpus;
        this->gpu_idx           = new int [num_gpus];

        do
        {
            if (num_gpus == 1 && gpu_idx == NULL)
            {
                if (retval = util::GRError(cudaGetDevice(&(this->gpu_idx[0])),
                    "ProblemBase cudaGetDevice failed", __FILE__, __LINE__)) break;
            }
            else
            {
                for (int gpu = 0; gpu < num_gpus; gpu++)
                    this->gpu_idx[gpu] = gpu_idx[gpu];
            }

            graph_slices = new GraphSlice<SizeT, VertexId, Value>*[num_gpus];

            if (num_gpus > 1)
            {
                util::CpuTimer cpu_timer;

                //printf("partition_method = %s\n", partition_method.c_str());
                if      (partition_method == "random")
                {
                    partitioner = new rp::RandomPartitioner <
                        VertexId, SizeT, Value, 
                        _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                } 
                else if (partition_method == "metis" )
                {
                    partitioner = new metisp::MetisPartitioner <
                        VertexId, SizeT, Value, 
                        _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                }
                else if (partition_method == "static")
                {
                    partitioner = new sp::StaticPartitioner <
                        VertexId, SizeT, Value, 
                        _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                }
                else if (partition_method == "cluster")
                {
                    partitioner = new cp::ClusterPartitioner <
                        VertexId, SizeT, Value, 
                        _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                }
                else if (partition_method == "biasrandom")
                {
                    partitioner = new brp::BiasRandomPartitioner <
                        VertexId, SizeT, Value, 
                        _ENABLE_BACKWARD, _KEEP_ORDER, _KEEP_NODE_NUM>
                        (*graph,num_gpus);
                }
                else 
                {
                    util::GRError("partition_method invalid", __FILE__,__LINE__);
                }
                cpu_timer.Start();
                retval = partitioner->Partition(
                     sub_graphs,
                     partition_tables,
                     convertion_tables,
                     original_vertexes,
                     in_counter,
                     out_offsets,
                     out_counter,
                     backward_offsets,
                     backward_partitions,
                     backward_convertions,
                     partition_factor,
                     partition_seed);
                cpu_timer.Stop();
                printf("partition end. (%f ms)\n", cpu_timer.ElapsedMillis());fflush(stdout);
                
                //util::cpu_mt::PrintCPUArray<SizeT, int>("partition0", partition_tables[0], graph -> nodes > 10 ? 10 : graph -> nodes);
                //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("convertion0", convertion_tables[0], graph -> nodes > 10 ? 10 : graph -> nodes);

                /*graph->DisplayGraph("org_graph", graph->nodes);
                util::cpu_mt::PrintCPUArray<SizeT,int>("partition0",partition_tables[0],graph->nodes);
                util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion0",convertion_tables[0],graph->nodes);
                //util::cpu_mt::PrintCPUArray<SizeT,Value>("edge_value",graph->edge_values,graph->edges);
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    sub_graphs[gpu].DisplayGraph("sub_graph",sub_graphs[gpu].nodes);
                    printf("%d\n",gpu);
                    util::cpu_mt::PrintCPUArray<SizeT,int     >("partition"           , partition_tables    [gpu+1], sub_graphs[gpu].nodes);
                    util::cpu_mt::PrintCPUArray<SizeT,VertexId>("convertion"          , convertion_tables   [gpu+1], sub_graphs[gpu].nodes);
                    //util::cpu_mt::PrintCPUArray<SizeT,SizeT   >("backward_offsets"    , backward_offsets    [gpu], sub_graphs[gpu].nodes);
                    //util::cpu_mt::PrintCPUArray<SizeT,int     >("backward_partitions" , backward_partitions [gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
                    //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("backward_convertions", backward_convertions[gpu], backward_offsets[gpu][sub_graphs[gpu].nodes]);
                }*/
                //for (int gpu=0;gpu<num_gpus;gpu++)
                //{
                //    cross_counter[gpu][num_gpus]=0;
                //    for (int peer=0;peer<num_gpus;peer++)
                //    {
                //        cross_counter[gpu][peer]=out_offsets[gpu][peer+1]-out_offsets[gpu][peer];
                //    }
                //    cross_counter[gpu][num_gpus]=in_offsets[gpu][num_gpus];
                //}
                /*for (int gpu=0;gpu<num_gpus;gpu++)
                for (int peer=0;peer<=num_gpus;peer++)
                {
                    in_offsets[gpu][peer]*=2;
                    out_offsets[gpu][peer]*=2;
                }*/
                if (retval) break;
            }
            else
            {
                sub_graphs = graph;
            }

            /*if (TO_TRACK)
            {
                for (VertexId v=0; v < graph->nodes; v++)
                if (to_track(-1, v))
                {
                    printf("Vertex %d: ", v);
                    if (num_gpus > 1)
                    {
                        int gpu = 0;
                        VertexId v_ = 0;
                        //for (gpu=0; gpu<num_gpus; gpu++)
                        //    for (v_=0; v_ < sub_graphs[gpu].nodes; v_++)
                        //        if (original_vertexes[gpu][v_] == v)
                        //            printf("(%d, %d), ", gpu, v_);
                        gpu = partition_tables[0][v];
                        v_ = convertion_tables[0][v];
                        printf("\n (%d, %d) -> {%d :", gpu, v_, 
                            sub_graphs[gpu].row_offsets[v_+1] - sub_graphs[gpu].row_offsets[v_]);
                        for (VertexId j = sub_graphs[gpu].row_offsets[v_];
                            j < sub_graphs[gpu].row_offsets[v_+1]; j++)
                            printf(" %d,", sub_graphs[gpu].column_indices[j]);
                        printf("}\n");

                        for (gpu = 0; gpu<num_gpus; gpu++)
                        for (v_=0; v_ < sub_graphs[gpu].nodes; v_++)
                        if (original_vertexes[gpu][v_] == v)
                        {
                            printf("{");
                            for (VertexId u=0; u < sub_graphs[gpu].nodes; u++)
                            for (VertexId j=sub_graphs[gpu].row_offsets[u];
                                j<sub_graphs[gpu].row_offsets[u+1]; j++)
                            if (sub_graphs[gpu].column_indices[j] == v_)
                            {
                                printf(" %d(%d),", u, original_vertexes[gpu][u]);
                            }
                            printf("} -> (%d, %d)\n", gpu, v_);
                        }
                    } else {
                        printf(" -> {%d :", graph->row_offsets[v+1] - graph->row_offsets[v]);
                        for (VertexId j = graph->row_offsets[v]; j< graph->row_offsets[v+1]; j++)
                            printf(" %d,", graph -> column_indices[j]);
                        printf("}\n");
                    }
                }
            }*/

            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                graph_slices[gpu] = new GraphSlice<SizeT, VertexId, Value>(this->gpu_idx[gpu]);
                if (num_gpus > 1)
                {
                    if (_ENABLE_BACKWARD)
                    {
                        retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs     [gpu]),
                            NULL,
                            partition_tables    [gpu + 1],
                            convertion_tables   [gpu + 1],
                            original_vertexes   [gpu],
                            in_counter          [gpu],
                            out_offsets         [gpu],
                            out_counter         [gpu],
                            backward_offsets    [gpu],
                            backward_partitions [gpu],
                            backward_convertions[gpu]);
                    }
                    else
                    {
                        retval = graph_slices[gpu]->Init(
                            stream_from_host,
                            num_gpus,
                            &(sub_graphs[gpu]),
                            NULL,
                            partition_tables [gpu + 1],
                            convertion_tables[gpu + 1],
                            original_vertexes[gpu],
                            in_counter       [gpu],
                            out_offsets      [gpu],
                            out_counter      [gpu],
                            NULL,
                            NULL,
                            NULL);
                    }
                }
                else
                {
                    retval = graph_slices[gpu]->Init(
                        stream_from_host,
                        num_gpus,
                        &(sub_graphs[gpu]),
                        inverse_graph,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL);
                }
                if (retval) break;
            }  // end for (gpu)

        }
        while (0);

        return retval;
    } // end Init(...)

};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
