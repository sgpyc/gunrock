// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * mf_test.cu
 *
 * @brief Test related functions for Max Flow algorithm.
 */

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a); \
    printf("\n");}

#pragma once

#ifdef BOOST_FOUND
    // Boost includes for CPU Push Relabel Max Flow reference algorithms
    #include <boost/config.hpp>
    #include <iostream>
    #include <string>
    #include <boost/graph/edmonds_karp_max_flow.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/read_dimacs.hpp>
#endif

#include <gunrock/app/mf/mf_helpers.cuh>
#include <queue>

namespace gunrock {
namespace app {
namespace mf {

/*****************************************************************************
 * Housekeeping Routines
 ****************************************************************************/

/**
 * @brief Displays the MF result
 *
 * @tparam ValueT     Type of capacity/flow/excess
 * @tparam VertxeT    Type of vertex 
 * 
 * @param[in] h_flow  Flow calculated on edges 
 * @param[in] source  Index of source vertex
 * @param[in] nodes   Number of nodes
 */
template<typename GraphT, typename ValueT, typename VertexT>
void DisplaySolution(GraphT graph, ValueT* h_flow, VertexT* reverse, 
	VertexT sink, VertexT nodes)
{
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::SizeT SizeT;
    ValueT flow_incoming_sink = 0;
    SizeT e_start = graph.CsrT::GetNeighborListOffset(sink);
    SizeT num_neighbors = graph.CsrT::GetNeighborListLength(sink);
    SizeT e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e)
    {
	ValueT flow = h_flow[reverse[e]];
	if (util::isValid(flow))
	    flow_incoming_sink += flow;
    }

    util::PrintMsg("The maximum amount of flow that is feasible to reach \
	    from source to sink is " + std::to_string(flow_incoming_sink), 
	    true, false);
}

/**
 * @brief For given vertex v, find neighbor which the smallest height.  
 *
 * @tparam ValueT	Type of capacity/flow/excess
 * @tparam VertxeT	Type of vertex
 * @tparam GraphT	Type of graph
 * @param[in] graph	Graph
 * @param[in] x		Index of vertex 
 * @param[in] height	Function of height on nodes 
 * @param[in] capacity	Function of capacity on edges
 * @param[in] flow	Function of flow on edges
 *
 * return Index the lowest neighbor of vertex x
 */
template<typename ValueT, typename VertexT, typename GraphT>
VertexT find_lowest(GraphT graph, VertexT x, VertexT* height, ValueT* flow, 
	VertexT source){
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CsrT CsrT;

    auto e_start = graph.CsrT::GetNeighborListOffset(x);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
    auto e_end = e_start + num_neighbors;
    VertexT lowest;
    SizeT lowest_id = util::PreDefinedValues<SizeT>::InvalidValue; 
    for (auto e = e_start; e < e_end; ++e)
    {
        //if (graph.CsrT::edge_values[e] - flow[e] > (ValueT)0){
        if (graph.CsrT::edge_values[e] - flow[e] > MF_EPSILON){
            auto y = graph.CsrT::GetEdgeDest(e);
            if (!util::isValid(lowest_id) || height[y] < lowest){
                lowest = height[y];
                lowest_id = e;
            }
        }
    }
    return lowest_id;
}

/**
  * @brief Relabel: increases height of given vertex
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] x	Index of vertex
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT, typename GraphT>
bool relabel(GraphT graph, VertexT x, VertexT* height, ValueT* flow, 
	VertexT source){
    typedef typename GraphT::CsrT CsrT;
    auto e = find_lowest(graph, x, height, flow, source);
    // graph.edges is unreachable value = there is no valid neighbour
    if (util::isValid(e)) {
        VertexT y = graph.CsrT::GetEdgeDest(e);
        if (height[y] >= height[x]){
    	    debug_aml("relabel %d H: %d->%d, res-cap %d-%d: %lf\n", x, height[x], 
    		    height[y]+1, x, y, graph.CsrT::edge_values[e]-flow[e]);
            height[x] = height[y] + 1;
            return true;
        }
    }
    return false;
}

/**
  * @brief Push: transfers flow from given vertex to neighbors in residual 
  *	   network which are lower than it.
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] x	Index of vertex
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT, typename GraphT>
bool push(GraphT& graph, VertexT x, ValueT* excess, VertexT* height,
    ValueT* flow, VertexT* reverse){
    typedef typename GraphT::CsrT CsrT;
    //if (excess[x] > (ValueT)0){
    if (excess[x] > MF_EPSILON){
        auto e_start = graph.CsrT::GetNeighborListOffset(x);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            auto y = graph.CsrT::GetEdgeDest(e);
            auto c = graph.CsrT::edge_values[e];
            //if (c - flow[e] > (ValueT) 0 and height[x] > height[y]){
            if (c - flow[e] > MF_EPSILON and height[x] > height[y]){
            auto move = std::min(c - flow[e], excess[x]);
    //		printf("push %lf from %d (H=%d) to %d (H=%d)\n", 
    //			move, x, height[x], y, height[y]);
            excess[x] -= move;
            excess[y] += move;
            flow[e] += move;
            flow[reverse[e]] -= move;
            return true;
            }
        }
    }
    return false;
}

/**
  * @brief Push relabel reference algorithm
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] source	Source vertex
  * @param[in] sink	Sink vertex
  * @param[in] reverse	For given edge returns reverse one
  *
  * return Value of computed max flow
  */
template<typename ValueT, typename VertexT, typename GraphT>
ValueT max_flow(GraphT& graph, ValueT* flow, ValueT* excess, VertexT* height, 
	VertexT source, VertexT sink, VertexT* reverse){
    bool update = true;

    int iter = 0;
    while (update) {
        ++iter;
        update = false;
        for (VertexT x = 0; x < graph.nodes; ++x){
            //if (x != sink and x != source and excess[x] > (ValueT)0){
            if (x != sink and x != source and excess[x] > MF_EPSILON){
                if (push(graph, x, excess, height, flow, reverse) or 
                        relabel(graph, x, height, flow, source))
                {
                    update = true;
                    if (iter > 0 && iter % 100 == 0)
                        relabeling(graph, source, sink, height, reverse, flow); 
                }
            }
        }
    }

    return excess[sink];
}

/**
  * @brief Min Cut algorithm
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] source	Source vertex
  * @param[in] sink	Sink vertex
  * @param[in] flow	Function of flow on edges
  * @param[out] min_cut	Function on nodes, 1 = connected to source, 0 = sink
  *
  */
template <typename VertexT, typename ValueT, typename GraphT>
void minCut(GraphT& graph, VertexT  src, ValueT* flow, int* min_cut, 
	    bool* vertex_reachabilities, ValueT* residuals)
{
    typedef typename GraphT::CsrT CsrT;
    memset(vertex_reachabilities, true, graph.nodes * sizeof(vertex_reachabilities[0]));
    std::queue<VertexT> que;
    que.push(src);
    min_cut[src] = 1;

    for (auto e = 0; e < graph.edges; e++) {
	residuals[e] = graph.CsrT::edge_values[e] - flow[e];
    }

    while (! que.empty()){
        auto v = que.front(); que.pop();

        auto e_start = graph.CsrT::GetNeighborListOffset(v);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            auto u = graph.CsrT::GetEdgeDest(e);
            if (vertex_reachabilities[u] and abs(residuals[e]) > MF_EPSILON){
                vertex_reachabilities[u] = false;
                que.push(u);
                min_cut[u] = 1;
            }
        }
    }
}

/****************************************************************************
 * MF Testing Routines
 ***************************************************************************/

/**
 * @brief Simple CPU-based reference MF implementations
 *
 * @tparam GraphT   Type of the graph
 * @tparam VertexT  Type of the vertex
 * @tparam ValueT   Type of the capacity/flow/excess
 * @param[in]  parameters Running parameters
 * @param[in]  graph      Input graph
 * @param[in]  src        The source vertex
 * @param[in]  sin        The sink vertex
 * @param[out] maxflow	  Value of computed maxflow reached sink
 * @param[out] reverse	  Computed reverse
 * @param[out] edges_flow Computed flows on edges
 *
 * \return     double      Time taken for the MF
 */
template <typename VertexT, typename ValueT, typename GraphT>
double CPU_Reference(
	util::Parameters &parameters,
	GraphT &graph,
	VertexT src,
	VertexT sin,
	ValueT &maxflow,
	VertexT *reverse,
	ValueT *flow)
{

    debug_aml("CPU_Reference start");
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CsrT CsrT;

    double elapsed = 0;

#if (BOOST_FOUND==1)
    
    debug_aml("boost found");
    using namespace boost;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
    typedef adjacency_list < vecS, vecS, directedS, 
	    property < vertex_name_t, std::string >,
	    property < edge_capacity_t, ValueT,
	    property < edge_residual_capacity_t, ValueT,
	    property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;
    
    Graph boost_graph;

    typename property_map < Graph, edge_capacity_t >::type 
	capacity = get(edge_capacity, boost_graph);

    typename property_map < Graph, edge_reverse_t >::type 
	rev = get(edge_reverse, boost_graph);

    typename property_map < Graph, edge_residual_capacity_t >::type 
	residual_capacity = get(edge_residual_capacity, boost_graph);

    std::vector<Traits::vertex_descriptor> verts;
    for (VertexT v = 0; v < graph.nodes; ++v)
	verts.push_back(add_vertex(boost_graph));
    
    Traits::vertex_descriptor source = verts[src];
    Traits::vertex_descriptor sink = verts[sin];
    debug_aml("src = %d, sin %d", source, sink);

    for (VertexT x = 0; x < graph.nodes; ++x){
        auto e_start = graph.CsrT::GetNeighborListOffset(x);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            VertexT y = graph.CsrT::GetEdgeDest(e);
            ValueT cap = graph.CsrT::edge_values[e];
            if (fabs(cap) <= 1e-12)
            continue;
            Traits::edge_descriptor e1, e2;
            bool in1, in2;
            tie(e1, in1) = add_edge(verts[x], verts[y], boost_graph);
            tie(e2, in2) = add_edge(verts[y], verts[x], boost_graph);
            if (!in1 || !in2){
            debug_aml("error");
            return -1;
            }
            capacity[e1] = cap;
            capacity[e2] = 0;
            rev[e1] = e2;
            rev[e2] = e1;
        }
    }
  
    //
    // Perform Boost reference
    //

    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    maxflow = edmonds_karp_max_flow(boost_graph, source, sink); 
    cpu_timer.Stop();
    elapsed = cpu_timer.ElapsedMillis();

    //
    // Extracting results on CPU
    //

    std::vector<std::vector<ValueT>> boost_flow; 
    boost_flow.resize(graph.nodes);
    for (auto x = 0; x < graph.nodes; ++x) 
	boost_flow[x].resize(graph.nodes, 0.0);
    typename graph_traits<Graph>::vertex_iterator u_it, u_end;
    typename graph_traits<Graph>::out_edge_iterator e_it, e_end;
    for (tie(u_it, u_end) = vertices(boost_graph); u_it != u_end; ++u_it){
        for (tie(e_it, e_end) = out_edges(*u_it, boost_graph); e_it != e_end; 
            ++e_it){
            if (capacity[*e_it] > 0){
            ValueT e_f = capacity[*e_it] - residual_capacity[*e_it];
                VertexT t = target(*e_it, boost_graph);
                //debug_aml("flow on edge %d - %d = %lf", *u_it, t, e_f);
                boost_flow[*u_it][t] = e_f;
            }
        }
        }
        for (auto x = 0; x < graph.nodes; ++x){
        auto e_start = graph.CsrT::GetNeighborListOffset(x);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            VertexT y = graph.CsrT::GetEdgeDest(e);
            flow[e] = boost_flow[x][y];
        }
    }

#else

    debug_aml("no boost");

    debug_aml("graph nodes %d, edges %d source %d sink %d src %d", 
	    graph.nodes, graph.edges, src, sin);

    ValueT*   excess =  (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
    VertexT*  height = (VertexT*)malloc(sizeof(VertexT)*graph.nodes);
    for (VertexT v = 0; v < graph.nodes; ++v){
        excess[v] = (ValueT)0;
        height[v] = (VertexT)0;
    }
     
    for (SizeT e = 0; e < graph.edges; ++e){
        flow[e] = (ValueT) 0;
    }

#if MF_DEBUG
    debug_aml("before relabeling");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }
#endif
    relabeling(graph, src, sin, height, reverse, flow);

#if MF_DEBUG
    debug_aml("after relabeling");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }
#endif

    //
    // Compute the preflow
    //
    auto e_start = graph.CsrT::GetNeighborListOffset(src);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
    auto e_end = e_start + num_neighbors;

    ValueT preflow = (ValueT) 0;
    for (SizeT e = e_start; e < e_end; ++e)
    {
        auto y = graph.CsrT::GetEdgeDest(e);
        auto c = graph.CsrT::edge_values[e];
        excess[y] += c;
        flow[e] = c;
        flow[reverse[e]] = -c;
        preflow += c;
    }

#if MF_DEBUG
    debug_aml("after preflow");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }

    {
        auto e_start = graph.CsrT::GetNeighborListOffset(src);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
        auto e_end = e_start + num_neighbors;
        for (SizeT e = e_start; e < e_end; ++e)
        {
            auto y = graph.CsrT::GetEdgeDest(e);
            debug_aml("height[%d] = %d", y, height[y]);
        }
        for (int i=0; i<graph.nodes; ++i){
            debug_aml("excess[%d] = %lf\n", i, excess[i]);
        }
    }
#endif

    //
    // Perform simple max flow reference
    //
    debug_aml("perform simple max flow reference");
    debug_aml("source %d, sink %d", src, sin);
    debug_aml("source excess %lf, sink excess %lf", excess[src], excess[sin]);
    debug_aml("pre flow push from source %lf", preflow);
    debug_aml("source height %d, sink height %d", height[src], height[sin]);
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    maxflow = max_flow(graph, flow, excess, height, src, sin, reverse);
   
    cpu_timer.Stop();
    elapsed = cpu_timer.ElapsedMillis();

    free(excess);
    free(height);

#endif
    
    return elapsed;
}

template <typename T>
__forceinline__ bool ToTrack(const T &v)
{
    //int num_targets=3;
    //T targets[] = {9, 3214, 2658};

    //for (auto i = 0; i < num_targets; i++)
    //    if (v == targets[i])
    //        return true;
    return false;
    //return true;
}

/**
 * @brief OpenMP based MF implementations
 *
 * @tparam GraphT   Type of the graph
 * @tparam VertexT  Type of the vertex
 * @tparam ValueT   Type of the capacity/flow/excess
 * @param[in]  parameters Running parameters
 * @param[in]  graph      Input graph
 * @param[in]  src        The source vertex
 * @param[in]  sin        The sink vertex
 * @param[out] maxflow	  Value of computed maxflow reached sink
 * @param[out] reverse	  Computed reverse
 * @param[out] edges_flow Computed flows on edges
 *
 * \return     double      Time taken for the MF
 */
template <typename VertexT, typename ValueT, typename GraphT>
double OMP_Reference(
	util::Parameters &parameters,
	GraphT  &graph,
	VertexT  source,
	VertexT  sink,
	ValueT  &maxflow,
	VertexT *reverses,
	ValueT  *flows,
    typename GraphT::SizeT &iterations)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CsrT  CsrT;

    bool     iter_stats          = parameters.template Get<bool    >("iter-stats");
    int      num_threads         = parameters.template Get<int     >("omp-threads");
    bool     quiet               = parameters.template Get<bool    >("quiet");
    uint64_t max_iter            = parameters.template Get<uint64_t>("max-iter");
    bool     merge_push_relabel  = parameters.template Get<bool    >("merge-push-relabel"); 
    uint64_t relabeling_interval = parameters.template Get<uint64_t>("relabeling-interval");
    bool     use_active_vertices = parameters.template Get<bool    >("active-vertices");

    util::CpuTimer cpu_timer;
    auto     nodes    = graph.nodes;
    auto     edges    = graph.edges;
    auto    &capacities = graph.CsrT::edge_values;
    ValueT  *excesses            = new ValueT [nodes];
    ValueT  *org_excesses        = new ValueT [nodes];
    VertexT *heights             = new VertexT[nodes]; 
    VertexT *next_heights        = new VertexT[nodes];
    bool    *pusheds             = new bool   [nodes];   
    VertexT *active_vertices     = NULL;
    VertexT *next_active_vertices= NULL;
    VertexT *active_markers      = NULL;
    SizeT    total_num_active_vertices = 0;
    SizeT    total_num_e_visited = 0;
    SizeT    total_num_valid_e   = 0;
    SizeT    total_num_pushes    = 0;
    SizeT    total_num_s_pushes  = 0;
    SizeT    total_num_relabels  = 0;
    SizeT    num_active_vertices = nodes;
    SizeT   *s_num_e_visited     = new SizeT[num_threads];
    SizeT   *s_num_valid_e       = new SizeT[num_threads];
    SizeT   *s_num_pushes        = new SizeT[num_threads];
    SizeT   *s_num_s_pushes      = new SizeT[num_threads];
    SizeT   *s_num_relabels      = new SizeT[num_threads];

    // Init
    for (VertexT v = 0; v < nodes; v++)
    {
        excesses[v] = 0;
        heights [v] = 0;
    }
    for (SizeT e = 0; e < edges; e++)
        flows[e] = 0;
    SizeT src_degree = graph.CsrT::GetNeighborListLength(source);
    SizeT src_offset = graph.CsrT::GetNeighborListOffset(source);
    for (SizeT e = src_offset; e < src_offset + src_degree; e++)
    {
        ValueT capacity = capacities[e];
        excesses[graph.CsrT::GetEdgeDest(e)] += capacity;
        flows[e] = capacity;
        flows[reverses[e]] = -capacity;
    }
    if (util::isValid(relabeling_interval))
        relabeling(graph, source, sink, heights, reverses, flows);
    else
        heights[source] = nodes;

    if (use_active_vertices)
    {
        active_vertices     = new VertexT[nodes];
        next_active_vertices= new VertexT[nodes];
        active_markers      = new VertexT[nodes];
        num_active_vertices = src_degree;
        for (SizeT e = src_offset; e < src_offset + src_degree; e++)
            active_vertices[e - src_offset] = graph.CsrT::GetEdgeDest(e);
        for (VertexT v = 0; v < nodes; v++)
            active_markers[v] = 0;
    }

    if (merge_push_relabel)
    {
        for (VertexT v = 0; v < nodes; v++)
            next_heights[v] = util::PreDefinedValues<VertexT>::MaxValue;
    }

    if (iter_stats)
        util::PrintMsg("Iter\t # active vertices\t #edges visited\t #valid edges\t "
            "#pushes\t #sturated pushes\t #relabels\t excesses[sink]\t "
            "#reachable vertices\t sink reachable", !quiet);

    iterations  = 0;
    bool  has_update = true;
    cpu_timer.Start();        
    while (has_update)
    {
        has_update = false;
        SizeT num_e_visited = 0;
        SizeT num_valid_e   = 0;
        SizeT num_pushes    = 0;
        SizeT num_s_pushes  = 0;
        SizeT num_relabels  = 0;
        SizeT next_num_active_vertices = 0;
        //#pragma omp parallel for num_threads(num_threads) \
        //    reduction(+:num_e_visited, num_valid_e, num_pushes, num_s_pushes, num_relabels)
        //for (SizeT i = 0; i < num_active_vertices; i++)
        #pragma omp parallel num_threads(num_threads)
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            SizeT t_num_e_visited = 0;
            SizeT t_num_valid_e   = 0;
            SizeT t_num_pushes    = 0;
            SizeT t_num_s_pushes  = 0;
            SizeT t_num_relabels  = 0;

            VertexT pos_start = num_active_vertices / num_threads * thread_num;
            VertexT pos_end   = num_active_vertices / num_threads * (thread_num + 1);
            if (thread_num == 0)
                pos_start = 0;
            if (thread_num + 1 == num_threads)
                pos_end = num_active_vertices;

            if (use_active_vertices)
            {
                for (VertexT pos = pos_start; pos < pos_end; pos++)
                    active_markers[active_vertices[pos]] = 0;
            }

            if (merge_push_relabel)
            {
                for (VertexT pos = pos_start; pos < pos_end; pos++)
                {
                    VertexT v = (use_active_vertices) ? active_vertices[pos] : pos;
                    next_heights[v] 
                        = util::PreDefinedValues<VertexT>::MaxValue;
                    //org_excesss[v] = excesses[v];
                }
            }
            
            #pragma omp barrier
            //if (thread_num == 0)
            //    util::PrintMsg(std::to_string(thread_num) + " Barrier 0");

            
            for (VertexT pos = pos_start; pos < pos_end; pos++)
            {
                VertexT v = (use_active_vertices) ? active_vertices[pos] : pos;
                if (v == source || v == sink)
                    continue;
                auto excess = excesses[v];
                if (merge_push_relabel && !use_active_vertices)
                    org_excesses[v] = excess;
                if (ToTrack(v))
                    util::PrintMsg(std::to_string(thread_num) + " " + std::to_string(v)
                        + " : excess = " + std::to_string(excess)
                        + ", height = " + std::to_string(heights[v]));
                if (!use_active_vertices && excess < MF_EPSILON)
                    continue;
                
                VertexT min_height = util::PreDefinedValues<VertexT>::MaxValue;
                bool pushed = false;
                SizeT e_start  = graph.CsrT::GetNeighborListOffset(v);
                SizeT v_degree = graph.CsrT::GetNeighborListLength(v);
                SizeT e_end    = e_start + v_degree;
                auto  height   = heights[v];
                
                for (SizeT e = e_start; e < e_end; e++)
                {
                    t_num_e_visited ++;
                    VertexT u = graph.CsrT::GetEdgeDest(e);
                    ValueT move = capacities[e] - flows[e];
                    if (move < MF_EPSILON)
                        continue;

                    t_num_valid_e ++;
                    
                    auto height_u = heights[u];
                    if (height <= height_u)
                    {
                        if (min_height > height_u)
                            min_height = height_u;
                        continue;
                    }

                    if (move > excess)
                        move = excess;
                    //if (move < MF_EPSILON)
                    //    continue;

                    if (ToTrack(v) || ToTrack(u))
                        util::PrintMsg(std::to_string(thread_num) + " Pushing " + std::to_string(move)
                            + " from " + std::to_string(v) 
                            + " to " + std::to_string(u));                
                    t_num_pushes ++; 
                    #pragma omp atomic
                    //{
                        excesses[v] -= move;
                    //}
                    #pragma omp atomic
                    //{
                        excesses[u] += move;
                    //}
                    #pragma omp atomic
                    //{
                        flows[e] += move;
                    //}
                    auto reverse_e = reverses[e];
                    if (merge_push_relabel)
                    {
                        ValueT pervious_flow = 0;
                        #pragma omp atomic capture
                        {
                            pervious_flow = flows[reverse_e];
                            flows[reverse_e] -= move;
                        }
                        if (capacities[reverse_e] - pervious_flow < MF_EPSILON)
                        {
                            VertexT old_height = 0;
                            VertexT new_height = height + 1;
                            do {
                                #pragma omp atomic capture
                                {
                                    old_height = next_heights[u]; 
                                    next_heights[u] = new_height;
                                }
                                if (old_height < new_height)
                                {
                                    new_height = old_height;
                                } else 
                                    break;
                            } while (true);
                        }
                    } else {
                        #pragma omp atomic
                        flows[reverse_e] -= move;
                    }

                    if (use_active_vertices)
                    {
                        VertexT pervious_marker = 0;
                        #pragma omp atomic capture
                        {
                            pervious_marker = active_markers[u];
                            active_markers[u] ++;
                        }
                        if (pervious_marker == 0)
                        {
                            VertexT pos_ = 0;
                            #pragma omp atomic capture
                            {
                                pos_ = next_num_active_vertices;
                                next_num_active_vertices ++;
                            }
                            next_active_vertices[pos_] = u;
                        }
                    }
                    pushed = true;
                    if (capacities[e] - move < MF_EPSILON)
                        t_num_s_pushes ++;
                    excess -= move;
                    if (excess < MF_EPSILON)
                        break;
                } // end of for e

                if (excess > MF_EPSILON && use_active_vertices)
                {
                    VertexT pervious_marker = 0;
                    #pragma omp atomic capture
                    {
                        pervious_marker = active_markers[v];
                        active_markers[v] ++;
                    }
                    if (pervious_marker == 0)
                    {
                        VertexT pos_ = 0;
                        #pragma omp atomic capture
                        {
                            pos_ = next_num_active_vertices;
                            next_num_active_vertices ++;
                        }
                        next_active_vertices[pos_] = v;
                    }
                }

                pusheds[v] = pushed;
                if (pushed)
                {
                    has_update = true;
                } else if (merge_push_relabel)
                {
                    if (min_height != util::PreDefinedValues<VertexT>::MaxValue &&
                        heights[v] <= min_height)
                    {
                        if (ToTrack(v))
                            util::PrintMsg(std::to_string(thread_num) + " Relabeling0 " + std::to_string(v)
                                + " to " + std::to_string(min_height + 1));
                        //heights[v] = min_height + 1;
                        //#pragma omp atomic
                            //next_heights[v] = min(next_heights[v], min_height + 1);
                        VertexT old_height = 0;
                        VertexT new_height = height + 1;
                        do {
                            #pragma omp atomic capture
                            {
                                old_height = next_heights[v]; 
                                next_heights[v] = new_height;
                            }
                            if (old_height < new_height)
                            {
                                new_height = old_height;
                            } else 
                                break;
                        } while (true);

                        has_update = true;
                        t_num_relabels ++;
                    }
                } 
            } // end of for pos
            
            //#pragma omp atomic
            //num_e_visited += t_num_e_visited;
            s_num_e_visited[thread_num] = t_num_e_visited;
            //#pragma omp atomic
            //num_valid_e += t_num_valid_e;
            s_num_valid_e  [thread_num] = t_num_valid_e;
            //#pragma omp atomic
            //num_pushes += t_num_pushes;
            s_num_pushes   [thread_num] = t_num_pushes;
            //#pragma omp atomic
            //num_s_pushes += t_num_s_pushes;
            s_num_s_pushes [thread_num] = t_num_s_pushes;
            //#pragma omp atomic
            //num_relabels += t_num_relabels;
            s_num_relabels [thread_num] = t_num_relabels;
        }

        //util::PrintMsg("Barrier 1");
        #pragma omp parallel num_threads(num_threads)
        {
            int thread_num = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            SizeT t_num_e_visited = 0;
            SizeT t_num_valid_e   = 0;
            SizeT t_num_pushes    = 0;
            SizeT t_num_s_pushes  = 0;
            SizeT t_num_relabels  = 0;

            VertexT pos_start = num_active_vertices / num_threads * thread_num;
            VertexT pos_end   = num_active_vertices / num_threads * (thread_num + 1);
            if (thread_num == 0)
                pos_start = 0;
            if (thread_num + 1 == num_threads)
                pos_end = num_active_vertices;

            //util::PrintMsg(std::to_string(thread_num) + " Waiting 1");

            //#pragma omp barrier
            //if (thread_num == 0)
            //    util::PrintMsg(std::to_string(thread_num) + " Barrier 1");

            if (merge_push_relabel)
            {
                for (VertexT pos = pos_start; pos < pos_end; pos++)
                {
                    VertexT v = (use_active_vertices) ? active_vertices[pos] : pos;
  
                    if (v == source || v == sink)
                        continue;
                    if (!use_active_vertices)
                    {
                        auto excess = org_excesses[v];
                        if (excess < MF_EPSILON)
                            continue;
                    }
                    if (pusheds[v])
                        continue;

                    heights[v] = next_heights[v];
                    if (use_active_vertices)
                    {
                        active_markers[v] ++;
                        if (active_markers[v] == 1)
                        {
                            VertexT pos_ = 0;
                            #pragma omp atomic capture
                            {
                                pos_ = next_num_active_vertices;
                                next_num_active_vertices ++;
                            }
                            next_active_vertices[pos_] = v;
                        }
                    }
                }
            } else {
                //#pragma omp parallel for num_threads(num_threads) reduction(+:num_relabels, num_e_visited)
                //for (SizeT i = 0; i < num_active_vertices; i++)
                //{
                //    VertexT v = (active_vertices == NULL) ? i : active_vertices[i];

                for (VertexT pos = pos_start; pos < pos_end; pos++)
                {
                    VertexT v = (use_active_vertices) ? active_vertices[pos] : pos;
  
                    if (v == source || v == sink)
                        continue;
                    auto excess = excesses[v];
                    if (excess < MF_EPSILON)
                        continue;
                    if (pusheds[v])
                        continue;
     
                    VertexT min_height = util::PreDefinedValues<VertexT>::MaxValue;
                    SizeT e_start  = graph.CsrT::GetNeighborListOffset(v);
                    SizeT v_degree = graph.CsrT::GetNeighborListLength(v);
                    SizeT e_end    = e_start + v_degree;
                    auto  height   = heights[v];
                    
                    for (SizeT e = e_start; e < e_end; e++)
                    {
                        num_e_visited ++;
                        ValueT move = capacities[e] - flows[e];
                        if (move < MF_EPSILON)
                            continue;
     
                        VertexT u = graph.CsrT::GetEdgeDest(e);
                        auto height_u = heights[u];
                        if (min_height > height_u)
                            min_height = height_u; 
                    }
                    if (min_height != util::PreDefinedValues<VertexT>::MaxValue &&
                        min_height >= heights[v])
                    {
                        if (ToTrack(v))
                            util::PrintMsg(std::to_string(thread_num) + " Relabeling1 " + std::to_string(v)
                                + " to " + std::to_string(min_height + 1));
                        heights[v] = min_height + 1;
                        has_update = true;
                        t_num_relabels ++;
            
                        if (use_active_vertices)
                        {            
                            active_markers[v] ++;
                            if (active_markers[v] == 1)
                            {
                                VertexT pos_ = 0;
                                #pragma omp atomic capture
                                {
                                    pos_ = next_num_active_vertices;
                                    next_num_active_vertices ++;
                                }
                                next_active_vertices[pos_] = v;
                            }
                        }
                    }
                }
            }

            //#pragma omp atomic
            //num_e_visited += t_num_e_visited;
            s_num_e_visited[thread_num] += t_num_e_visited;
            //#pragma omp atomic
            //num_valid_e += t_num_valid_e;
            s_num_valid_e  [thread_num] += t_num_valid_e;
            //#pragma omp atomic
            //num_pushes += t_num_pushes;
            s_num_pushes   [thread_num] += t_num_pushes;
            //#pragma omp atomic
            //num_s_pushes += t_num_s_pushes;
            s_num_s_pushes [thread_num] += t_num_s_pushes;
            //#pragma omp atomic
            //num_relabels += t_num_relabels;
            s_num_relabels [thread_num] += t_num_relabels;
        }

        for (int t = 0; t < num_threads; t++)
        {
            num_e_visited += s_num_e_visited[t];
            num_valid_e   += s_num_valid_e  [t];
            num_pushes    += s_num_pushes   [t];
            num_s_pushes  += s_num_s_pushes [t];
            num_relabels  += s_num_relabels [t];
        }
        if (iter_stats)
        {
            util::PrintMsg(
                  std::to_string(iterations) + "\t"
                + std::to_string(num_active_vertices) + "\t"
                + std::to_string(num_e_visited) + "\t"
                + std::to_string(num_valid_e) + "\t"
                + std::to_string(num_pushes) + "\t"
                + std::to_string(num_s_pushes) + "\t"
                + std::to_string(num_relabels) + "\t"
                + std::to_string(excesses[sink]), !quiet);
        } 
        iterations ++;
        total_num_active_vertices += num_active_vertices;
        total_num_e_visited += num_e_visited;
        total_num_valid_e += num_valid_e;
        total_num_pushes += num_pushes;
        total_num_s_pushes += num_s_pushes;
        total_num_relabels += num_relabels;
        if (use_active_vertices)
        {
            auto temp = next_active_vertices;
            next_active_vertices = active_vertices;
            active_vertices = temp;
            num_active_vertices = next_num_active_vertices;
            next_num_active_vertices = 0;
        }

        if (util::isValid(max_iter) && iterations > max_iter)
            break;
        if ((iterations % relabeling_interval) == 0)
            relabeling(graph, source, sink, heights, reverses, flows);
    }
    cpu_timer.Stop();
    util::PrintMsg(
          "Total\t"
        + std::to_string(total_num_active_vertices) + "\t"
        + std::to_string(total_num_e_visited) + "\t"
        + std::to_string(total_num_valid_e) + "\t"
        + std::to_string(total_num_pushes) + "\t"
        + std::to_string(total_num_s_pushes) + "\t"
        + std::to_string(total_num_relabels) + "\t"
        + std::to_string(excesses[sink]), !quiet);

    maxflow = excesses[sink];
    delete[] excesses       ; excesses        = NULL;
    delete[] org_excesses   ; org_excesses    = NULL;
    delete[] heights        ; heights         = NULL;
    delete[] next_heights   ; next_heights    = NULL;
    delete[] pusheds        ; pusheds         = NULL;
    delete[] active_vertices; active_vertices = NULL;
    delete[] next_active_vertices; next_active_vertices = NULL;
    delete[] active_markers ; active_markers  = NULL;
    delete[] s_num_e_visited; s_num_e_visited = NULL;
    delete[] s_num_valid_e  ; s_num_valid_e   = NULL;
    delete[] s_num_pushes   ; s_num_pushes    = NULL;
    delete[] s_num_s_pushes ; s_num_s_pushes  = NULL;
    delete[] s_num_relabels ; s_num_relabels  = NULL;
    return cpu_timer.ElapsedMillis();
}

/**
 * @brief Validation of MF results
 *
 * @tparam     GraphT	      Type of the graph
 * @tparam     ValueT	      Type of the distances
 *
 * @param[in]  parameters     Excution parameters
 * @param[in]  graph	      Input graph
 * @param[in]  source	      The source vertex
 * @param[in]  sink           The sink vertex
 * @param[in]  h_flow	      Computed flow on edges 
 * @param[in]  ref_flow	      Reference flow on edges
 * @param[in]  verbose	      Whether to output detail comparsions
 *
 * \return     int  Number of errors
 */
template <typename GraphT, typename ValueT, typename VertexT>
uint64_t Validate_Results(
    util::Parameters  &parameters,
    GraphT		  &graph,
    VertexT		  source,
    VertexT		  sink,
    ValueT		  *h_flow,
    VertexT		  *reverse,
    int           *min_cut,
    ValueT		  *ref_flow = NULL,
    bool		  verbose = true)
{
    typedef typename GraphT::CsrT   CsrT;
    typedef typename GraphT::SizeT  SizeT;  

    uint64_t num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    bool quick = parameters.Get<bool>("quick");
    auto nodes = graph.nodes;

    ValueT flow_incoming_sink = (ValueT)0;
    for (auto u = 0; u < graph.nodes; ++u){
        auto e_start = graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            auto v = graph.CsrT::GetEdgeDest(e);
            if (v != sink)
                continue;
            auto flow_e_in = h_flow[e];
            //printf("flow(%d->%d) = %lf (incoming sink)\n", u, v, flow_e_in);
            if (util::isValid(flow_e_in))
                flow_incoming_sink += flow_e_in;
        }
    }
    util::PrintMsg("Max Flow = " + std::to_string(flow_incoming_sink), !quiet);

    if (min_cut != NULL)
    {
        util::PrintMsg("Min cut validity: ", !quiet, false);
        // Verify min cut h_flow
        ValueT mincut_flow = (ValueT)0;
        for (auto u = 0; u < graph.nodes; ++u){
            if (min_cut[u] == 1){
                auto e_start = graph.CsrT::GetNeighborListOffset(u);
                auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
                auto e_end = e_start + num_neighbors;
                for (auto e = e_start; e < e_end; ++e){
                    auto v = graph.CsrT::GetEdgeDest(e);
                    if (min_cut[v] == 0){
                        auto f = graph.CsrT::edge_values[e];
                        mincut_flow += f;
                    }
                }
            }
        }
        if (fabs(mincut_flow - flow_incoming_sink) > MF_EPSILON_VALIDATE)
        {
            ++num_errors;
            util::PrintMsg("FAIL: Min cut " + std::to_string(mincut_flow) +
                    " and max flow " + std::to_string(flow_incoming_sink) + 
                    " are not equal", !quiet);
        } else {
            util::PrintMsg("PASS!", !quiet);
        }
        util::PrintMsg("MIN CUT flow = " + std::to_string(mincut_flow));
    }

    // Verify the result
    /*if (!quick)
    {
        util::PrintMsg("Flow Validity (compare results):\n", !quiet, false);

        auto ref_flow_incoming_sink = (ValueT)0;
        for (auto u = 0; u < graph.nodes; ++u){
            auto e_start = graph.CsrT::GetNeighborListOffset(u);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; ++e){
                auto v = graph.CsrT::GetEdgeDest(e);
                if (v != sink)
                    continue;
                auto flow_e_in = ref_flow[e];
                if (util::isValid(flow_e_in))
                    ref_flow_incoming_sink += flow_e_in;
            }
        }

        if (fabs(flow_incoming_sink-ref_flow_incoming_sink) > MF_EPSILON_VALIDATE)
        {
            ++num_errors;
            debug_aml("flow_incoming_sink %lf, ref_flow_incoming_sink %lf", \
                    flow_incoming_sink, ref_flow_incoming_sink);
        }

        if (num_errors > 0)
        {
            util::PrintMsg(std::to_string(num_errors) + " errors occurred.", 
                    !quiet);
        }else
        {
            util::PrintMsg("PASS", !quiet);
        }
    }
    else
    {*/
        util::PrintMsg("Flow Validity: ", !quiet, false);

        for (VertexT v = 0; v < nodes; ++v)
        {
            if (v == source || v == sink)
                continue;
            auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
            ValueT flow_v = (ValueT) 0;
            for (auto e = e_start; e < e_end; ++e)
            {
                if (util::isValid(h_flow[e]))
                {
                    if (h_flow[e] > 0 && 
                        h_flow[e] - graph.CsrT::edge_values[e] > MF_EPSILON_VALIDATE)
                    {
                        num_errors ++;
                        if (num_errors == 1)
                            util::PrintMsg("FAIL: edge " + std::to_string(e)
                                + ", flow = " + std::to_string(h_flow[e])
                                + ", capacity = " + std::to_string(graph.CsrT::edge_values[e]));
                    } else 
                        flow_v += h_flow[e];
                } else {
                    ++num_errors;
                    debug_aml("flow for edge %d is invalid\n", e);
                }
            }
            if (fabs(flow_v) > MF_EPSILON_VALIDATE){
                debug_aml("Excess for vertex %d is %lf > %llf\n", 
                        v, fabs(flow_v), 1e-12);
            } else
                continue;
            ++num_errors;
            if (num_errors == 1)
                util::PrintMsg("FAIL: for vertex " + std::to_string(v) +
                    " excess " + std::to_string(flow_v) + 
                    " is not equal to 0", !quiet);
        }
        if (num_errors > 0)
        {
            util::PrintMsg(std::to_string(num_errors) + " errors occurred.", 
                    !quiet);
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    //}

    util::PrintMsg("Max Validity: ", !quiet, false);
    VertexT *active_vertices = new VertexT[graph.nodes];
    char    *vertex_markers  = new char   [graph.nodes];
    ValueT  *possible_flows  = new ValueT [graph.nodes];
    VertexT *vertex_parents  = new VertexT[graph.nodes];
    SizeT   *parent_edges    = new SizeT  [graph.nodes];
    for (VertexT v = 0; v < graph.nodes; v++)
    {
        vertex_markers[v] = 0;
        possible_flows[v] = 0;
    }
    VertexT head = 0, tail = 0;
    active_vertices[0] = source;
    possible_flows[source] = 1e20;
    vertex_markers[source] = 1;
    vertex_parents[source] = util::PreDefinedValues<VertexT>::InvalidValue;
    while (head >= tail)
    {
        VertexT v = active_vertices[tail];
        vertex_markers[v] = 2;
        tail ++;
        if (possible_flows[v] < MF_EPSILON_VALIDATE)
            continue;

        SizeT edge_start = graph.CsrT::GetNeighborListOffset(v);
        SizeT degree = graph.CsrT::GetNeighborListLength(v);
        SizeT edge_end = edge_start + degree;

        for (SizeT e = edge_start; e < edge_end; e++)
        {
            VertexT u = graph.CsrT::GetEdgeDest(e);
            if (vertex_markers[u] == 2)
                continue;
            ValueT residue = graph.CsrT::edge_values[e] - h_flow[e]; 
            if (residue > possible_flows[v])
                residue = possible_flows[v];
            if (residue > possible_flows[u])
            {
                possible_flows[u] = residue;
                vertex_parents[u] = v;
                parent_edges  [u] = e;
                if (vertex_markers[u] == 0)
                {
                    vertex_markers[u] = 1;
                    head ++;
                    active_vertices[head] = u;
                }
            }
        }
    }
    if (possible_flows[sink] > MF_EPSILON_VALIDATE)
    {
        util::PrintMsg("FAIL. Possible extra flow of "
            + std::to_string(possible_flows[sink])
            + " from source " + std::to_string(source)
            + " to sink " + std::to_string(sink), !quiet);
        VertexT v = sink;
        util::PrintMsg(std::to_string(v), !quiet, false);
        while (util::isValid(v) && v != source)
        {
            SizeT e = parent_edges[v];
            v = vertex_parents[v];
            util::PrintMsg(" <-(" + std::to_string(e)
                + ", " + std::to_string(graph.CsrT::edge_values[e])
                + ", " + std::to_string(h_flow[e])
                + ")- " + std::to_string(v), !quiet, false);
        }
        util::PrintMsg("", !quiet);
    } else 
        util::PrintMsg("PASS", !quiet);

    delete[] active_vertices; active_vertices = NULL;
    delete[] vertex_markers ; vertex_markers  = NULL;
    delete[] possible_flows ; possible_flows  = NULL;
    delete[] vertex_parents ; vertex_parents  = NULL;
    delete[] parent_edges   ; parent_edges    = NULL;
    if (!quick && verbose)
    {
        // Display Solution
        util::PrintMsg("Max Flow of the GPU result:");
        DisplaySolution(graph, h_flow, reverse, sink, graph.nodes);
        if (ref_flow != NULL)
        {
            util::PrintMsg("Max Flow of the CPU results:");
            DisplaySolution(graph, ref_flow, reverse, sink, graph.nodes);
        }
        util::PrintMsg("");
    }

    return num_errors;
}

} // namespace mf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

