// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_kernel.cuh
 *
 * @brief kernel functions Base Graph Problem Enactor
 */

#pragma once

namespace gunrock {
namespace app {

template <typename SizeT1, typename SizeT2>
__global__ void Accumulate_Num (
    SizeT1 *num,
    SizeT2 *sum)
{
    sum[0]+=num[0];
}

template <typename VertexId, typename SizeT>
__global__ void Copy_Preds (
    const SizeT     num_elements,
    const VertexId* keys,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x*blockDim.x+threadIdx.x;
    VertexId t;

    while (x<num_elements)
    {
        t = keys[x];
        out_preds[t] = in_preds[t];
        x+= STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Update_Preds (
    const SizeT     num_elements,
    const SizeT     nodes,
    const VertexId* keys,
    const VertexId* org_vertexs,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x*blockDim.x + threadIdx.x;
    VertexId t, p;

    while (x<num_elements)
    {
        t = keys[x];
        p = in_preds[t];
        if (p<nodes) out_preds[t] = org_vertexs[p];
        x+= STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename MakeOutHandle>
__global__ void Assign_Marker(
    const typename MakeOutHandle::Direction direction,
    const SizeT            num_elements,
    const int              num_peers,
    const int              start_peer,
    const VertexId* const  keys_in,
    const int*      const  forward_partition,
    const SizeT*    const  backward_offsets,
    const int*      const  backward_partition,
          SizeT**          markers)
{
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
    VertexId key = 0;
    int gpu = 0;

    if (threadIdx.x < num_peers)
        s_marker[threadIdx.x] = markers[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        if (key < 0) { x+= STRIDE; continue;}
       
        for (gpu =0; gpu<num_peers; gpu++)
            s_marker[gpu][x] = 0; 
        if ((direction & MakeOutHandle::Direction::FORWARD) != 0)
        {
            gpu = forward_partition[key] - start_peer;
            if (gpu >=0 && gpu < num_peers)
                s_marker[gpu][x] = 1;
        }

        if ((direction & MakeOutHandle::Direction::BACKWARD) != 0)
        {
            for (SizeT i = backward_offsets[key]; i< backward_offsets[key+1]; i++)
            {
                gpu = backward_partition[i] - start_peer;
                if (gpu>=0 && gpu < num_peers)
                    s_marker[gpu][x] = 1;
            }
        }
        x += STRIDE;
    }
}

template <typename MakeOutHandle>
__global__ void Make_Out(MakeOutHandle* d_handle)
{
    typedef typename MakeOutHandle::SizeT    SizeT   ;
    typedef typename MakeOutHandle::VertexId VertexId;
    typedef typename MakeOutHandle::Value    Value   ;

    __shared__ MakeOutHandle s_handle;
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = threadIdx.x;
    VertexId key = 0;
    int target_gpu = 0, host_gpu = 0;
    SizeT start_offset = 0, end_offset = 0;
    int *t_partition = NULL;
    SizeT *t_convertion = NULL;

    while (x<sizeof(MakeOutHandle))
    {
        ((char*)&s_handle)[x] = ((char*)d_handle)[x];
        x+=blockDim.x;
    }
    __syncthreads();

    target_gpu = s_handle.target_gpu;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    while ( x < s_handle.num_elements )
    {
        key = s_handle.keys_in[x];
        if (key < 0) 
        { // invalid key
            x+= STRIDE; continue;
        }
        if (s_handle.markers[x] == s_handle.markers[x+1])
        { // not marked for current GPU
            x+= STRIDE; continue;
        }
        host_gpu = s_handle.forward_partition[key];
        if (host_gpu != 0 || target_gpu == 0)
        { // remote vertex or local vertex to local GPU => forward
            start_offset = key; end_offset = key+1;
            t_partition  = s_handle.forward_partition;
            t_convertion = s_handle.forward_convertion;
        } else { // local vertex to remote GPU => backward
            start_offset = s_handle.backward_offset[key];
            end_offset   = s_handle.backward_offset[key+1];
            t_partition  = s_handle.backward_partition;
            t_convertion = s_handle.backward_convertion;
        }
        for (SizeT j=start_offset; j<end_offset; j++)
        {
            if (target_gpu != t_partition[j]) continue;
            SizeT pos = s_handle.markers[x];
            if (host_gpu == 0 && target_gpu == 0)
                s_handle.keys_out[pos] = key;
            else s_handle.keys_out[pos] = t_convertion[j];

            for (int i=0; i<s_handle.num_vertex_associates; i++)
                s_handle.vertex_outs[i][pos] = s_handle.vertex_orgs[i][key];
            for (int i=0; i<s_handle.num_value__associates; i++)
                s_handle.value__outs[i][pos] = s_handle.value__orgs[i][key];
        }
    }
}

template <typename VertexId, typename SizeT>
__global__ void Mark_Queue (
    const SizeT     num_elements,
    const VertexId* keys,
          unsigned int* marker)
{
    VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    if (x< num_elements) marker[keys[x]]=1;
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
