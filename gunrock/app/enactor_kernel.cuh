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
    const int              gpu_num,
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
            {
                s_marker[gpu][x] = 1;
                if (to_track(gpu_num, key))
                    printf("%d\t %s\t forward: [%d] markers[%d][%d] -> 1\n", 
                        gpu_num, __func__, key, gpu, x);
            }
        }

        if ((direction & MakeOutHandle::Direction::BACKWARD) != 0)
        {
            for (SizeT i = backward_offsets[key]; i< backward_offsets[key+1]; i++)
            {
                gpu = backward_partition[i] - start_peer;
                if (gpu>=0 && gpu < num_peers)
                {
                    s_marker[gpu][x] = 1;
                    if (to_track(gpu_num, key))
                        printf("%d\t %s\t backward: [%d] markers[%d][%d] -> 1\n", 
                            gpu_num, __func__, key, gpu, x);
                }
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
    //const typename MakeOutHandle::Direction direction = d_handle->direction,
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
        if (x==0)
        {
            if (s_handle.markers[x] == 0)
            {
                x+= STRIDE; continue;
            }
        } else if (s_handle.markers[x] == s_handle.markers[x-1])
        { // not marked for current GPU
            x+= STRIDE; continue;
        }

        host_gpu = s_handle.forward_partition[key];
        //printf("x = %d, key = %d, host_gpu = %d, target_gpu = %d\n",
        //    x, key, host_gpu, target_gpu);
        if (host_gpu != 0 || target_gpu == 0) 
        { // remote vertex or local vertex to local GPU => forward
            start_offset = key; 
            end_offset   = key+1;
            t_partition  = s_handle.forward_partition;
            t_convertion = s_handle.forward_convertion;
            //printf("forward, x = %d, key = %d, host_gpu = %d, target_gpu = %d, start_offset = %d, end_offset = %d\n", x, key, host_gpu, target_gpu, start_offset, end_offset);
        } else { // local vertex to remote GPU => backward
            start_offset = s_handle.backward_offset[key];
            end_offset   = s_handle.backward_offset[key+1];
            t_partition  = s_handle.backward_partition;
            t_convertion = s_handle.backward_convertion;
            //printf("backward, x = %d, key = %d, host_gpu = %d, target_gpu = %d, start_offset = %d, end_offset = %d\n", x, key, host_gpu, target_gpu, start_offset, end_offset);
        }
        for (SizeT j=start_offset; j<end_offset; j++)
        {
            if (target_gpu != t_partition[j]) continue;
            SizeT pos = s_handle.markers[x] - 1;
            if (host_gpu == 0 && target_gpu == 0)
            {
                s_handle.keys_out[pos] = key;
                if (to_track(s_handle.gpu_num, key))
                    printf("%d\t %s\t [%d] %d\n",
                        s_handle.gpu_num, __func__, key, pos);
            } else {
                s_handle.keys_out[pos] = t_convertion[j];
                for (int i=0; i<s_handle.num_vertex_associates; i++)
                    s_handle.vertex_outs[i][pos] = s_handle.vertex_orgs[i][key];
                for (int i=0; i<s_handle.num_value__associates; i++)
                    s_handle.value__outs[i][pos] = s_handle.value__orgs[i][key];
                if (to_track(s_handle.gpu_num, key))
                    printf("%d\t %s\t [%d] -> [%d] %d,%d\n", 
                        s_handle.gpu_num, __func__, key, s_handle.keys_out[pos],
                        pos, s_handle.vertex_outs[0][pos]);
            }
        }
        x += STRIDE;
    }
}

/*template <typename VertexId, typename SizeT>
__global__ void Mark_Queue (
    const SizeT     num_elements,
    const VertexId* keys,
          unsigned int* marker)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements) 
    {
        marker[keys[x]]=1;
        x += STRIDE;
    }
}*/

template <typename VertexId, typename SizeT, typename Value>
__global__ void Check_Queue(
    const SizeT     num_elements,
    const int       gpu_num,
    const SizeT     num_nodes,
    const long long iteration,
    const VertexId* keys,
    const Value*    labels)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (key >= num_nodes || keys < 0)
            printf("%d\t %lld\t %s: x, key = %d, %d\n", gpu_num, iteration, __func__, x, key);
        else {
            Value label = labels[key];
            if ((label != iteration+1 && label != iteration)
              || label < 0)
            {
                printf("%d\t %lld\t %s: x, key, label = %d, %d, %d\n",
                    gpu_num, iteration, __func__, x, key, label);
            }
        }
        x += STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename Value>
__global__ void Check_Range(
    const SizeT num_elements,
    const int   gpu_num,
    const long long iteration,
    const Value lower_limit,
    const Value upper_limit,
    const Value* values)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        Value value = values[x];
        if (value > upper_limit || value < lower_limit)
        {
            printf("%d\t %lld\t %s: x = %d, %d not in (%d, %d)\n",
                gpu_num, iteration, __func__, x, value, lower_limit, upper_limit);
        }
        x += STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Check_Exist(
    const SizeT num_elements,
    const int   gpu_num,
    const int   check_num,
    const long long iteration,
    const VertexId* keys)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (to_track(gpu_num, key))
            printf("%d\t %lld\t %s: [%d] presents at %d\n",
                gpu_num, iteration, __func__, key, check_num);
        x += STRIDE;
    }
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
