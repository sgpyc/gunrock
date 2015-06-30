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

template <typename VertexId, typename SizeT>
__global__ void Assign_Marker_Forward(
    const SizeT            num_elements,
    const int              num_peers,
    const int              start_peer,
    const VertexId* const  keys_in,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    int gpu;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_peers)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        gpu = partition_table[key] - start_peer;
        if (gpu >=0 && gpu < num_peers)
            s_marker[gpu] = 1;
        //for (int i=0; i<num_peers; i++)
        //    s_marker[i][x] = (i==gpu)?1:0;
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename MakeOutArray>
__global__ void Assign_Marker(
    const typename MakeOutArray::Direction direction,
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
        if ((direction & MakeOutArray::Direction::FORWARD) != 0)
        {
            gpu = forward_partition[key] - start_peer;
            if (gpu >=0 && gpu < num_peers)
                s_marker[gpu][x] = 1;
        }

        if ((direction & MakeOutArray::Direction::BACKWARD) != 0)
        {
            for (SizeT i = backward_offsets[key]; i< backward_offsets[key+1]; i++)
            {
                gpu = backward_partition[i] - start_peer;
                if (gpu>=0 && gpu < num_peers)
                    s_marker[gpu][x] = 1;
            }
        }
    }
}

template <typename VertexId, typename SizeT>
__global__ void Assign_Marker_Backward(
    const SizeT            num_elements,
    const int              num_peers,
    const int              start_peer,
    const VertexId* const  keys_in,
    const SizeT*    const  offsets,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    int gpu = 0;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_peers)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        if (key!=-1) 
            for (SizeT i=offsets[key]; i<offsets[key+1]; i++)
            {
                gpu = partition_table[i] - start_peer;
                if (gpu >=0 && gpu < num_peers)
                  s_marker[gpu][x]=1;
            }
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out_Forward(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        int      target = partition_table[key];
        SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

        if (target==0)
        {
            s_keys_outs[0][pos]=key;
        } else {
            s_keys_outs[target][pos]=convertion_table[key];
            #pragma unrool
            for (int i=0;i<num_vertex_associates;i++)
                s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                    =s_vertex_associate_orgs[i][key];
            #pragma unrool
            for (int i=0;i<num_value__associates;i++)
                s_value__associate_outss[target*num_value__associates+i][pos]
                    =s_value__associate_orgs[i][key];
        }
        x+=STRIDE;
    }
}

template <typename MakeOutArray>
__global__ void Make_Out(MakeOutArray* d_array)
{
    typedef typename MakeOutArray::SizeT    SizeT   ;
    typedef typename MakeOutArray::VertexId VertexId;
    typedef typename MakeOutArray::Value    Value   ;

    __shared__ MakeOutArray s_array;
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = threadIdx.x;
    VertexId key = 0;
    int target_gpu = 0, host_gpu = 0;
    SizeT start_offset = 0, end_offset = 0;
    int *t_partition = NULL;
    SizeT *t_convertion = NULL;

    while (x<sizeof(MakeOutArray))
    {
        ((char*)&s_array)[x] = ((char*)d_array)[x];
            x+=blockDim.x;
    }
    __syncthreads();

    target_gpu = s_array.target_gpu;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    while ( x < s_array.num_elements )
    {
        key = s_array.keys_in[x];
        if (key < 0) 
        { // invalid key
            x+= STRIDE; continue;
        }
        if (s_array.markers[x] == s_array.markers[x+1])
        { // not marked for current GPU
            x+= STRIDE; continue;
        }
        host_gpu = s_array.forward_partition[key];
        if (host_gpu != 0 || target_gpu == 0)
        { // remote vertex or local vertex to local GPU => forward
            start_offset = key; end_offset = key+1;
            t_partition  = s_array.forward_partition;
            t_convertion = s_array.forward_convertion;
        } else { // local vertex to remote GPU => backward
            start_offset = s_array.backward_offset[key];
            end_offset   = s_array.backward_offset[key+1];
            t_partition  = s_array.backward_partition;
            t_convertion = s_array.backward_convertion;
        }
        for (SizeT j=start_offset; j<end_offset; j++)
        {
            if (target_gpu != t_partition[j]) continue;
            SizeT pos = s_array.markers[x];
            if (host_gpu == 0 && target_gpu == 0)
                s_array.keys_out[pos] = key;
            else s_array.keys_out[pos] = t_convertion[j];

            for (int i=0; i<s_array.num_vertex_associates; i++)
                s_array.vertex_outs[i][pos] = s_array.vertex_orgs[i][key];
            for (int i=0; i<s_array.num_value__associates; i++)
                s_array.value__outs[i][pos] = s_array.value__orgs[i][key];
        }
    }
}

template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out_Backward(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  SizeT*      const offsets,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        if (key <0) {x+=STRIDE; continue;}
        for (SizeT j=offsets[key];j<offsets[key+1];j++)
        {
            int      target = partition_table[j];
            SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

            if (target==0)
            {
                s_keys_outs[0][pos]=key;
            } else {
                s_keys_outs[target][pos]=convertion_table[j];
                #pragma unrool
                for (int i=0;i<num_vertex_associates;i++)
                    s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                        =s_vertex_associate_orgs[i][key];
                #pragma unrool
                for (int i=0;i<num_value__associates;i++)
                    s_value__associate_outss[target*num_value__associates+i][pos]
                        =s_value__associate_orgs[i][key];
            }
        }
        x+=STRIDE;
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
