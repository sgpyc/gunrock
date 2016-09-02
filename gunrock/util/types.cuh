// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * types.cuh
 *
 * @brief data types and limits defination
 */

#pragma once



namespace gunrock {
namespace util {

// Ensure no un-specialized types will be compiled
extern __device__ __host__ void Error_UnsupportedType();

// Max values of each type
template <typename T>
__device__ __host__ __forceinline__ T MaxValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed char MaxValue<signed char>()
{
    return CHAR_MAX;
}

template <>
__device__ __host__ __forceinline__ unsigned char MaxValue<unsigned char>()
{
    return UCHAR_MAX;
}

template <>
__device__ __host__ __forceinline__ signed short int MaxValue<signed short int>()
{
    return SHRT_MAX;
}

template <>
__device__ __host__ __forceinline__ unsigned short int MaxValue<unsigned short int>()
{
    return USHRT_MAX;
}

template <>
__device__ __host__ __forceinline__ signed int MaxValue<signed int>()
{
    return INT_MAX;
}

template <>
__device__ __host__ __forceinline__ unsigned int MaxValue<unsigned int>()
{
    return UINT_MAX;
}

template <>
__device__ __host__ __forceinline__ signed long long int MaxValue<signed long long int>()
{
    return LLONG_MAX;
}

template <>
__device__ __host__ __forceinline__ unsigned long long int MaxValue<unsigned long long int>()
{
    return ULLONG_MAX;
}

// Min value of each type
template <typename T>
__device__ __host__ __forceinline__ T MinValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed char MinValue<signed char>()
{
    return SCHAR_MIN;
}

template <>
__device__ __host__ __forceinline__ unsigned char MinValue<unsigned char>()
{
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed short MinValue<signed short>()
{
    return SHRT_MIN;
}

template <>
__device__ __host__ __forceinline__ unsigned short MinValue<unsigned short>()
{
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed int MinValue<signed int>()
{
    return INT_MIN;
}

template <>
__device__ __host__ __forceinline__ unsigned int MinValue<unsigned int>()
{
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed long long int MinValue<signed long long int>()
{
    return LLONG_MIN;
}

template <>
__device__ __host__ __forceinline__ unsigned long long int MinValue<unsigned long long int>()
{
    return 0;
}


/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllZeros_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 4>()
{
    return (T)0x00000000;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 8>()
{
    return (T)0x0000000000000000;
}*/

// Allzeros for each type
template <typename T>
__device__ __host__ __forceinline__ T AllZeros()
{
    //return AllZeros_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed char AllZeros<signed char>()
{
    return (signed char)0x00;
}

template <>
__device__ __host__ __forceinline__ unsigned char AllZeros<unsigned char>()
{
    return (unsigned char)0x00U;
}

template <>
__device__ __host__ __forceinline__ signed short int AllZeros<signed short int>()
{
    return (signed short int)0x0000;
}

template <>
__device__ __host__ __forceinline__ unsigned short int AllZeros<unsigned short int>()
{
    return (unsigned short int)0x0000U;
}


template <>
__device__ __host__ __forceinline__ signed int AllZeros<signed int>()
{
    return (signed int)0x00000000;
}

template <>
__device__ __host__ __forceinline__ unsigned int AllZeros<unsigned int>()
{
    return (unsigned int)0x00000000U;
}

template <>
__device__ __host__ __forceinline__ signed long long int AllZeros<signed long long int>()
{
    return (signed long long int)0x0000000000000000LL;
}

template <>
__device__ __host__ __forceinline__ unsigned long long int AllZeros<unsigned long long int>()
{
    return (unsigned long long int)0x0000000000000000ULL;
}


/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllOnes_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 4>()
{
    return (T)0xFFFFFFFF;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 8>()
{
    return (T)0xFFFFFFFFFFFFFFFF;
}*/

template <typename T>
__device__ __host__ __forceinline__ T AllOnes()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ signed char AllOnes<signed char>()
{
    return (signed char)0xFF;
}

template <>
__device__ __host__ __forceinline__ unsigned char AllOnes<unsigned char>()
{
    return (unsigned char)0xFFU;
}

template <>
__device__ __host__ __forceinline__ signed short int AllOnes<signed short int>()
{
    return (signed short int)0xFFFF;
}

template <>
__device__ __host__ __forceinline__ unsigned short int AllOnes<unsigned short int>()
{
    return (unsigned short int)0xFFFFU;
}


template <>
__device__ __host__ __forceinline__ signed int AllOnes<signed int>()
{
    return (signed int)0xFFFFFFFF;
}

template <>
__device__ __host__ __forceinline__ unsigned int AllOnes<unsigned int>()
{
    return (unsigned int)0xFFFFFFFFU;
}

template <>
__device__ __host__ __forceinline__ signed long long int AllOnes<signed long long int>()
{
    return (signed long long int)0xFFFFFFFFFFFFFFFFLL;
}

template <>
__device__ __host__ __forceinline__ unsigned long long int AllOnes<unsigned long long int>()
{
    return (unsigned long long int)0xFFFFFFFFFFFFFFFFULL;
}


template <typename T>
__device__ __host__ __forceinline__ T InvalidValue()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int InvalidValue<int>()
{
    return (int)-1;
}

template <>
__device__ __host__ __forceinline__ long long InvalidValue<long long>()
{
    return (long long)-1;
}

template <typename T>
__device__ __host__ __forceinline__ bool isValid(T val)
{
    return val >= 0;//(val != InvalidValue<T>());
}

template <typename T, int SIZE>
struct VectorType{ /*typedef UnknownType Type;*/};
template <> struct VectorType<int      , 1> {typedef int       Type;};
template <> struct VectorType<int      , 2> {typedef int2      Type;};
template <> struct VectorType<int      , 3> {typedef int3      Type;};
template <> struct VectorType<int      , 4> {typedef int4      Type;};
template <> struct VectorType<long long, 1> {typedef long long Type;};
template <> struct VectorType<long long, 2> {typedef longlong2 Type;};
template <> struct VectorType<long long, 3> {typedef longlong3 Type;};
template <> struct VectorType<long long, 4> {typedef longlong4 Type;};

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
