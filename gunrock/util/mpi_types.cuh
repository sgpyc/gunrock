// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mpi_types.cuh
 *
 * @brief MPI type convertor
 */
#pragma once

#include <mpi.h>

namespace gunrock {
namespace util {

template <typename T>
MPI_Datatype Get_Mpi_Type()
{
    extern void UnSupportedType();
    UnSupportedType();
    return MPI_CHAR;
};

template <>
MPI_Datatype Get_Mpi_Type<signed char>()
{
    return MPI_CHAR;
};

template <>
MPI_Datatype Get_Mpi_Type<unsigned char>()
{
    return MPI_UNSIGNED_CHAR;
};

template <>
MPI_Datatype Get_Mpi_Type<signed short int>()
{
    return MPI_SHORT;
};

template <>
MPI_Datatype Get_Mpi_Type<unsigned short int>()
{
    return MPI_UNSIGNED_SHORT;
};

template <>
MPI_Datatype Get_Mpi_Type<signed int>()
{
    return MPI_INT;
};

/*template <>
MPI_Datatype Get_Mpi_Type<unsigned int>()
{
    return MPI_UNSIGNED_INT;
};*/

template <>
MPI_Datatype Get_Mpi_Type<signed long int>()
{
    return MPI_LONG;
};

template <>
MPI_Datatype Get_Mpi_Type<unsigned long int>()
{
    return MPI_UNSIGNED_LONG;
};

template <>
MPI_Datatype Get_Mpi_Type<signed long long int>()
{
    return MPI_LONG_LONG;
};

template <>
MPI_Datatype Get_Mpi_Type<unsigned long long int>()
{
    return MPI_UNSIGNED_LONG_LONG;
};

template <>
MPI_Datatype Get_Mpi_Type<float>()
{
    return MPI_FLOAT;
};

template <>
MPI_Datatype Get_Mpi_Type<double>()
{
    return MPI_DOUBLE;
};

template <>
MPI_Datatype Get_Mpi_Type<long double>()
{
    return MPI_LONG_DOUBLE;
};

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
