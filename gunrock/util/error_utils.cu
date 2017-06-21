// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * error_utils.cu
 *
 * @brief Error handling utility routines
 */

#include <stdio.h>
#include <mpi.h>
#include <gunrock/util/error_utils.cuh>

void gunrock::util::PrintMsg(
    std::string msg)
{
    int gpu_idx, mpi_rank;
    cudaGetDevice(&gpu_idx);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    fprintf(stdout, "[rank %d, gpu %d] %s\n",
        mpi_rank, gpu_idx, msg.c_str());
    fflush(stdout);
}

void gunrock::util::PrintMsg(
    std::string msg, const char *filename, int line)
{
    int gpu_idx, mpi_rank;
    cudaGetDevice(&gpu_idx);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    fprintf(stdout, "[%s, %d @ rank %d, gpu %d] %s\n",
        filename, line, mpi_rank, gpu_idx, msg.c_str());
    fflush(stdout);
}

void gunrock::util::PrintErrorMsg(
    std::string msg, const char *filename, int line)
{
    int gpu_idx, mpi_rank;
    cudaGetDevice(&gpu_idx);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    fprintf(stderr, "[%s, %d @ rank %d, gpu %d] %s\n",
        filename, line, mpi_rank, gpu_idx, msg.c_str());
    fflush(stderr);
}

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t gunrock::util::GRError(
    cudaError_t error,
    const char *message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        PrintErrorMsg(std::string(message) + " (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        filename, line);
    }
    return error;
}

cudaError_t gunrock::util::GRError(
    cudaError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        PrintErrorMsg(message + " (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        filename, line);
    }
    return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t gunrock::util::GRError(
    const char *message,
    const char *filename,
    int line,
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        PrintErrorMsg(std::string(message) + " (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        filename, line);
    }
    return error;
}

cudaError_t gunrock::util::GRError(
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        PrintErrorMsg(message + " (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        filename, line);
    }
    return error;
}

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t gunrock::util::GRError(
    cudaError_t error,
    bool print)
{
    if (error && print) {
        PrintErrorMsg(" (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        "Unknown file", 0);
    }
    return error;
}

/**
 * Checks and resets last CUDA error.  If set, displays last error message in accordance with debug mode.
 */
cudaError_t gunrock::util::GRError(
    bool print)
{
    cudaError_t error = cudaGetLastError();
    if (error && print) {
        PrintErrorMsg(" (CUDA error "
            + std::to_string(error) + std::string(cudaGetErrorString(error)),
        "Unknown file", 0);
    }
    return error;
}

std::string gunrock::util::GetErrorString(gunrock::util::gunrockError_t error)
{
    switch (error) {
    case gunrock::util::GR_UNSUPPORTED_INPUT_DATA:
        return "unsupported input data";
        default:
        return "unknown error";
    }
}
gunrock::util::gunrockError_t gunrock::util::GRError(
    gunrock::util::gunrockError_t error,
    std::string message,
    const char *filename,
    int line,
    bool print)
{
    if (error && print) {
        PrintErrorMsg(message + " Gunrock error: " + GetErrorString(error),
            filename, line);
    }
    return error;
}
