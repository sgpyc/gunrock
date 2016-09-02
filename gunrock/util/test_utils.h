// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.h
 *
 * @brief Utility Routines for Tests
 */

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#elif defined(CLOCK_PROCESS_CPUTIME_ID)
    #include <sys/time.h>
#elif defined(BOOST_VERSION)
    #include <boost/timer/timer.hpp>
#else
    #include <sys/resource.h>
    #include <time.h>
#endif

#include <stdio.h>
//#include <math.h>
//#include <float.h>
//#include <cassert>
#include <map>
#include <string>
#include <vector>
//#include <stack>
#include <sstream>
#include <iostream>
#include <fstream>
//#include <algorithm>
//#include <utility>
//#include <gunrock/util/random_bits.h>
//#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>

// #include <gunrock/util/gitsha1.cpp>

namespace gunrock {
namespace util    {

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
private:
    int argc;
    char ** argv;

protected:
    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int _argc, char **_argv) : argc(_argc), argv(_argv)
    {
        int untyped_counter = 0;

        for (int i = 1; i < argc; i++)
        {
            std::string arg = std::string(argv[i]);
            std::string key, val;
            std::string::size_type pos = arg.find('=');

            if ((arg[0] != '-') || (arg[1] != '-'))
            { // untyped parameters
                val = arg;
                switch (untyped_counter)
                {
                case 0: key = "graph-type"; break;
                case 1: key = "graph-file"; break;
                default: printf("Unsupported parameter %s.\n", argv[i]);
                }
                untyped_counter ++;

            } else if (pos == std::string::npos)
            { // flag
                key = std::string(arg, 2, arg.length() - 2);
                val = "true";

            } else { // value
                key = std::string(arg, 2, pos - 2);
                val = std::string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool CheckCmdLineFlag(const char* arg_name)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {
            if (itr -> second == "false") return false;
            return true;
        }
        return false;
    }

    // Returns the value specified for a given commandline
    // parameter --<flag>=<value>
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {
            std::istringstream str_stream(itr->second);
            str_stream >> val;
        }
    }

    template <typename T>
    T GetCmdLineArgument(const char *arg_name)
    {
        std::map<std::string, std::string>::iterator itr;
        T val;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {
            std::istringstream str_stream(itr->second);
            str_stream >> val;
        }
        return val;
    }

    // Set value for a given commandline parameter
    template <typename T>
    void SetCmdLineArgument(const char *arg_name, T val)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        { // pervious record found
            itr -> second = std::to_string(val);
        } else {
            pairs[std::string(arg_name)] = std::to_string(val);
        }
    }

    // Returns the values specified for a given commandline
    // parameter --<flag>=<value>,<value>*
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        // Recover multi-value string
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end())
        {

            // Clear any default values
            vals.clear();

            std::string val_string = itr->second;
            std::istringstream str_stream(val_string);
            std::string::size_type old_pos = 0;
            std::string::size_type new_pos = 0;

            // Iterate comma-separated values
            T val;
            while ((new_pos = val_string.find(',', old_pos)) != std::string::npos)
            {

                if (new_pos != old_pos)
                {
                    str_stream.width(new_pos - old_pos);
                    str_stream >> val;
                    vals.push_back(val);
                }

                // skip over comma
                str_stream.ignore(1);
                old_pos = new_pos + 1;
            }

            // Read last value
            str_stream >> val;
            vals.push_back(val);
        }
    }

    // The number of pairs parsed
    int ParsedArgc()
    {
        return pairs.size();
    }

    std::string GetEntireCommandLine() const
    {
        std::string commandLineStr = "";
        for (int i = 0; i < argc; i++)
        {
            commandLineStr.append(std::string(argv[i]).append((i < argc - 1) ? " " : ""));
        }
        return commandLineStr;
    }

    template <typename T>
    void ParseArgument(const char *name, T &val)
    {
        if (CheckCmdLineFlag(name))
        {
            GetCmdLineArgument(name, val);
        }
    }

    /*char * GetCmdLineArgvGraphType()
    {
        char * graph_type = argv[1];
        return graph_type;
    }

    char * GetCmdLineArgvDataset()
    {
        char * market_filename;
        size_t graph_args = argc - pairs.size() - 1;
        market_filename =  (graph_args == 2) ? argv[2] : NULL;
        return market_filename;
    }*/

    /*char * GetCmdLineArgvQueryDataset()
    {
	char * market_fileName;
        size_t graph_args = argc - pairs.size() - 1;
	market_fileName = (graph_args>1) ? argv[2] : NULL;
	return market_fileName;
    }

    char * GetCmdLineArgvDataDataset()
    {
	char * market_fileName;
        size_t graph_args = argc - pairs.size() - 1;
	market_fileName = (graph_args>2) ? ((graph_args==5) ? argv[graph_args-1] : argv[graph_args]) : NULL;
	return market_fileName;
    }

    char * GetCmdLineArgvQueryLabel()
    {
	char * label_fileName;
        size_t graph_args = argc - pairs.size() - 1;
	label_fileName = (graph_args==5) ? argv[3] : NULL;
	return label_fileName;
    }

    char * GetCmdLineArgvDataLabel()
    {
        char * label_fileName;
        size_t graph_args = argc - pairs.size() - 1;
        label_fileName = (graph_args==5) ? argv[graph_args] : NULL;
        return label_fileName;
    }*/

    cudaError_t GetDeviceList(int* &gpu_idx, int &num_gpus)
    {
        cudaError_t retval = cudaSuccess;
        std::vector<int> temp_devices;
        if (CheckCmdLineFlag("device"))  // parse device list
        {
            GetCmdLineArguments<int>("device", temp_devices);
            num_gpus = temp_devices.size();
        }
        else  // use single device with index 0
        {
            num_gpus = 1;
            int t_gpu_idx;
            if (retval = util::GRError(cudaGetDevice(&t_gpu_idx),
                "cudaGetDevice failed", __FILE__, __LINE__))
                return retval;
            temp_devices.push_back(t_gpu_idx);
        }
        if (gpu_idx != NULL) delete[] gpu_idx;
        gpu_idx = new int[temp_devices.size()];
        for (std::size_t i=0; i<temp_devices.size(); i++)
            gpu_idx[i] = temp_devices[i];
        return retval;
    }
};

void DeviceInit(CommandLineArgs &args);
cudaError_t SetDevice(int dev);

class Statistic
{
    double mean;
    double m2;
    int count;

public:
    Statistic() : mean(0.0), m2(0.0), count(0) {}

    /**
     * @brief Updates running statistic, returning bias-corrected
     * sample variance.
     *
     * Online method as per Knuth.
     *
     * @param[in] sample
     * @returns Something
     */
    double Update(double sample)
    {
        count++;
        double delta = sample - mean;
        mean = mean + (delta / count);
        m2 = m2 + (delta * (sample - mean));
        return m2 / (count - 1);                //bias-corrected
    }
};

struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return (stop - start) * 1000;
    }

#elif defined(CLOCK_PROCESS_CPUTIME_ID)

    double start;
    double stop;

    void Start()
    {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        start = tv.tv_sec + 1.e-6 * tv.tv_usec;
    }

    void Stop()
    {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        stop = tv.tv_sec + 1.e-6 * tv.tv_usec;
    }

    double ElapsedMillis()
    {
        return 1000 * (stop - start);
    }

#elif defined(BOOST_VERSION)

    boost::timer::cpu_timer::cpu_timer cpu_t;

    void Start()
    {
        cpu_t.start();
    }

    void Stop()
    {
        cpu_t.stop();
    }

    float ElapsedMillis()
    {
        return cpu_t.elapsed().wall / 1000000.0;
    }
#else

    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;
        return (sec * 1000) + (usec / 1000);
    }

#endif
};

// Quite simple KeyValuePair struct for doing
// Key-value sorting according to keys
template<typename A, typename B>
struct KeyValuePair
{
    A Key;
    B Value;
    bool operator<(const KeyValuePair<A, B>& rhs)
    {
        return this->Key < rhs.Key;
    }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
