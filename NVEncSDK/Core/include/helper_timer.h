/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// Helper Timing Functions
#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H

// includes, project

// Definition of the StopWatch Interface, this is used if we don't want to use the CUT functions
// But rather in a self contained class interface
class StopWatchInterface
{
    public:
        StopWatchInterface() {};
        virtual ~StopWatchInterface() {};

    public:
        //! Start time measurement
        virtual void start() = 0;

        //! Stop time measurement
        virtual void stop() = 0;

        //! Reset time counters to zero
        virtual void reset() = 0;

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        virtual float getTime() = 0;

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        virtual float getAverageTime() = 0;
};


//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#if defined (WIN32) || defined (_WIN32)
// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

#include <vector> 
//! Windows specific implementation of StopWatch
class StopWatchWin : public StopWatchInterface
{
    public:
        //! Constructor, default
        StopWatchWin() :
            start_time(),     end_time(),
            diff_time(0.0f),  total_time(0.0f),
            running(false), clock_sessions(0), freq(0), freq_set(false)
        {
            if (! freq_set)
            {
                // helper variable
                LARGE_INTEGER temp;

                // get the tick frequency from the OS
                QueryPerformanceFrequency((LARGE_INTEGER *) &temp);

                // convert to type in which it is needed
                freq = ((double) temp.QuadPart) / 1000.0;

                // rememeber query
                freq_set = true;
            }
        };

        // Destructor
        ~StopWatchWin() { };

    public:
        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline float getTime();

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        inline float getAverageTime();

    private:
        // member variables

        //! Start of measurement
        LARGE_INTEGER  start_time;
        //! End of measurement
        LARGE_INTEGER  end_time;

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;

        //! tick frequency
        double  freq;

        //! flag if the frequency has been set
        bool  freq_set;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::start()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::stop()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &end_time);
    diff_time = (float)
                (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

    total_time += diff_time;
    clock_sessions++;
    running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running)
    {
        QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getTime()
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running)
    {
        LARGE_INTEGER temp;
        QueryPerformanceCounter((LARGE_INTEGER *) &temp);
        retval += (float)
                  (((double)(temp.QuadPart - start_time.QuadPart)) / freq);
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getAverageTime()
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}
#else
// Declarations for Stopwatch on Linux and Mac OSX
// includes, system
#include <sys/time.h>

//! Windows specific implementation of StopWatch
class StopWatchLinux : public StopWatchInterface
{
    public:
        //! Constructor, default
        StopWatchLinux() :
            start_time(), diff_time(0.0), total_time(0.0),
            running(false), clock_sessions(0)
        { };

        // Destructor
        virtual ~StopWatchLinux()
        { };

    public:
        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline float getTime();

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        inline float getAverageTime();

    private:

        // helper functions

        //! Get difference between start time and current time
        inline float getDiffTime();

    private:

        // member variables

        //! Start of measurement
        struct timeval  start_time;

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::start()
{
    gettimeofday(&start_time, 0);
    running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::stop()
{
    diff_time = getDiffTime();
    total_time += diff_time;
    running = false;
    clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running)
    {
        gettimeofday(&start_time, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getTime()
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running)
    {
        retval += getDiffTime();
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getAverageTime()
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getDiffTime()
{
    struct timeval t_time;
    gettimeofday(&t_time, 0);

    // time difference in milli-seconds
    return (float)(1000.0 * (t_time.tv_sec - start_time.tv_sec)
                   + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif // WIN32 or _WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkCreateTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkCreateTimer called object %08x\n", (void *)*timer_interface);
#if defined (WIN32) || defined (_WIN32)
    *timer_interface = (StopWatchInterface *)new StopWatchWin();
#else
    *timer_interface = (StopWatchInterface *)new StopWatchLinux();
#endif
    return (*timer_interface != NULL) ? true : false;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkDeleteTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkDeleteTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        delete *timer_interface;
        *timer_interface = NULL;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkStartTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkStartTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->start();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkStopTimer(StopWatchInterface **timer_interface)
{
    // printf("sdkStopTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->stop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkResetTimer(StopWatchInterface **timer_interface)
{
    // printf("sdkResetTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->reset();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
inline float
sdkGetAverageTimerValue(StopWatchInterface **timer_interface)
{
    //  printf("sdkGetAverageTimerValue called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        return (*timer_interface)->getAverageTime();
    }
    else
    {
        return 0.0f;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
inline float
sdkGetTimerValue(StopWatchInterface **timer_interface)
{
    // printf("sdkGetTimerValue called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        return (*timer_interface)->getTime();
    }
    else
    {
        return 0.0f;
    }
}

#endif // HELPER_TIMER_H
