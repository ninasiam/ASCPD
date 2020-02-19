#ifndef TIMERS_HPP
#define TIMERS_HPP

#include <time.h>
#include <sys/time.h>

/*--+ Get time in seconds +--*/
inline double gettime(void)
{
    struct timeval ttime;
    gettimeofday(&ttime, NULL);
    /* tv_sec : seconds, tv_usec : microseconds */
    return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

/*--+ Start timer +--*/
inline double tic(void)
{
    return gettime();
}

/*--+ Get specific lap time +--*/ 
inline double toc(double start_timer)
{
    double stop_timer = gettime();
    double lap = stop_timer - start_timer;
    return lap;
}

#endif