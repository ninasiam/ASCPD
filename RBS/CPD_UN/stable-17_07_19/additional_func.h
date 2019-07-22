#ifndef ADDITIONAL_FUNC_H 
#define ADDITIONAL_FUNC_H
#include <time.h>
#include <sys/time.h>


double gettime(void) {
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	/* tv_sec : seconds, tv_usec : mikroseconds */
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

double tic(void){
	return gettime();
}

double toc(double start_timer){
	double stop_timer = gettime();
  double lap = stop_timer - start_timer;
	return lap;
}

#endif