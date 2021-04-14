#ifndef OMP_LIB_HPP
#define OMP_LIB_HPP

#include <omp.h>
#include "master_library.hpp"

/* -- Declare custom reduction of variable_type : MatrixXd -- */
#pragma omp declare reduction(sum                           \
                              : Eigen::MatrixXd             \
                              : omp_out = omp_out + omp_in) \
    initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))


/* -- Get number of threads, defined by enviromental variable $(OMP_NUM_THREADS). -- */
inline unsigned int get_num_threads(void)
{
    double threads;
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
    return threads;
}

#endif