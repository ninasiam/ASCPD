#ifndef SOLVE_ALS_HPP
#define SOLVE_ALS_HPP

#include "master_lib.hpp"
// #include "khatri_rao_product.hpp"
#include "FullKhatriRaoProduct.hpp"
#include "timers.hpp"
#include "mttkrp.hpp"
#include "updateFactor.hpp"
// #include "cpderr.hpp"
#include "compute_fval.hpp"
#include "omp_lib.hpp"
#include "normalize.hpp"
#include "lineSearchAccel.hpp"

/* 
 * void solveALSCPD(<IN>  const int                        tensor_order, 
 *                  <IN>  const Ref<const VectorXi>        tensor_dims, 
 *                  <IN>  const int                        tensor_rank,        
 *                  <IN>  const Ref<const VectorXi>        constraints,
 *                  <OUT> std::array<MatrixXd, TNS_ORDER>  &Init_Factors,
 *                  <OUT> std::array<MatrixXd, TNS_ORDER>  &True_Tensor_Mat,
 *                  <OUT> double                           &f_value,        
 *                  <IN>  int                              &iter, 
 *                  <IN>  const int                        MAX_ITER,     
 *                  <IN>  const double                     epsilon)
 * 
 * Description: Solves tensor factorization using CPD--ALS method.
 * 
 * param tensor_order    : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param constraints     : contains each factor's constraints,
 * param Init_Factors    : contains Initial Factors,
 * param True_Tensor_Mat : is the matricized output Tensor of type MatrixXd,
 * param f_value         : is the cost function's value
 * param iter            : is the CPD-ALS AO_iter,
 * param epsilon         : positive constant used as a stopping criterion.
 */
template <std::size_t TNS_SIZE>
void solveALSCPD(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank,
                 const Ref<const VectorXi> constraints,
                 std::array<MatrixXd, TNS_SIZE> &Init_Factors,
                 std::array<MatrixXd, TNS_SIZE> &True_Tensor_Mat,
                 double &f_value, int &iter, const int MAX_ITER, const double epsilon)
{
    // MatrixXd Est_Tensor_Mat = True_Tensor_Mat[tensor_order - 1];
    /*--+ Allocate memory for all Khatri-Rao Matrices  +--*/
    // std::array<MatrixXd, TNS_ORDER> Khatri_Rao;
    std::array<MatrixXd, TNS_SIZE> MTTKRP;
    /*--+ Allocate memory for all previous Factors (used in Acceleration) +--*/
    std::array<MatrixXd, TNS_SIZE> Prev_Factors;

    // Acceleration Variables.
    const int k_0 = 1; 
    double acc_coeff = 3;
    int    acc_fail = 0;

    for(int mode=0; mode<tensor_order; mode++)
    {
        int dim = tensor_dims.prod() / tensor_dims(mode);
        // Khatri_Rao[mode] = MatrixXd::Zero(tensor_rank, dim);
        MTTKRP[mode] = MatrixXd::Zero(tensor_dims(mode), tensor_rank);
    }

    double frob_X = True_Tensor_Mat[TNS_SIZE-1].squaredNorm();

    MatrixXd V(tensor_rank, tensor_rank);
    MatrixXd F_T_F(tensor_rank, tensor_rank);

    double start_t_mttkrp, start_t_compute_cpderr, start_t_update_factor, stop_t_mttkrp = 0, stop_t_compute_cpderr = 0, stop_t_update_factor = 0;
    double start_t_khatrirao, start_t_computeV, stop_t_khatrirao = 0, stop_t_computeV = 0;
    double start_t_normalize, start_t_accelerate, stop_t_normalize = 0, stop_t_accelerate = 0;
    
    /*--+ Print Initial Message  +--*/
    #ifdef PRINT_INFO
    std::cout << std::endl
              << " \t--\t--\t--\t--\t--\t--\t--\t--" << std::endl
              << " \t--\t\t BEGIN ALGORITHM \t\t\t--" << std::endl
              << " \t--\t--\t--\t--\t--\t--\t--\t--" << std::endl;
    #endif
    const unsigned int threads_num = get_num_threads();
    double start_t_cpd_als = tic();

    while (1)
    {
        for (int mode = 0; mode < tensor_order; mode++)
        {
            start_t_computeV = tic();
            V.setOnes();
            for (int ii = 0, dim_i = 0; ii < tensor_order - 1; ii++, dim_i++)
            {
                if (dim_i == mode)
                {
                    dim_i = dim_i + 1;
                }
                #ifdef FACTORS_ARE_TRANSPOSED
                    F_T_F.noalias() = Init_Factors[dim_i] * Init_Factors[dim_i].transpose();
                #else
                    F_T_F.noalias() = Init_Factors[dim_i].transpose() * Init_Factors[dim_i];                    
                #endif
                V = V.cwiseProduct(F_T_F);
            }
            stop_t_computeV += toc(start_t_computeV);

            // start_t_khatrirao = tic();
            // FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, mode, Init_Factors, Khatri_Rao, get_num_threads());
            // stop_t_khatrirao += toc(start_t_khatrirao);

            /*--+ Compute MTTKRP +--*/
            start_t_mttkrp = tic();
            // Init_Factors[mode] = True_Tensor_Mat[mode] * Khatri_Rao * V;
            
            // mttkrp(True_Tensor_Mat[mode], Khatri_Rao[mode], MTTKRP[mode], tensor_dims, mode, threads_num);
            // --- No KhatriRao Matrix version

            //--+ Compute MTTKRP without using any intermediate variable (No FULL Khatri-Rao product) --+
            mttpartialkrp(tensor_order, tensor_dims, tensor_rank, mode, Init_Factors, True_Tensor_Mat[mode], MTTKRP[mode], threads_num);
            stop_t_mttkrp += toc(start_t_mttkrp);
            
            start_t_update_factor = tic();
            /*--+ Update Factor A_(n) +--*/
            updateFactor(MTTKRP[mode], V, Init_Factors[mode], constraints, mode);
            stop_t_update_factor += toc(start_t_update_factor);

            if (mode == tensor_order - 1)
            {   
                start_t_compute_cpderr = tic();
                // cpd_err = cpderr(True_Tensor_Mat[mode], Init_Factors[mode], Khatri_Rao[mode], Est_Tensor_Mat, tensor_dims, mode);
                f_value = compute_fval(MTTKRP[mode], Init_Factors[mode], V, frob_X);
                stop_t_compute_cpderr += toc(start_t_compute_cpderr);
            }
        }
        
        /*--+ Normalization Step +--*/
        start_t_normalize = tic();
        normalize(tensor_dims, tensor_rank, Init_Factors);
        stop_t_normalize += toc(start_t_normalize);
        
        /*--+ Print CPD_ALS_INFO +--*/
        #ifdef PRINT_INFO
            std::cout << "\titer = " << iter << " \t -- \t || X - X_est || = " << f_value << std::endl;
        #endif    
        iter++;
        if (iter > MAX_ITER || f_value < epsilon)
        {
            break;
        }

        /*--+ Acceleration Step +--*/
        if (iter > k_0)
        {
            start_t_accelerate = tic();
            // lineSearchAccel(tensor_dims, tensor_rank, Prev_Factors, Init_Factors, True_Tensor_Mat, Khatri_Rao, MTTKRP,
                            // &acc_fail, &acc_coeff, iter, f_value, frob_X);
            /* Use the following in PartialKRP version */
            lineSearchAccel(tensor_dims, tensor_rank, Prev_Factors, Init_Factors, True_Tensor_Mat, MTTKRP,
                            &acc_fail, &acc_coeff, iter, f_value, frob_X);
            stop_t_accelerate += toc(start_t_accelerate);
        }

        for(int mode = 0; mode < tensor_order; mode++)
        {
            Prev_Factors[mode] = Init_Factors[mode]; // Keep old Factors for acceleration .
        }

    }
    double stop_t_cpd_als = toc(start_t_cpd_als);

    /*--+ Print terminating INFO +--*/
    // std::cout << std::endl;

    /*--+ Print TIME_INFO +--*/
    #ifdef PRINT_INFO
        std::cout <<        std::endl 
                  << ">+ ALS_CPD()      : Elapsed time: " << stop_t_cpd_als        << " seconds" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ compute V    : Elapsed time: " << stop_t_computeV       << " seconds" << " \t" << 100 * stop_t_computeV / stop_t_cpd_als       << " % (Total time)" << std::endl;
        // std::cout << " |" << std::endl 
        //           << " |__ khatrirao()  : Elapsed time: " << stop_t_khatrirao      << " seconds" << " \t" << 100 * stop_t_khatrirao / stop_t_cpd_als      << " % (Total time)" << std::endl;
        // std::cout << " |" << std::endl
        std::cout << " |" << std::endl 
                  << " |__ khatrirao()  : FULL KRP NOT COMPUTED" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ mttkrp()     : Elapsed time: " << stop_t_mttkrp         << " seconds" << " \t" << 100 * stop_t_mttkrp / stop_t_cpd_als         << " % (Total time)" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ updateFactor : Elapsed time: " << stop_t_update_factor  << " seconds" << " \t" << 100 * stop_t_update_factor / stop_t_cpd_als  << " % (Total time)" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ cpderr()     : Elapsed time: " << stop_t_compute_cpderr << " seconds" << " \t" << 100 * stop_t_compute_cpderr / stop_t_cpd_als << " % (Total time)" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ normalize()  : Elapsed time: " << stop_t_normalize      << " seconds" << " \t" << 100 * stop_t_normalize / stop_t_cpd_als << " % (Total time)" << std::endl;
        std::cout << " |" << std::endl 
                  << " |__ accelerate() : Elapsed time: " << stop_t_accelerate     << " seconds" << " \t" << 100 * stop_t_accelerate / stop_t_cpd_als << " % (Total time)" << std::endl;
        std::cout << std::endl;
        std::cout << ">- Threads used = " << threads_num << std::endl << std::endl;
    #endif
}

#endif