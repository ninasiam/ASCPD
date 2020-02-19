#ifndef LINE_SEARCH_ACCEL
#define LINE_SEARCH_ACCEL

#include "master_lib.hpp"
#include "compute_fval.hpp"
#include "khatri_rao_product.hpp"
#include "FullKhatriRaoProduct.hpp"
#include "mttkrp.hpp"
#include "omp.h"

#include <cmath>

/* 
 * void lineSearchAccel(<IN>     const Ref<const VectorXi>       tensor_dims, 
 *                      <IN>     const int                       tensor_rank,        
 *                      <IN>     std::array<MatrixXd, TNS_ORDER> &OldFactors,
 *                      <OUT>    std::array<MatrixXd, TNS_ORDER> &NewFactors,
 *                      <IN>     std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat,
 *                      <IN/OUT> std::array<MatrixXd, TNS_ORDER> &Khatri_Rao,
 *                      <IN/OUT> std::array<MatrixXd, TNS_ORDER> &MTTKRP,
 *                      <IN/OUT> int                             *acc_fail,  
 *                      <IN/OUT> int                             *acc_coeff, 
 *                      <IN>     const                           int iter, 
 *                      <IN>     const double                    f_value, 
 *                      <IN>     const double                    frob_X)
 *                      
 * 
 * Description: Function that implements the acceleration step.
 *
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param OldFactors      : contains all Factors of iteration "AO_iter - 1",
 * param NewFactors      : contains all new accelerated Factors,
 * param True_Tensor_Mat : is the current matricized Tensor X_{current_mode},
 * param Khatri_Rao      : is the updated Khatri-Rao Product for Factor A_{current_mode},
 * param MTTKRP          : is the updated MTTKRP,
 * param acc_fail        : is the number of accelerating steps which led to a greater cost function, 
 * param acc_coeff       : is the acceleration coefficient,
 * param iter            : is the AO_iter,
 * param f_value         : is the cost function's value before the acceleration step,
 * param frob_X          : is the frobenious norm of input Tensor X.
 */

template <std::size_t TNS_SIZE>
void lineSearchAccel(const Ref<const VectorXi> tensor_dims, const int tensor_rank,
                     std::array<MatrixXd, TNS_SIZE> &OldFactors,
                     std::array<MatrixXd, TNS_SIZE> &NewFactors,
                     std::array<MatrixXd, TNS_SIZE> &True_Tensor_Mat,
                     std::array<MatrixXd, TNS_SIZE> &Khatri_Rao,
                     std::array<MatrixXd, TNS_SIZE> &MTTKRP,
                     int *acc_fail, double *acc_coeff, const int iter,
                     const double f_value, const double frob_X)
{
    double f_accel;
    double acc_step = pow(iter + 1, (1.0 / (*acc_coeff)));

    const int tensor_order = tensor_dims.size();

    std::array<MatrixXd, TNS_SIZE> Accel_Factors;
    MatrixXd V(tensor_rank, tensor_rank);
    MatrixXd F_T_F(tensor_rank, tensor_rank);

    // Update ALL Factors using acceleration method.
    for(int mode_i=0; mode_i<tensor_order; mode_i++)
    {
        Accel_Factors[mode_i].noalias() = OldFactors[mode_i] + acc_step * (NewFactors[mode_i] - OldFactors[mode_i]);
    }

    // Compute NEW matrix V.
    V.setOnes();
    for (int mode_i = 0; mode_i < tensor_order - 1; mode_i++)
    {
        #ifdef FACTORS_ARE_TRANSPOSED
            F_T_F = Accel_Factors[mode_i] * Accel_Factors[mode_i].transpose();
        #else
            F_T_F = Accel_Factors[mode_i].transpose() * Accel_Factors[mode_i];
        #endif
        V = V.cwiseProduct(F_T_F);
    }

    // Compute NEW Khatri-Rao product.
    FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, tensor_order - 1, Accel_Factors, Khatri_Rao, 1);
    // Khatri_Rao[TNS_ORDER - 1].block(0, 0, tensor_rank, Accel_Factors[TNS_ORDER - 2].rows()) = Accel_Factors[TNS_ORDER - 2].transpose();
    // for (int mode_i = TNS_ORDER - 3, rows = Accel_Factors[TNS_ORDER - 2].rows(); mode_i >= 0; mode_i--)
    // {
    //     Khatri_Rao_Product(Khatri_Rao[TNS_ORDER - 1], Accel_Factors[mode_i], Khatri_Rao[TNS_ORDER - 1], tensor_rank, rows, get_num_threads());
    //     rows = rows * Accel_Factors[mode_i].rows();
    // }

    // Compute NEW MTTKRP.
    mttkrp(True_Tensor_Mat[tensor_order - 1], Khatri_Rao[tensor_order - 1], MTTKRP[tensor_order - 1], tensor_dims, tensor_order - 1, get_num_threads());

    // Compute NEW f_val.
    f_accel = compute_fval(MTTKRP[tensor_order - 1], Accel_Factors[tensor_order - 1], V, frob_X);

    if (f_value > f_accel)
    {
        // Update ALL Factors.
        for(int mode_i = 0; mode_i < tensor_order; mode_i++)
        {
            NewFactors[mode_i] = Accel_Factors[mode_i];
        }
    }
    else
    {
        (*acc_fail)++;
    }


    if (*acc_fail == 5)
    {
        *acc_fail = 0;
        (*acc_coeff)++;
    }
}

// --- No usage of KhatriRao Matrix ---
/* 
 * void lineSearchAccel(<IN>     const Ref<const VectorXi>       tensor_dims, 
 *                      <IN>     const int                       tensor_rank,        
 *                      <IN>     std::array<MatrixXd, TNS_ORDER> &OldFactors,
 *                      <OUT>    std::array<MatrixXd, TNS_ORDER> &NewFactors,
 *                      <IN>     std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat,
 *                      <IN/OUT> std::array<MatrixXd, TNS_ORDER> &MTTKRP,
 *                      <IN/OUT> int                             *acc_fail,  
 *                      <IN/OUT> int                             *acc_coeff, 
 *                      <IN>     const                           int iter, 
 *                      <IN>     const double                    f_value, 
 *                      <IN>     const double                    frob_X)
 *                      
 * 
 * Description: Function that implements the acceleration step.
 *
 * param tensor_dims     : is the vector containing the tensor's dimentions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param OldFactors      : contains all Factors of iteration "AO_iter - 1",
 * param NewFactors      : contains all new accelerated Factors,
 * param True_Tensor_Mat : is the current matricized Tensor X_{current_mode},
 * param MTTKRP          : is the updated MTTKRP,
 * param acc_fail        : is the number of accelerating steps which led to a greater cost function, 
 * param acc_coeff       : is the acceleration coefficient,
 * param iter            : is the AO_iter,
 * param f_value         : is the cost function's value before the acceleration step,
 * param frob_X          : is the frobenious norm of input Tensor X.
 */
template <std::size_t TNS_SIZE>
void lineSearchAccel(const Ref<const VectorXi> tensor_dims, const int tensor_rank,
                     std::array<MatrixXd, TNS_SIZE> &OldFactors,
                     std::array<MatrixXd, TNS_SIZE> &NewFactors,
                     std::array<MatrixXd, TNS_SIZE> &True_Tensor_Mat,
                     std::array<MatrixXd, TNS_SIZE> &MTTKRP,
                     int *acc_fail, double *acc_coeff, const int iter,
                     const double f_value, const double frob_X)
{
    double f_accel;
    double acc_step = pow(iter + 1, (1.0 / (*acc_coeff)));

    const int tensor_order = tensor_dims.size();

    std::array<MatrixXd, TNS_SIZE> Accel_Factors;
    MatrixXd V(tensor_rank, tensor_rank);
    MatrixXd F_T_F(tensor_rank, tensor_rank);

    // Update ALL Factors using acceleration method.
    for (int mode_i = 0; mode_i < tensor_order; mode_i++)
    {
        Accel_Factors[mode_i].noalias() = OldFactors[mode_i] + acc_step * (NewFactors[mode_i] - OldFactors[mode_i]);
    }

    // Compute NEW matrix V.
    V.setOnes();
    for (int mode_i = 0; mode_i < tensor_order - 1; mode_i++)
    {
        #ifdef FACTORS_ARE_TRANSPOSED
            F_T_F.noalias() = Accel_Factors[mode_i] * Accel_Factors[mode_i].transpose();
        #else
            F_T_F.noalias() = Accel_Factors[mode_i].transpose() * Accel_Factors[mode_i];
        #endif
        V = V.cwiseProduct(F_T_F);
    }

    mttpartialkrp(tensor_order, tensor_dims, tensor_rank, tensor_order - 1, Accel_Factors, True_Tensor_Mat[tensor_order - 1], MTTKRP[tensor_order - 1], get_num_threads());

    // Compute NEW f_val.
    f_accel = compute_fval(MTTKRP[tensor_order - 1], Accel_Factors[tensor_order - 1], V, frob_X);

    if (f_value > f_accel)
    {
        // Update ALL Factors.
        for (int mode_i = 0; mode_i < tensor_order; mode_i++)
        {
            NewFactors[mode_i] = Accel_Factors[mode_i];
        }
    }
    else
    {
        (*acc_fail)++;
    }

    if (*acc_fail == 5)
    {
        *acc_fail = 0;
        (*acc_coeff)++;
    }
}

#endif