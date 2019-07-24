/*--------------------------------------------------------------------------------------------------*/
/* 				HEADER FILE cpd_functions.h					    */
/*    		(Containes declerations of functions used by cpd_eigen.cpp)		            */
/*                                                                           	                    */
/* A. P. Liavas								                            */
/* Georgios Kostoulas                                                                               */
/* Georgios Lourakis										    */
/* 15/11/2017                                              					    */
/*--------------------------------------------------------------------------------------------------*/
#ifndef CPD_FUNCTIONS_H		// if cpd_functions.h hasn't been included yet...
#define CPD_FUNCTIONS_H		// #define this so the compiler knows it has been included

// #define EIGEN_DEFAULT_TO_ROW_MAJOR

// /* Uncomment if running at DALI*/
// #include "/home/ipapagiannakos/Libraries/eigen3/Eigen/Dense"
// #include "/home/ipapagiannakos/Libraries/eigen3/Eigen/Core"
/* Uncomment if Eigen is allready installed...*/
#include <Eigen/Dense>
#include <Eigen/Core>
/* Uncomment if running at ARIS-HPC*/
// #include "../eigen3/Eigen/Dense"
// #include "../eigen3/Eigen/Core"

#include <numa.h>
#include <omp.h>
#include <iostream>

#include <cstdlib>

#pragma omp declare reduction(sum: Eigen::MatrixXd: omp_out = omp_out + omp_in) \
                    initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

typedef struct omp_obj{
	int sockets;
	int threads;
	int threads_per_socket;
} omp_obj;

using namespace Eigen;

void rand_shuffle_indices(const VectorXi &dim_size, MatrixXi &rand_indices, int factor);

void create_subfactors(const MatrixXd &Factor, MatrixXd &subFactor, const MatrixXd &Matr_Tensor, MatrixXd &Matr_subTensor, const VectorXi &dim_size, const VectorXi &block_size, MatrixXi &rand_indices, MatrixXi &B_cal, int iter, int factor);

void merge_Factors(MatrixXd &Factor, const MatrixXd &subFactor, const VectorXi &block_size, MatrixXi &B_cal, int factor);

double Get_Objective_Value(const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C_Kr, const Ref<const MatrixXd> A_T_A, const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X);

double Get_Objective_Value_Accel(omp_obj omp_var, size_t I, size_t J, const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr, const Ref<const MatrixXd> A_T_A,
								 const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X);

void Khatri_Rao_Product(omp_obj omp_var, const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> Kr);

void Line_Search_Accel(omp_obj omp_var, const Ref<const MatrixXd> A_old_N, const Ref<const MatrixXd> B_old_N, const Ref<const MatrixXd> C_old_N, Ref<MatrixXd> A,
					   Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C, Ref<MatrixXd> KhatriRao_BA,
					   const Ref<const MatrixXd> X_C, int *acc_fail, int *acc_coeff, int iter, double f_value, double frob_X);

void Normalize(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> A_T_A, Ref<MatrixXd> B_T_B, Ref<MatrixXd> C_T_C);

void Normalize_Init(Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C);

void Read_Data(omp_obj omp_var, Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> X_A, Ref<MatrixXd> X_B, Ref<MatrixXd> X_C, int I, int J, int K, int R);

void Read_Data(omp_obj omp_var, Ref<MatrixXd> A, Ref<MatrixXd> B, Ref<MatrixXd> C, Ref<MatrixXd> X_A, Ref<MatrixXd> X_C, int I, int J, int K, int R);

void Read_From_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name, int skip);

//void Read_X_C(int n_I, int n_J, int n_K, int skip_I, int skip_J, int skip_K, int I, int J, Ref<MatrixXd> X_C_one_file_T,const char *file_name);

void Set_Info(int* R, int* I, int* J, int* K, const char *file_name);

void Workerjob(omp_obj omp_var, size_t J, size_t K, MatrixXd &A, const Ref<const MatrixXd> C_T_C, const Ref<const MatrixXd> A_T_A, const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> Kr, int factor);

void Workerjob_C(omp_obj omp_var, size_t I, size_t J, MatrixXd &C, MatrixXd &W_C, const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> A_T_A, const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr);

#endif
