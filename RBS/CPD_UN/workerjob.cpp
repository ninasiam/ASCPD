/*--------------------------------------------------------------------------------------------------*/
/* 							Function for Updating Factors A, B										*/
/*    						                                     								   	*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"
#include <iostream>

using namespace Eigen;

void Workerjob(omp_obj omp_var, size_t J, size_t K, MatrixXd &A, const Ref<const MatrixXd> C_T_C, const Ref<const MatrixXd> A_T_A,
			   const Ref<const MatrixXd> X_A, const Ref<const MatrixXd> Kr, int factor)
{
	size_t I = A.rows();
	int R = A.cols();
	MatrixXd Z(R, R);
	MatrixXd W = MatrixXd::Zero(I, R);
	A.setZero();

	Z.noalias() = (C_T_C).cwiseProduct(A_T_A);	// Z_A = C^T*C .* B^T*B  or  Z_B = C^T*C .* A^T*A
	Z.noalias() = Z.inverse();

	Eigen::setNbThreads(1);

	if (factor == 2)
	{
		omp_set_nested(1);
		#pragma omp parallel num_threads(omp_var.sockets)
		{	
			const int outer_thread_id = omp_get_thread_num();
			numa_run_on_node(outer_thread_id);

			#pragma omp parallel for reduction(sum : W) \
								 num_threads(omp_var.threads_per_socket) \
								 schedule(dynamic) \
								 default(none) \
								 shared(I, J, K, R, omp_var)

			for (int k = 0; k < int(K / omp_var.sockets) + (outer_thread_id == omp_var.sockets - 1) * (K % omp_var.sockets); k++)
			{
				W.noalias() += X_A.block((k + outer_thread_id * int(K / omp_var.sockets)) * I, 0, I, J) * Kr.block((k + outer_thread_id * int(K / omp_var.sockets)) * J, 0, J, R); // W = X_B * Kr_B
			}

			#pragma omp barrier

			#pragma omp master
			{
				#pragma omp parallel for default(none) \
									 num_threads(omp_var.threads_per_socket) \
									 reduction(sum : A) \
									 shared(W, Z, R)

				for (int r = 0; r < R; r++)
				{
					A.noalias() += W.col(r) * Z.row(r);
				}
			}
		}
	}
	else{
		// W = X_A * Kr_A
		omp_set_nested(1);
		#pragma omp parallel num_threads(omp_var.sockets)
		{
			const int outer_thread_id = omp_get_thread_num();
			numa_run_on_node(outer_thread_id);
			#pragma omp parallel for reduction(sum : W) \
			 					 num_threads(omp_var.threads_per_socket) \
			  					 schedule(dynamic) \
			  					 default(none) \
								 shared(I, K, J, R, omp_var)
			for (int j = 0; j < int(J / omp_var.sockets) + (outer_thread_id == omp_var.sockets - 1) * (J % omp_var.sockets); j++)
			{
				W.noalias() += X_A.block(0, (j + outer_thread_id * int(J / omp_var.sockets)) * K, I, K) * Kr.block((j + outer_thread_id * int(J / omp_var.sockets)) * K, 0, K, R); // W = X_A * Kr_A
			}
			
			#pragma omp barrier

			#pragma omp master
			{
				#pragma omp parallel for default(none) \
									 num_threads(omp_var.threads_per_socket) \
									 reduction(sum : A) \
									 shared(W, Z, R)

				for (int r = 0; r < R; r++)
				{
					A.noalias() += W.col(r) * Z.row(r);
				}
			}
		}
	}
	
	// A.noalias() = W * Z.inverse();
	Eigen::setNbThreads(omp_var.threads);
}
