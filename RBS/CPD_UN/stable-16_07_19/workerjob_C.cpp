/*--------------------------------------------------------------------------------------------------*/
/* 							Function for Updating Factors C											*/
/*    																							   	*/
/*    																							   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"

using namespace Eigen;

void Workerjob_C(omp_obj omp_var, size_t I, size_t J, MatrixXd &C, MatrixXd &W_C, const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> A_T_A,
				 const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr)
{
	int K = C.rows();
	int R = C.cols();
	MatrixXd Z_C(R, R);
	W_C.setZero();
	C.setZero();
	Z_C.noalias() = (B_T_B).cwiseProduct(A_T_A);	// Z_C = B^T*B .* A^T*A

	Z_C.noalias() = Z_C.inverse();
	// W_C = X_C * Kr;

	// MatrixXd W = MatrixXd::Zero(K, R);
	// int n = Eigen::nbThreads();
	Eigen::setNbThreads(1);
	// #pragma omp parallel for reduction(sum : W) schedule(runtime) default(none) shared(R, K, I, J)
	// for (size_t j=0; j<J; j++){
	// 	W.noalias() += X_C.block(0, size_t(j * I), K, I) * Kr.block(size_t(j * I), 0, I, R); // W_C = X_C * Kr_C
	// }
	// const int sockets = 2;//numa_num_configured_nodes();
	// const int cores = numa_num_configured_cpus();
	// const int cores_per_socket = cores / sockets;
	// const int threads_per_socket = n / sockets;
	
	omp_set_nested(1);
	#pragma omp parallel num_threads(omp_var.sockets)
	{
		const int outer_thread_id = omp_get_thread_num();
		numa_run_on_node(outer_thread_id);
		#pragma omp parallel for reduction(sum : W_C) \
							 num_threads(omp_var.threads_per_socket) \
							 schedule(dynamic) \
							 default(none) \
							 shared(I, K, J, R, omp_var)
		for (int j = 0; j < int(J / omp_var.sockets) + (outer_thread_id == omp_var.sockets - 1) * (J % omp_var.sockets); j++)
		{
			W_C.noalias() += X_C.block(0, (j + outer_thread_id * int(J / omp_var.sockets)) * I, K, I) * Kr.block((j + outer_thread_id * int(J / omp_var.sockets)) * I, 0, I, R);
		}

		#pragma omp master
		{
			#pragma omp parallel for default(none) \
									num_threads(omp_var.threads_per_socket) \
									reduction(sum : C) \
									shared(W_C, Z_C, R)

			for (int r = 0; r < R; r++)
			{
				C.noalias() += W_C.col(r) * Z_C.row(r);
			}
		}
	}

	// Eigen::setNbThreads(omp_var.threads);

	// C.noalias() = W_C * Z_C.inverse();
	

	Eigen::setNbThreads(omp_var.threads);

	// 	W_C = X_C * Kr;
	// 	C.noalias() = W_C * Z_C.inverse();
}
