/*--------------------------------------------------------------------------------------------------*/
/* 				Function for the computation of the Objective Value	of the CPD problem	 			*/
/*    							(used in the acceleration step)									   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/ 
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "cpd_functions.h"

using namespace Eigen;

double Get_Objective_Value_Accel(omp_obj omp_var, size_t I, size_t J, const Ref<const MatrixXd> C, const Ref<const MatrixXd> X_C, const Ref<const MatrixXd> Kr, const Ref<const MatrixXd> A_T_A,
								 const Ref<const MatrixXd> B_T_B, const Ref<const MatrixXd> C_T_C, double frob_X)
{
	double global_sum;
	int R = C.cols();
	size_t K = C.rows();
	MatrixXd temp_X1 = MatrixXd::Zero(R,R);
	MatrixXd temp_X2;
	MatrixXd temp_Result = MatrixXd::Zero(K, R);


	Eigen::setNbThreads(1);

	omp_set_nested(1);
	#pragma omp parallel num_threads(omp_var.sockets)
	{
		const int outer_thread_id = omp_get_thread_num();
		numa_run_on_node(outer_thread_id);
		#pragma omp parallel for reduction(sum : temp_Result) \
							 num_threads(omp_var.threads_per_socket) \
							 schedule(dynamic) \
							 default(none) \
							 shared(I, K, J, R, omp_var)
		for (int j = 0; j < int(J / omp_var.sockets) + (outer_thread_id == omp_var.sockets - 1) * (J % omp_var.sockets); j++)
		{
			temp_Result.noalias() += X_C.block(0, (j + outer_thread_id * int(J / omp_var.sockets)) * I, K, I) * Kr.block((j + outer_thread_id * int(J / omp_var.sockets)) * I, 0, I, R);
		}

		#pragma omp master
		{
			#pragma omp parallel for default(none) \
								num_threads(omp_var.threads_per_socket) \
								reduction(sum : temp_X1) \
								shared(K, temp_Result)

			for (int k = 0; k < K; k++)
			{
				temp_X1.noalias() += temp_Result.transpose().col(k) * C.row(k);
			}
		}
	}
	Eigen::setNbThreads(omp_var.threads);

	// temp_X1.noalias() = temp_Result.transpose() * C;
	
	// temp_X1.noalias() = (Kr.transpose() * X_C.transpose()) * C;
	temp_X2.noalias() = A_T_A.cwiseProduct(B_T_B) * C_T_C;
	global_sum = temp_X1.trace();
		
	return sqrt(frob_X - 2 * global_sum + temp_X2.trace());
}
