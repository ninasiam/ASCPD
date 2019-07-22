/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the Computation of the Khatri-Rao-Product 							*/
/*    							of matrices U2 and U1											   	*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;
using namespace std;

void Khatri_Rao_Product(omp_obj omp_var, const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> KhatriRao)
{
	// int i,j;
	// VectorXd temp = VectorXd::Zero(U1.rows());

	Eigen::setNbThreads(1);

	numa_run_on_node(0);
		
	#pragma omp parallel for num_threads(omp_var.threads_per_socket) \
							schedule(dynamic) \
							default(none) \
							shared(KhatriRao)

	for (int j = 0; j < U2.cols(); j++){
		VectorXd temp = U1.col(j);
		for (int i = 0; i < U2.rows(); i++){
			KhatriRao.block(size_t(i * U1.rows()), j, U1.rows(), 1).noalias() = U2(i, j) * temp;
		}
	}

	Eigen::setNbThreads(omp_var.threads);
}
