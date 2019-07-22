/*--------------------------------------------------------------------------------------------------*/
/* 					Function for the optimal Nesterov algorithm for the 							*/
/*    						Non-Negative Least Squares problem									   	*/
/*                (calls Compute_SVD, G_Lambda and update_alpha functions)      					*/
/*                                                                           						*/
/* A. P. Liavas																						*/
/* Georgios Lourakis																				*/ 
/* Georgios Kostoulas																				*/
/* 15/11/2017                                              											*/
/*--------------------------------------------------------------------------------------------------*/
#include "ntf_functions.h"

using namespace Eigen;


void Compute_SVD(double *L, double *mu, const Ref<const MatrixXd> Z) {
	JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
	*L = svd.singularValues().maxCoeff();
	*mu = svd.singularValues().minCoeff();
}

void G_Lambda(double *lambda, double *q, double *L, double mu) {
	*q = mu / (*L);
	if (1 / (*q) > 1e6)
		*lambda = 10 * mu;
	else if (1 / (*q) > 1e3)
		*lambda = mu;
	else
		*lambda = mu / 10;

	*L += (*lambda);
	mu += (*lambda);
	*q = mu / (*L);
}

void update_alpha(double alpha, double q, double *new_alpha)
{
	double a, b, c, D;
	a = 1;
	b = alpha * alpha - q;
	c = -alpha * alpha;
	D = b * b - 4 * a * c;

	*new_alpha = (-b + sqrt(D)) / 2;
}

void Nesterov_Matrix_Nnls(omp_obj omp_var, Ref<MatrixXd> Z, Ref<MatrixXd> W, Ref<MatrixXd> A_init, double delta_1, double delta_2)
{
	int m = A_init.rows();
	int r = A_init.cols();
	double L, mu, lambda, q, alpha, new_alpha, beta, beta_incr, constant;

	int nesterov_iter=0, MAX_ITER=50;

	MatrixXd grad_Y(m, r);
	MatrixXd Y(m, r);
	MatrixXd new_A(m, r);
	MatrixXd A(m, r);
	MatrixXd Zero_Matrix = MatrixXd::Zero(m, r);
	
	Compute_SVD(&L, &mu, Z);

	G_Lambda(&lambda, &q, &L, mu);

	Z += lambda * MatrixXd::Identity(r, r);
	W -= lambda * A_init;
	
	alpha = 1; constant = 1;
	
	A = A_init;
	Y = A_init;
	Eigen::setNbThreads(1);

	while(1){
		grad_Y = W;										// |
		// grad_Y.noalias() += Y * Z;						// | grad_Y = W + Y * Z.transpose();

		#pragma omp parallel for default(none) \
								 num_threads(omp_var.threads_per_socket) \
								 reduction(sum : grad_Y) \
								 shared(Y, Z, r)

		for (int row = 0; row < r; row++)
		{
			grad_Y.noalias() += Y.col(row) * Z.row(row);
		}

		if (grad_Y.cwiseProduct(Y).cwiseAbs().maxCoeff() <= delta_1 && grad_Y.minCoeff() >= -delta_2)
			break;
		
		new_A = (Y - grad_Y/L).cwiseMax(Zero_Matrix);
		
		update_alpha(alpha, q, &new_alpha);
		beta = alpha * (1 - alpha) / (alpha*alpha + new_alpha);

		Y = (1 + beta) * new_A - beta * A;
		
		A = new_A;
		alpha = new_alpha;
		// nesterov_iter++;
	}
	A_init = A;

	Eigen::setNbThreads(omp_var.threads);
}
