#ifndef KHATRI_RAO_PRODUCT_HPP
#define KHATRI_RAO_PRODUCT_HPP

#include "master_lib.hpp"
#include "omp_lib.hpp"

/*-- Case: KhatriRao is transposed --*/
/* 
 * Khatri_Rao_Product(<IN>  const Ref<const MatrixXd> U2,
 *                    <IN>  const Ref<const MatrixXd> U1
 *                    <OUT> Ref<MatrixXd>             KhatriRao,
 * 					  <IN>  int                       rows, 
 * 					  <IN>  int                       cols, 
 * 					  <IN>  const unsigned int        num_threads )
 * 
 * Description: Implements the Khatri-Rao Product.
 * 
 * param U2          : KhatriRaoProduct(A_{1}, A_{2}, ..., A_{n-1}),
 * param U1          : current Factor A_{n},
 * param KhatriRao   : KhatriRaoProduct(A_{1}, A_{2}, ..., A_n),
 * param rows        : rows of output matrix KhatriRao,
 * param cols        : cols of output KhatriRao,
 * param num_threads : number of threads used to implement the Khatri-Rao Product in parallel. (Not efficient ... TODO).
 */
inline void Khatri_Rao_Product(const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> KhatriRao, int rows, int cols, const unsigned int num_threads)
{
	MatrixXd tmpU2(rows, cols);
	tmpU2 = U2.block(0, 0, rows, cols);

	#pragma omp parallel for collapse(2) default(shared) num_threads(num_threads)//schedule(static,4)
	for (int j = 0; j < cols; j++)
	{
		for (int i = 0; i < U1.rows(); i++)
		{
			KhatriRao.block(0, j * U1.rows() + i, U1.cols(), 1) = (tmpU2.col(j)).cwiseProduct((U1.row(i)).transpose());
		}
	}
}

// void Khatri_Rao_Product(omp_obj omp_var, const Ref<const MatrixXd> U2, const Ref<const MatrixXd> U1, Ref<MatrixXd> KhatriRao)
// {
// 	for (int j = 0; j < U2.cols(); j++)
// 	{
// 		for (int i = 0; i < U2.rows(); i++)
// 		{
// 			KhatriRao.block(size_t(i * U1.rows()), j, U1.rows(), 1).noalias() = U2(i, j) * U1.col(j);
// 		}
// 	}
// }

#endif