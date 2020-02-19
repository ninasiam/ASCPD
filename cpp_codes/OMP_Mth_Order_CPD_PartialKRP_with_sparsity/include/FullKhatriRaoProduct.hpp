#ifndef FULL_KRP
#define FULL_KRP

#include "master_lib.hpp"
#include "omp_lib.hpp"
#include "khatri_rao_product.hpp"

/* 
 * void FullKhatriRaoProduct(<IN>  const int                       tensor_order, 
 *                           <IN>  const Ref<const VectorXi>       tensor_dims, 
 *                           <IN>  const int                       tensor_rank, 
 *                           <IN>  const int                       current_mode,         
 *                           <IN>  const Ref<const VectorXi>       tensor_dims, 
 *                           <IN>  const int                       current_mode,
 *                           <IN>  std::array<MatrixXd, TNS_ORDER> &Init_Factors, 
 *                           <IN>  std::array<MatrixXd, TNS_ORDER> &Khatri_Rao, 
 *                           <IN>  const int                       num_threads)
 * 
 * Description: Computes Matricized Tensor Times Khatri-Rao Product.
 * 
 * param tensor_order : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims  : is the vector containing the tensor's dimensions,
 * param tensor_rank  : is the rank of input Tensor X,
 * param current_mode : is the current Factor mode in {1, 2, ..., TNS_ORDER}.
 * param Init_Factors : contains Initial Factors,
 * param Khatri_Rao   : is the respective Khatri-Rao Product used to update Factor A_{current_mode}
 * param num_threads  : is the number of available threads, defined by the environmental variable $(OMP_NUM_THREADS).
 */
template <std::size_t TNS_SIZE>
void FullKhatriRaoProduct(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank, const int current_mode,
                          std::array<MatrixXd, TNS_SIZE> &Init_Factors, std::array<MatrixXd, TNS_SIZE> &Khatri_Rao, const int num_threads)
{
    int mode_N = tensor_order - 1;

    int mode_1 = 0;

    if (current_mode == mode_N)
    {
        mode_N--;
    }
    else if(current_mode == mode_1)
    {
        mode_1 = 1;
    }

    // dim = I_(1) * ... * I_(current_mode-1) * I_(current_mode+1) * ... * I_(N)
    int dim = tensor_dims.prod() / tensor_dims(current_mode);

    // num_of_blocks = I_(mode_1+1) x I_(mode_1+2) x ... x I_(mode_N), where <I_(mode_1)> #rows of the starting factor.
    int num_of_blocks = dim / tensor_dims(mode_1); 

    VectorXi rows_offset(tensor_order - 2);
    for (int ii = tensor_order - 3, jj = mode_N; ii >= 0; ii--, jj--)
    {
        if(jj == current_mode)
        {
            jj--;
        }
        if (ii == tensor_order - 3)
        {
            rows_offset(ii) = num_of_blocks / tensor_dims(jj);
        }
        else
        {
            rows_offset(ii) = rows_offset(ii + 1) / tensor_dims(jj);
        }
    }

    #ifdef FACTORS_ARE_TRANSPOSED
    MatrixXd Kr(tensor_rank,1);
    #else
    MatrixXd Kr(1,tensor_rank);
    #endif
    // #pragma omp parallel for default(shared) private(Kr) num_threads(num_threads)
    for(int block_idx = 0; block_idx < num_of_blocks; block_idx++)
    {
        // Compute Kr = KhatriRao(A_(mode_N)(l,:), A_(mode_N-1)(k,:), ..., A_(2)(j,:))
        // Initiallize vector Kr as Kr = A_(mode_N)(l,:)
        #ifdef FACTORS_ARE_TRANSPOSED
        Kr = Init_Factors[mode_N].col( (block_idx / rows_offset(tensor_order-3)) % tensor_dims(mode_N) );

        // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
        for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
        {
            if (jj == current_mode)
            {
                jj--;
            }
            Kr = (Init_Factors[jj].col( (block_idx / rows_offset(ii)) % tensor_dims(jj) ) ).cwiseProduct(Kr);
        }

        // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
        for (int col=0; col < tensor_dims(mode_1); col++)
        {
            Khatri_Rao[current_mode].block(0, (block_idx * tensor_dims(mode_1)) + col, tensor_rank, 1) = (Init_Factors[mode_1].col(col)).cwiseProduct(Kr);
        }
        #else
        Kr = Init_Factors[mode_N].row( (block_idx / rows_offset(tensor_order-3)) % tensor_dims(mode_N) );
        // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
        for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
        {
            if (jj == current_mode)
            {
                jj--;
            }
            Kr = (Init_Factors[jj].row( (block_idx / rows_offset(ii)) % tensor_dims(jj) ) ).cwiseProduct(Kr);
        }

        // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
        for (int row=0; row < tensor_dims(mode_1); row++)
        {
            Khatri_Rao[current_mode].block(0, (block_idx * tensor_dims(mode_1)) + row, tensor_rank, 1) = ((Init_Factors[mode_1].row(row)).cwiseProduct(Kr)).transpose();
        }
        #endif
    }
}

template <std::size_t TNS_SIZE>
void FullKhatriRaoProduct(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank, const int current_mode,
                          std::array<MatrixXd, TNS_SIZE> &Init_Factors, MatrixXd &Khatri_Rao, const int num_threads)
{
    int mode_N = tensor_order - 1;

    int mode_1 = 0;

    if (current_mode == mode_N)
    {
        mode_N--;
    }
    else if(current_mode == mode_1)
    {
        mode_1 = 1;
    }

    // dim = I_(1) * ... * I_(current_mode-1) * I_(current_mode+1) * ... * I_(N)
    int dim = tensor_dims.prod() / tensor_dims(current_mode);

    // num_of_blocks = I_(mode_1+1) x I_(mode_1+2) x ... x I_(mode_N), where <I_(mode_1)> #rows of the first factor
    int num_of_blocks = dim / tensor_dims(mode_1); 

    VectorXi rows_offset(tensor_order - 2);
    for (int ii = tensor_order - 3, jj = mode_N; ii >= 0; ii--, jj--)
    {
        if(jj == current_mode)
        {
            jj--;
        }
        if (ii == tensor_order - 3)
        {
            rows_offset(ii) = num_of_blocks / tensor_dims(jj);
        }
        else
        {
            rows_offset(ii) = rows_offset(ii + 1) / tensor_dims(jj);
        }
    }

    #ifdef FACTORS_ARE_TRANSPOSED
    MatrixXd Kr(tensor_rank,1);
    #else
    MatrixXd Kr(1,tensor_rank);
    #endif
    // #pragma omp parallel for default(shared) private(Kr) num_threads(num_threads)
    for(int block_idx = 0; block_idx < num_of_blocks; block_idx++)
    {
        // Compute Kr = KhatriRao(A_(mode_N)(l,:), A_(mode_N-1)(k,:), ..., A_(2)(j,:))
        // Initiallize vector Kr as Kr = A_(mode_N)(l,:)
        #ifdef FACTORS_ARE_TRANSPOSED
        Kr = Init_Factors[mode_N].col( (block_idx / rows_offset(tensor_order-3)) % tensor_dims(mode_N) );
        
        // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
        for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
        {
            if (jj == current_mode)
            {
                jj--;
            }
            Kr = (Init_Factors[jj].col( (block_idx / rows_offset(ii)) % tensor_dims(jj) ) ).cwiseProduct(Kr);
        }

        // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
        for (int col=0; col < tensor_dims(mode_1); col++)
        {
            Khatri_Rao.block(0, (block_idx * tensor_dims(mode_1)) + col, tensor_rank, 1) = (Init_Factors[mode_1].col(col)).cwiseProduct(Kr);
        }
        #else
        Kr = Init_Factors[mode_N].row( (block_idx / rows_offset(tensor_order-3)) % tensor_dims(mode_N) );
        // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
        for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
        {
            if (jj == current_mode)
            {
                jj--;
            }
            Kr = (Init_Factors[jj].row( (block_idx / rows_offset(ii)) % tensor_dims(jj) ) ).cwiseProduct(Kr);
        }

        // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
        for (int row=0; row < tensor_dims(mode_1); row++)
        {
            Khatri_Rao.block(0, (block_idx * tensor_dims(mode_1)) + row, tensor_rank, 1) = ((Init_Factors[mode_1].row(row)).cwiseProduct(Kr)).transpose();
        }
        #endif
    }
}

/*
 *******************************************************************************************************************************
 * NOTE:
 * The following lines contain the previous version of full Khatri-Rao product.

for (int mode = 0; mode < tensor_order; mode++)
{
    // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
    int dim = tensor_dims.prod() / tensor_dims(mode);

    int mode_N = tensor_order - 1;
    if (mode == mode_N)
    {
        mode_N = mode - 1;
    }

    Khatri_Rao[mode].block(0, 0, tensor_rank, Init_Factors[mode_N].rows()) = Init_Factors[mode_N].transpose();

    for (int ii = 0, rows = Init_Factors[mode_N].rows(), curr_dim = mode_N - 1; ii < tensor_order - 2; ii++, curr_dim--)
    {
        if (curr_dim == mode)
        {
            curr_dim = curr_dim - 1;
        }
        Khatri_Rao_Product(Khatri_Rao[mode], Init_Factors[curr_dim], Khatri_Rao[mode], tensor_rank, rows, threads_num);
        rows = rows * Init_Factors[curr_dim].rows();
    }
}
 *******************************************************************************************************************************
 */

#endif