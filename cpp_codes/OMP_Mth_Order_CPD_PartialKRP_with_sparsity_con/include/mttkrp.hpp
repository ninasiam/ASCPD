#ifndef MTTKRP_HPP
#define MTTKRP_HPP

#include "master_lib.hpp"
#include "omp_lib.hpp"

/* 
 * void mttkrp(<IN>  const Ref<const MatrixXd> Tensor_X, 
 *             <IN>  const Ref<const MatrixXd> KhatriRao, 
 *             <OUT> MatrixXd                  &MTTKRP,                   
 *             <IN>  const Ref<const VectorXi> tensor_dims, 
 *             <IN>  const int                 current_mode,
 *             <IN>  const unsigned int        num_threads)
 * 
 * Description: Computes Matricized Tensor Times Khatri-Rao Product.
 * 
 * param Tensor_X     : is the matricized input tensor,
 * param KhatriRao    : is the Khatri-Rao Product for Factor A_{current_mode}
 * param MTTKRP       : is the resulting matrix,
 * param tensor_dims  : is the vector containing the tensor's dimensions,
 * param current_mode : is the current Factor mode in {1, 2, ..., TNS_ORDER}.
 * param num_threads  : is the number of available threads, defined by the environmental variable $(OMP_NUM_THREADS).
 */
void mttkrp(const Ref<const MatrixXd> Tensor_X, const Ref<const MatrixXd> KhatriRao, MatrixXd &MTTKRP,
            const Ref<const VectorXi> tensor_dims, const int current_mode, const unsigned int num_threads)
{
    #ifndef EIGEN_DONT_PARALLELIZE
    Eigen::setNbThreads(1);
    #endif

    MTTKRP.setZero();

    VectorXi reduced_tensor_dim = tensor_dims;
    reduced_tensor_dim(current_mode) = 1;
    /*--- Larger blocks of size I(n) x ([I(1) x ... x I(n-1) x I(n+1) x ... I(N)] / <max(I)>)  - Fewer for-loop iterations : <max(I)> ---*/
    // int max_dim = reduced_tensor_dim.maxCoeff();
    // int offset = reduced_tensor_dim.prod();
    // offset = (offset / max_dim);

    /*--- Smaller blocks of size I(n) x <max(I)>  - More for-loop iterations : (I(1) x ... x I(n-1) x I(n+1) x ... I(N)) ---*/
    int max_dim = reduced_tensor_dim.prod();
    int offset = reduced_tensor_dim.maxCoeff();
    max_dim = (max_dim / offset);


    size_t inner_num_threads = num_threads / NUM_SOCKETS;

    int floor_offset = (int)(offset / inner_num_threads);
    int left_overs   = offset % inner_num_threads;
    int left_max_dim = (left_overs != 0);
    int max_dim_thr  = max_dim * inner_num_threads;
    int num_blocks   = max_dim_thr + left_max_dim;
    int block_left   = max_dim * left_overs;

    // #pragma omp parallel for reduction(sum : MTTKRP) //schedule(dynamic, 1)
    // for (int block = 0; block < num_blocks; block++)
    // {
    //     MTTKRP.noalias() += Tensor_X.block(0, block * floor_offset, MTTKRP.rows(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left) * KhatriRao.block(block * floor_offset, 0, (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left, MTTKRP.cols());
    // }
    //---TRANSPOSED VERSION---
    omp_set_nested(1);

    #pragma omp parallel for num_threads(NUM_SOCKETS) proc_bind(spread)
    for (int sock_id=0; sock_id<NUM_SOCKETS; sock_id++)
    {
        #pragma omp parallel for reduction(sum : MTTKRP) schedule(static, 1) num_threads(inner_num_threads) proc_bind(close)
        for (int block = sock_id * (int)(num_blocks / NUM_SOCKETS); block < (sock_id + 1) * (int)(num_blocks / NUM_SOCKETS) + (sock_id + 1 == NUM_SOCKETS) * (num_blocks % NUM_SOCKETS); block++)
        {
            MTTKRP.noalias() += Tensor_X.block(0, block * floor_offset, MTTKRP.rows(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left) * KhatriRao.block(0, block * floor_offset, MTTKRP.cols(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left).transpose();
        }
    }
    #ifndef EIGEN_DONT_PARALLELIZE
    Eigen::setNbThreads(num_threads);
    #endif
}

// --- No usage of KhatriRao Matrix ---
/* 
 * void mttpartialkrp(<IN>  const int                       tensor_order, 
 *                    <IN>  const Ref<const VectorXi>       tensor_dims, 
 *                    <IN>  const int                       tensor_rank, 
 *                    <IN>  const int                       current_mode,
 *                    <IN>  std::array<MatrixXd, TNS_ORDER> &Init_Factors,
 *                    <IN>  const Ref<const MatrixXd>       Tensor_X, 
 *                    <OUT> MatrixXd                        &MTTKRP, 
 *                    <IN>  const unsigned int              num_threads)
 * 
 * Description: Computes Matricized Tensor Times Khatri-Rao Product.
 * 
 * param tensor_order    : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param current_mode : is the current Factor mode in {1, 2, ..., TNS_ORDER}.
 * param Init_Factors    : contains Initial Factors,
 * param Tensor_X     : is the matricized input tensor,
 * param KhatriRao    : is the Khatri-Rao Product for Factor A_{current_mode}
 * param MTTKRP       : is the resulting matrix,
 * param num_threads  : is the number of available threads, defined by the environmental variable $(OMP_NUM_THREADS).
 */
template <std::size_t TNS_SIZE>
void mttpartialkrp(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank, const int current_mode,
                   std::array<MatrixXd, TNS_SIZE> &Init_Factors, const Ref<const MatrixXd> Tensor_X, MatrixXd &MTTKRP,
                   const unsigned int num_threads)
{
    #ifndef EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(1);
    #endif

    MTTKRP.setZero();

    int mode_N = tensor_order - 1;

    int mode_1 = 0;

    if (current_mode == mode_N)
    {
        mode_N--;
    }
    else if (current_mode == mode_1)
    {
        mode_1 = 1;
    }

    // MatrixXd PartialKR(tensor_dims(mode_1), tensor_rank);

    // dim = I_(1) * ... * I_(current_mode-1) * I_(current_mode+1) * ... * I_(N)
    int dim = tensor_dims.prod() / tensor_dims(current_mode);

    // num_of_blocks = I_(mode_1+1) x I_(mode_1+2) x ... x I_(mode_N), where <I_(mode_1)> #rows of the starting factor.
    int num_of_blocks = dim / tensor_dims(mode_1);

    VectorXi rows_offset(tensor_order - 2);
    for (int ii = tensor_order - 3, jj = mode_N; ii >= 0; ii--, jj--)
    {
        if (jj == current_mode)
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

    size_t inner_num_threads = num_threads / NUM_SOCKETS;
    if (inner_num_threads < 1){inner_num_threads = 1;}
    uint chunk_size = 64 / tensor_rank;
    if (chunk_size < 1){chunk_size = 1;}
    else if(64 % tensor_rank != 0){chunk_size++;}

    omp_set_nested(1);

    #pragma omp parallel for num_threads(NUM_SOCKETS) proc_bind(spread)
    for (int sock_id=0; sock_id<NUM_SOCKETS; sock_id++)
    {
        #pragma omp parallel for reduction(sum: MTTKRP) schedule(static, chunk_size) num_threads(inner_num_threads) proc_bind(close)
        // for (int block_idx = 0; block_idx < num_of_blocks; block_idx++)
        for (int block_idx = sock_id * (int)(num_of_blocks / NUM_SOCKETS); block_idx < (sock_id + 1) * (int)(num_of_blocks / NUM_SOCKETS) + (sock_id + 1 == NUM_SOCKETS) * (num_of_blocks % NUM_SOCKETS); block_idx++)
        {
            // Compute Kr = KhatriRao(A_(mode_N)(l,:), A_(mode_N-1)(k,:), ..., A_(2)(j,:))
            // Initialize vector Kr as Kr = A_(mode_N)(l,:)
            #ifdef FACTORS_ARE_TRANSPOSED
                MatrixXd Kr(tensor_rank,1);
                Kr = Init_Factors[mode_N].col((block_idx / rows_offset(tensor_order - 3)) % tensor_dims(mode_N));
                MatrixXd PartialKR(tensor_rank, tensor_dims(mode_1));

                // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
                for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
                {
                    if (jj == current_mode)
                    {
                        jj--;
                    }
                    Kr = (Init_Factors[jj].col((block_idx / rows_offset(ii)) % tensor_dims(jj))).cwiseProduct(Kr);
                }

                // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
                for (int col = 0; col < tensor_dims(mode_1); col++)
                {
                    PartialKR.col(col) = ((Init_Factors[mode_1].col(col)).cwiseProduct(Kr));
                }
                
                MTTKRP.noalias() += Tensor_X.block(0, block_idx * tensor_dims(mode_1), tensor_dims(current_mode), tensor_dims(mode_1)) * PartialKR.transpose();        

            #else
                MatrixXd Kr(1,tensor_rank);
                Kr = Init_Factors[mode_N].row((block_idx / rows_offset(tensor_order - 3)) % tensor_dims(mode_N));
                MatrixXd PartialKR(tensor_dims(mode_1), tensor_rank);
                // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
                for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
                {
                    if (jj == current_mode)
                    {
                        jj--;
                    }
                    Kr = (Init_Factors[jj].row((block_idx / rows_offset(ii)) % tensor_dims(jj))).cwiseProduct(Kr);
                }

                // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
                for (int row = 0; row < tensor_dims(mode_1); row++)
                {
                    PartialKR.row(row)  = ((Init_Factors[mode_1].row(row)).cwiseProduct(Kr));
                }
                
                MTTKRP.noalias() += Tensor_X.block(0, block_idx * tensor_dims(mode_1), tensor_dims(current_mode), tensor_dims(mode_1)) * PartialKR;
                    
            #endif
            
        }
    }
    #ifndef EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(num_threads);
    #endif
}

/*--- Non-NUMA aware version ---*/
/*
void mttkrp(const Ref<const MatrixXd> Tensor_X, const Ref<const MatrixXd> KhatriRao, MatrixXd &MTTKRP,
            const Ref<const VectorXi> tensor_dims, const int current_mode, const unsigned int num_threads)
{
#ifndef EIGEN_DONT_PARALLELIZE
    Eigen::setNbThreads(1);
#endif

    VectorXi reduced_tensor_dim = tensor_dims;
    reduced_tensor_dim(current_mode) = 1;
    int max_dim = reduced_tensor_dim.maxCoeff();

    MTTKRP.setZero();

    int offset = reduced_tensor_dim.prod();
    offset = (offset / max_dim);

    int floor_offset = (int)(offset / num_threads);
    int left_overs = offset % num_threads;
    int left_max_dim = (left_overs != 0);
    int max_dim_thr = max_dim * num_threads;
    int num_blocks = max_dim_thr + left_max_dim;
    int block_left = max_dim * left_overs;

// #pragma omp parallel for reduction(sum : MTTKRP) //schedule(dynamic, 1)
// for (int block = 0; block < num_blocks; block++)
// {
//     MTTKRP.noalias() += Tensor_X.block(0, block * floor_offset, MTTKRP.rows(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left) * KhatriRao.block(block * floor_offset, 0, (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left, MTTKRP.cols());
// }
//---TRANSPOSED VERSION---
#pragma omp parallel for reduction(sum \
                                   : MTTKRP) schedule(dynamic, 1)
    for (int block = 0; block < num_blocks; block++)
    {
        MTTKRP.noalias() += Tensor_X.block(0, block * floor_offset, MTTKRP.rows(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left) * KhatriRao.block(0, block * floor_offset, MTTKRP.cols(), (block < max_dim_thr) * floor_offset + (block == num_blocks - 1) * block_left).transpose();
    }

#ifndef EIGEN_DONT_PARALLELIZE
    Eigen::setNbThreads(num_threads);
#endif
}
*/

// For Non-transposed Khatri-Rao use the following ...
// void mttkrp(const Ref<const MatrixXd> Tensor_X, const Ref<const MatrixXd> KhatriRao, MatrixXd &MTTKRP,
//                    const Ref<const VectorXi> Tensor_Dimensions, const int current_mode, const unsigned int num_threads)
// {
//     #ifndef EIGEN_DONT_PARALLELIZE
//     Eigen::setNbThreads(1);
//     #endif

//     VectorXi reduced_tensor_dim = Tensor_Dimensions;
//     reduced_tensor_dim(current_mode) = 1;
//     int max_dim = reduced_tensor_dim.maxCoeff();

//     MTTKRP.setZero();

//     int offset = reduced_tensor_dim.prod();
//     offset = (offset / max_dim);

//     #pragma omp parallel for reduction(sum: MTTKRP) default(shared) num_threads(num_threads)
//     for (int block = 0; block < max_dim; block++)
//     {
//         MTTKRP.noalias() += Tensor_X.block(0, block * offset, MTTKRP.rows(), offset) * KhatriRao.block(block * offset, 0, offset, MTTKRP.cols());
//     }
    
//     #ifndef EIGEN_DONT_PARALLELIZE
//     Eigen::setNbThreads(num_threads);
//     #endif
// }

#endif