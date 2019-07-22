// Check : https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_khr_fp64.html

#include "additional_func.hpp"

int main(int argc, char const *argv[]){
    srand(time(0));
    struct ocl_parameters ocl_p;
    int debug_mode = FALSE;
    const int LIST_SIZE = 1024; // default value : 1024
    const int M = LIST_SIZE;
    const int N = LIST_SIZE;
    const int K = LIST_SIZE;
    char kernel_func_name[] = "matrix_mul";
    int isTransposed;
    ////////////////////////////////////////////////////////////////////////////
    // Option 1
    // char kernel_file_name[] = "matxd_mul_kernel_opt.cl";
    // size_t global_work_size = (size_t)LIST_SIZE; // <Global>
    // size_t local_work_size = TS;         // <Local>
    // isTransposed = TRUE;
    // Option 2
    // char kernel_file_name[] = "matxd_mul_kernel_opt0.cl";
    // size_t global_work_size = (size_t)LIST_SIZE;  // <Global>
    // size_t local_work_size =  TS;        // <Local>
    // isTransposed = FALSE;
    // Option 3
    char kernel_file_name[] = "matxd_mul_kernel_opt1.cl";
    size_t global_work_size[] = {(size_t)(LIST_SIZE/WPTM),(size_t)(LIST_SIZE/WPTN)};  // <Global>
    size_t local_work_size[] = {(size_t)(TSM/WPTM), (size_t)(TSN/WPTN)};        // <Local>
    isTransposed = TRUE;
    ////////////////////////////////////////////////////////////////////////////

    // Create Matrices ...
    matX_type A = matX_type::Random(M,K);
    matX_type B = matX_type::Random(K,N);
    matX_type B_T = matX_type::Zero(N,K);
    matX_type C = matX_type::Zero(M,N);
    matX_type C2 = matX_type::Zero(M,N);
    if(isTransposed){ B_T = B.transpose(); }
    initialize_OCl(&ocl_p, kernel_func_name, kernel_file_name, M, N, K, global_work_size, local_work_size);
    double start_t, stop_t;
    if(isTransposed){
      start_t = tic();
      Matrix_Mul(A, B_T, C, &ocl_p);
      stop_t = toc(start_t);
    }else{
      start_t = tic();
      Matrix_Mul(A, B, C, &ocl_p);
      stop_t = toc(start_t);
    }
    std::cout << "total time elapsed using OCl = " << stop_t << " sec" << std::endl;

    // Display the result to the screen
    if(debug_mode){std::cout << "C = \n" << C << '\n';}
    if(C.norm()==0.0){ print_error("norm(C) = 0 !"); }
    // Repeat Mat. multiplication using Eigen
    double start_t_eig = tic();
    C2 = A*B;
    double stop_t_eig = toc(start_t_eig);
    std::cout << "time elapsed using Eigen = " << stop_t_eig << " sec" << std::endl;
    // Display the result to the screen
    if(debug_mode){std::cout << "C = \n" << C2 << '\n';}
    std::cout << "C_ocl - C_eig = " << (C-C2).norm() << '\n';

    // Clean up
    finalize_OCl(&ocl_p);
    return 0;
}
