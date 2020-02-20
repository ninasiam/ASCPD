#include "../../include/master_lib.hpp"
#include "../../include/khatri_rao_product.hpp"
#include "../../include/FullKhatriRaoProduct.hpp"
#include "../../include/timers.hpp"

#include "testFullKhatriRaoProduct.hpp"

int main(int argc, char **argv)
{

    const int tensor_order = TNS_ORDER;
    VectorXi tensor_dims(tensor_order);
    int tensor_rank = 20;
    VectorXi constraints(tensor_order);


    constraints.setZero();

    VectorXd temp_tensor_dims(tensor_order);
    temp_tensor_dims = 100 * (temp_tensor_dims.setRandom() + VectorXd::Ones(tensor_order));
    tensor_dims = temp_tensor_dims.cast<int>();

    // tensor_dims.setConstant(100);

    /*--+ Print INFO message +--*/
    std::cout << "> Tensor of -Order \t =  " << tensor_order << "\n\t    -Rank \t = " << tensor_rank << "\n\t    -Dimensions  =";
    for (int mode = 0; mode < tensor_order; mode++)
    {
        std::cout << " " << tensor_dims(mode);
    }
    std::cout << "\n\t    -Constraints = ";
    for (int mode = 0; mode < tensor_order; mode++)
    {
        std::cout << " " << constraints(mode);
    }
    std::cout << std::endl;

    Eigen::Tensor<double, TNS_ORDER> True_Tensor;
    std::array<MatrixXd, tensor_order> True_Factors;
    std::array<MatrixXd, tensor_order> True_Tensor_Mat;

    std::array<MatrixXd, tensor_order> Init_Factors;

    for (int mode = 0; mode < tensor_order; mode++)
    {
        if (constraints(mode) == 0)
        {
            Init_Factors[mode] = MatrixXd::Random(tensor_dims[mode], tensor_rank);
        }
        else if (constraints(mode) == 1) // Nesterov
        {
            Init_Factors[mode] = (MatrixXd::Random(tensor_dims[mode], tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
        }
        else if (constraints(mode) == 2) // Initial Factor for orthogonality constraints...
        {
            Init_Factors[mode] = (MatrixXd::Random(tensor_dims[mode], tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
        }
    }

    std::array<MatrixXd, TNS_ORDER> Khatri_Rao;
    std::array<MatrixXd, TNS_ORDER> Khatri_Rao2;

    for (int mode = 0; mode < tensor_order; mode++)
    {
        int dim = tensor_dims.prod() / tensor_dims(mode);
        Khatri_Rao[mode] = MatrixXd::Zero(tensor_rank, dim);
        Khatri_Rao2[mode] = MatrixXd::Zero(tensor_rank, dim);
    }

    double start_t_khatrirao, stop_t_khatrirao = 0;
    double start_t_khatrirao2, stop_t_khatrirao2 = 0;

    for (int mode = 0; mode < tensor_order; mode++)
    {
        // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
        int dim = tensor_dims.prod() / tensor_dims(mode);

        start_t_khatrirao = tic();

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
            Khatri_Rao_Product(Khatri_Rao[mode], Init_Factors[curr_dim], Khatri_Rao[mode], tensor_rank, rows, get_num_threads());
            rows = rows * Init_Factors[curr_dim].rows();
        }
        stop_t_khatrirao += toc(start_t_khatrirao);

        start_t_khatrirao2 = tic();
        FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, mode, Init_Factors, Khatri_Rao2, get_num_threads());
        stop_t_khatrirao2 += toc(start_t_khatrirao2);

        std::cout << "mode = " << mode << "\t:";
        std::cout << "norm(KhatriRao(method1) - KhatriRao(method2)) = " << (Khatri_Rao[mode] - Khatri_Rao2[mode]).norm() << "\n";
        // std::cout << Khatri_Rao[mode] << "\n--------------------------------------------------------------------------------------------------------\n";
        // std::cout << Khatri_Rao2[mode] << std::endl;
    }
    std::cout << "Elapsed Time using method 1\t = \t" << stop_t_khatrirao << std::endl;
    std::cout << "Elapsed Time using method 2\t = \t" << stop_t_khatrirao2 << std::endl;
}