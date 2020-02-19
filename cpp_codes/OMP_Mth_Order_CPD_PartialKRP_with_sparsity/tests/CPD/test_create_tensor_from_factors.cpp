#include "../../include/master_lib.hpp"

#include "../../include/solveALSCPD.hpp"
#include "../../include/createTensorFromFactors.hpp"
#include "../../include/createTensor.hpp"
#include "../../include/createMatricizedTensor.hpp"
#include "../../include/ReadWrite.hpp"

#include "test_cpd.hpp"

using namespace Eigen;

/* <----------------------------------------------------------------------------------------------------------> */
/* <----------------------------------------		MAIN		----------------------------------------------> */

int main(int argc, char **argv)
{
    const int tensor_order = TNS_ORDER;
    VectorXi tensor_dims(tensor_order);
    int tensor_rank = 20;
    VectorXi constraints(tensor_order);

    // Unconstrained Factors:
    // constraints.setZero();
    // constraints(3) = 1;

    // Non-negative Factors: Run Nesterov
    constraints.setConstant(1);
    // constraints(2) = 0; // Set one unconstrained

    // Orthogonality Constraints...
    // constraints.setConstant(2);
    // constraints(3) = 2;

    // VectorXd temp_tensor_dims(tensor_order);
    // temp_tensor_dims = 70 * (temp_tensor_dims.setRandom() + VectorXd::Ones(tensor_order));
    // tensor_dims = temp_tensor_dims.cast<int>();

    tensor_dims.setConstant(100);

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
    //

    std::array<MatrixXd, tensor_order> True_Factors;
    /*--+ Initialize Seed +--*/
    // srand((unsigned int)tensor_dims.prod());
    // srand(5);

    /*--+ Create each factor A_(n) +--*/
    for (int mode = 0; mode < tensor_order; mode++)
    {
        if (constraints(mode) == 0)
        {
            #ifdef FACTORS_ARE_TRANSPOSED
			MatrixXd tmpFactor = MatrixXd::Random(tensor_dims(mode), tensor_rank);
			True_Factors[mode] = tmpFactor.transpose();
            #else
            True_Factors[mode] = MatrixXd::Random(tensor_dims(mode), tensor_rank);
            #endif
        }
        else if (constraints(mode) == 1) // Nesterov
        {
            #ifdef FACTORS_ARE_TRANSPOSED
			MatrixXd tmpFactor = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
			True_Factors[mode] = tmpFactor.transpose();
            #else
			True_Factors[mode] = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
            #endif
        }
        else if (constraints(mode) == 2) // Orthogonality constraints
        {
			MatrixXd tmpFactor = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
            JacobiSVD<MatrixXd> svd(tmpFactor, ComputeThinU | ComputeThinV);
			
			#ifdef FACTORS_ARE_TRANSPOSED
			True_Factors[mode] = (svd.matrixU()).transpose();
			#else
            True_Factors[mode] = svd.matrixU();
			#endif
		}
        else
        {
            std::cerr << "Unknown type of constraint for Factor of mode = " << mode << std::endl;
            return -1;
        }
        // std::cout << True_Factors[mode] << "\n" << std::endl;
    }

    int mode = 0;
    Eigen::MatrixXd Mode1Matricization = Eigen::MatrixXd(tensor_dims(0), tensor_dims.prod() / tensor_dims(0));
    
    // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
    int dim = tensor_dims.prod() / tensor_dims(mode);

    MatrixXd Khatri_Rao = MatrixXd::Zero(tensor_rank, dim);

    FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, mode, True_Factors, Khatri_Rao, get_num_threads());
    // True_Tensor_Mat[mode] = True_Factors[mode] * Khatri_Rao.transpose();
    Mode1Matricization = MatrixXd::Zero(tensor_dims(mode), dim);

    ftkrp(True_Factors[mode], Khatri_Rao, Mode1Matricization, tensor_dims, mode, get_num_threads());

    /*--+ Write X_(1) to file +--*/
    std::string filename = "Tensor.dat";
    std::cout << "Writing Tensor to file..." << std::endl;

    writeEigenDataToFile(filename, Mode1Matricization);
}