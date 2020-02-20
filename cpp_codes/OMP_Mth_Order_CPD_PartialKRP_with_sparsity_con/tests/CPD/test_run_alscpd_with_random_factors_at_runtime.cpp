#include "../../include/master_lib.hpp"

#include "../../include/solveALSCPD.hpp"
#include "../../include/createTensorFromFactors.hpp"
#include "../../include/createTensor.hpp"
#include "../../include/createMatricizedTensor.hpp"

#include "test_cpd.hpp"

using namespace Eigen;

/* <----------------------------------------------------------------------------------------------------------> */
/* <----------------------------------------		MAIN		----------------------------------------------> */

int main(int argc, char **argv){

	const int tensor_order = TNS_ORDER;
	VectorXi  tensor_dims(tensor_order);
	int       tensor_rank = 50;
	VectorXi  constraints(tensor_order);

	// Unconstrained Factors:
	constraints.setZero();
	// constraints(3) = 1;

	// Non-negative Factors: Run Nesterov
	// constraints.setConstant(1);
	// constraints(2) = 0; // Set one unconstrained

	// Orthogonality Constraints...
	// constraints.setConstant(2);
	// constraints(3) = 2;
	
	// VectorXd temp_tensor_dims(tensor_order);
	// temp_tensor_dims = 80 * (temp_tensor_dims.setRandom() + VectorXd::Ones(tensor_order));
	// tensor_dims = temp_tensor_dims.cast <int> (); 

	tensor_dims.setConstant(400);

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

	Eigen::Tensor<double, TNS_ORDER> True_Tensor;
	std::array<MatrixXd, tensor_order> True_Factors;
	std::array<MatrixXd, tensor_order> True_Tensor_Mat;

	std::array<MatrixXd, tensor_order> Init_Factors;

	// srand(5); // Initialize Seed

	/*--+ Create each factor A_(n) +--*/
    for (int mode = 0; mode < tensor_order; mode++)
    {
        if (constraints(mode) == 0)
        {
            #ifdef FACTORS_ARE_TRANSPOSED
            MatrixXd tmp = MatrixXd::Random(tensor_dims(mode), tensor_rank);
            True_Factors[mode] = tmp.transpose(); // MatrixXd::Random(tensor_rank, tensor_dims(mode));
            #else
            True_Factors[mode] = MatrixXd::Random(tensor_dims(mode), tensor_rank);
            #endif
        }
        else if (constraints(mode) == 1) // Nesterov
        {
            #ifdef FACTORS_ARE_TRANSPOSED
            MatrixXd tmp = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
            True_Factors[mode] = tmp.transpose();
            #else
            True_Factors[mode] = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
            #endif
        }
        else if (constraints(mode) == 2)
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

	/*--+ Create each matr. tensor X_(n) +--*/
	create_tensor_from_factors(tensor_order, tensor_dims, tensor_rank, True_Factors, True_Tensor_Mat, get_num_threads());

	// createTensor(tensor_order, tensor_dims, tensor_rank, True_Tensor, True_Factors);
	// createMatricizedTensor(tensor_order, tensor_dims, True_Tensor, True_Tensor_Mat);
	
	/*--+ Initialize Factors +--*/
	double epsilon = 1.0e-4;
    for (int mode = 0; mode < tensor_order; mode++)
    {
        if (constraints(mode) == 0)
        {
            #ifdef FACTORS_ARE_TRANSPOSED
			MatrixXd tmpFactor = MatrixXd::Random(tensor_dims(mode), tensor_rank);
			Init_Factors[mode] = tmpFactor.transpose();
			#else
			Init_Factors[mode] = MatrixXd::Random(tensor_dims(mode), tensor_rank);
            #endif
        }
        else if (constraints(mode) == 1) // Nesterov
        {
            #ifdef FACTORS_ARE_TRANSPOSED
			MatrixXd tmpFactor = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
			Init_Factors[mode] = tmpFactor.transpose();
			#else
			Init_Factors[mode] = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
            #endif
        }
		else if (constraints(mode) == 2) // Orthogonality constraints
		{
			MatrixXd tmpFactor = (MatrixXd::Random(tensor_dims(mode), tensor_rank) + MatrixXd::Ones(tensor_dims(mode), tensor_rank)) / 2;
			JacobiSVD<MatrixXd> svd(tmpFactor, ComputeThinU | ComputeThinV);
			
			#ifdef FACTORS_ARE_TRANSPOSED
			Init_Factors[mode] = (svd.matrixU()).transpose();
			#else
			Init_Factors[mode] = svd.matrixU();
			#endif
		}
        else
        {
            std::cerr << "Unknown type of constraint for Factor of mode = " << mode << std::endl;
            return -1;
        }
        // std::cout << True_Factors[mode] << "\n" << std::endl;
    }

	/*--+ Run CPD - ALS +--*/
	int iter = 0;
	int MAX_ITER = 10;
	double cpd_err = 0;

	solveALSCPD(tensor_order, tensor_dims, tensor_rank, constraints, Init_Factors, True_Tensor_Mat, cpd_err, iter, MAX_ITER, epsilon);

	return 0;
}														// End of main
