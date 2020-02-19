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

int main(int argc, char **argv){

	const int tensor_order = TNS_ORDER;
	VectorXi  tensor_dims(tensor_order);
	int       tensor_rank = 20;
	VectorXi  constraints(tensor_order);

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
	// tensor_dims = temp_tensor_dims.cast <int> ();

	tensor_dims.setConstant(100);

	/*--+ Print INFO message +--*/
	std::cout << "> Tensor of -Order \t =  " << tensor_order << "\n\t    -Rank \t = " << tensor_rank << "\n\t    -Dimensions  =";
	for(int mode=0; mode < tensor_order; mode++)
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

	std::array<MatrixXd, tensor_order> True_Tensor_Mat;

	std::array<MatrixXd, tensor_order> Init_Factors;

	/*--+ Read X_(1) from file +--*/
	std::string filename = "Tensor.dat";
	std::cout << "Reading Tensor from file..." << std::endl;

	/*--+ Store DATA from file to mode-1 Matricization +--*/
	True_Tensor_Mat[0] = Eigen::MatrixXd(tensor_dims(0), tensor_dims.prod() / tensor_dims(0));
	readEigenDataFromFile(filename, True_Tensor_Mat[0]);

	createTensor(tensor_order, tensor_dims, tensor_rank, True_Tensor, True_Tensor_Mat[0]);
	createMatricizedTensor(tensor_order, tensor_dims, True_Tensor, True_Tensor_Mat);

	/*--+ Initialize Factors +--*/
	srand( (unsigned int) tensor_dims.prod() ); // Initialize Seed

	double epsilon = 1.0e-4;
	/*--+ Create each factor A_(n) +--*/
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
	int MAX_ITER = 200;
	double cpd_err = 0;

	solveALSCPD(tensor_order, tensor_dims, tensor_rank, constraints, Init_Factors, True_Tensor_Mat, cpd_err, iter, MAX_ITER, epsilon);

	/*--+ Write Factors to files +--*/
	std::cout << "Writing all Factors to files..." << std::endl;
	for (int mode=0; mode < tensor_order; mode++)
	{
		std::string prefix = "factor";
		std::string str_mode = std::to_string(mode);
		std::string file_extension = ".dat";
		std::string f_out_name = prefix + str_mode + file_extension;
		#ifdef FACTORS_ARE_TRANSPOSED
		MatrixXd tmpFactor = Init_Factors[mode].transpose();
		writeEigenDataToFile(f_out_name, tmpFactor);
		#else
		writeEigenDataToFile(f_out_name, Init_Factors[mode]);
		#endif
	}
	return 0;
}														// End of main
