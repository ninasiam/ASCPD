#include "../../include/master_lib.hpp"
#include "../../include/ReadWrite.hpp"

#include "testReadWrite.hpp"

#include <string>

/* TEST: writeMode1Matricization(...) and readMode1Matricization(...)  */
int main(int argc, char **argv)
{
    const int       tensor_rank = 5;
    Eigen::VectorXi tensor_dims(TNS_ORDER);
    Eigen::VectorXd temp_tensor_dims(TNS_ORDER);
    temp_tensor_dims = 50 * (temp_tensor_dims.setRandom() + VectorXd::Ones(TNS_ORDER));
    tensor_dims = temp_tensor_dims.cast<int>();
    
    std::cout << "Tensor dims = \n" << tensor_dims << std::endl;

    // Allocate Output Matrix.
    Eigen::MatrixXd WriteMaticizedTensor(tensor_dims(0), tensor_dims.prod()/tensor_dims(0));

    // Initiallize Output Matrix.
    WriteMaticizedTensor.setRandom();

    std::string filename = "file.dat";

    // Write Matricized Tensor to file.
    writeEigenDataToFile(filename, WriteMaticizedTensor);

    // Allocate Input Matrix.
    Eigen::MatrixXd ReadMaticizedTensor(tensor_dims(0), tensor_dims.prod() / tensor_dims(0));

    // Read Matricized Tensor from file.
    readEigenDataFromFile(filename, ReadMaticizedTensor);

    std::cout << "norm(R - W) = " << (ReadMaticizedTensor - WriteMaticizedTensor).norm() << std::endl;

//     std::cout << std::endl << ReadMaticizedTensor << std::endl
//               << std::endl << WriteMaticizedTensor << std::endl;
}