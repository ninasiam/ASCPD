#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"

int main(int argc, char **argv){

    
    const int TNS_ORDER = 3;                                      //Declarations
    const int R = 2;
    
    VectorXi  tns_dims(TNS_ORDER);
    VectorXi  block_size(TNS_ORDER);
    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;

     //Initialize the tensor
    tns_dims.setConstant(4); 
    True_Tensor.resize(tns_dims);
    True_Tensor.setConstant(1.0f);
    True_Tensor = True_Tensor + True_Tensor.random();
    
                                        
    block_size.setConstant(2);

    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;

    Eigen::Tensor< double, 0 > frob_X  = True_Tensor.square().sum().sqrt();
    
    cout << "Frob_X:"  << frob_X << endl; 

    cout << "True Tensor: \n" <<True_Tensor << endl;

    double* Tensor_pointer = True_Tensor.data();

    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = MatrixXd::Random(tns_dims(factor), R);
    }

    // srand((unsigned) time(0));
    int mode;
    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    MatrixXd T_mode(tns_dims(mode), block_size(mode));

    
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                             idxs, factor_idxs, T_mode);
    cout << "Matricization mode  " << mode <<  " = \n " << T_mode << endl;

    MatrixXd KR_sampled(block_size(mode), R);
    
   

    symmetric::Sample_KhatriRao( mode, R, idxs, Init_Factors, KR_sampled);

    cout << "KR_sampled   " << " = \n " << KR_sampled << endl;

    MatrixXd KR_full(tns_dims(1)*tns_dims(0), R);
    Khatri_Rao_Product( Init_Factors[1], Init_Factors[0], KR_full);
    cout << "KR_full   " << " = \n " << KR_full << endl;

    double AO_tol = 0.001;
    int MAX_MTTKRP = 10;
    
    symmetric::solve_BrasCPaccel(AO_tol, MAX_MTTKRP, R, frob_X, tns_dims, block_size, Init_Factors, Tensor_pointer);
    return 0;
}