#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"

int main(int argc, char **argv){

    Eigen::Tensor<double, 4> True_Tensor(2,2,2,2);
    True_Tensor.setConstant(1.0f);

    True_Tensor = True_Tensor + True_Tensor.random();
    cout << True_Tensor << endl;
    // cout << True_Tensor(0,0,0) << endl;
    // cout << True_Tensor(1,0,0) << endl;

    // cout << True_Tensor(0,1,0) << endl;
    // cout << True_Tensor(1,1,0) << endl;

    // cout << True_Tensor(0,0,1) << endl;
    // cout << True_Tensor(0,1,1) << endl;

    // cout << True_Tensor(1,1,1) << endl;

    

    double* Tensor_pointer = True_Tensor.data();
    // cout << "Tensor_pointer[5] = " << Tensor_pointer[5] << endl;

    // //sampling fibers mode_1
    // //Sample fiber (:,0,0)

    // VectorXd fiber1(2,1);
    // for(int i = 0; i < 2; i++)
    // {
    //     fiber1(i) = *(Tensor_pointer + i);
    // }
    // cout << "fiber1" <<fiber1 << endl;

    // //Sample fiber (:,0,0) using linear indexing
    // VectorXd fiber2(2,1);
    // for(int j = 0; j < 2; j++)
    // {
    //     fiber2(j) = Tensor_pointer[j*1 + 0*2 + 0*(2*2)];
    // }
    // cout << "fiber2" <<fiber2 << endl;

    // //Sample fiber (:,1,1)

    // VectorXd fiber3(2,1);
    // for(int i = 0; i < 2; i++)
    // {
    //     fiber3(i) = *(Tensor_pointer + i);
    // }
    // cout << "fiber3" <<fiber3 << endl;

    // //Sample fiber (:,1,1) using linear indexing
    // VectorXd fiber4(2,1);
    // for(int j = 0; j < 2; j++)
    // {
    //     fiber4(j) = Tensor_pointer[j*1 + 1*2 + 1*(2*2)];
    // }
    // cout << "fiber4" <<fiber4 << endl;

    // //Sample fiber (0,:,1) using linear indexing
    // VectorXd fiber5(2,1);
    // for(int j = 0; j < 2; j++)
    // {
    //     fiber5(j) = Tensor_pointer[0*1 + j*2 + 1*(2*2)];
    // }
    // cout << "fiber5" <<fiber5 << endl;

    // //Sample fiber (1,1,:) using linear indexing
    // VectorXd fiber6(2,1);
    // for(int j = 0; j < 2; j++)
    // {
    //     fiber6(j) = Tensor_pointer[1*1 + 1*2 + j*(2*2)];
    // }
    // cout << "fiber6" <<fiber6 << endl;



    VectorXi tns_dims(4,1), block_size(4,1);
    tns_dims.setConstant(2);
    block_size.setConstant(1);
    
    MatrixXi idxs(block_size(0),4);
    MatrixXi factor_idxs(block_size(0),3);
    int mode = 3;
    MatrixXd T_mode(tns_dims(mode), block_size(mode));
    

    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                      idxs, factor_idxs, T_mode);

    cout << "Matricization mode 3 = \n " << T_mode << endl;


    free(Tensor_pointer);
    return 0;

}