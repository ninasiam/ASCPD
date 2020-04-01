#ifndef SOLVE_BRASCPACCEL_HPP
#define SOLVE_BRASCPACCEL_HPP

#include "master_library.hpp"
#include "mttkrp.hpp"
#include "sampling_funs.hpp"
#include "omp_lib.hpp"
#include "calc_gradient.hpp"


namespace symmetric
{   
    
    struct struct_mode
    {   

        MatrixXi idxs;
        MatrixXi factor_idxs;
        MatrixXd T_s;
        MatrixXd KR_s; 
        MatrixXd Grad;
        MatrixXd Zero_Matrix;	 

        void struct_mode_init(int m, int n, int k, int r)     //m bs(mode), n order, p prod(dim), k dim, r rank
        {
             idxs = MatrixXi::Zero(m, n); 
             factor_idxs = MatrixXi::Zero(m, n - 1);
             T_s = MatrixXd::Zero(k, m);
             KR_s = MatrixXd::Zero(m, r);     
             Grad = MatrixXd::Zero(k, r);
             Zero_Matrix = MatrixXd::Zero(k, r);
        }

        void destruct_struct_mode()
        {
             idxs.resize(0,0); 
             factor_idxs.resize(0,0);
             T_s.resize(0,0);
             KR_s.resize(0,0);  
             Grad.resize(0,0);
             Zero_Matrix.resize(0,0);

        }

    };


    template <size_t  TNS_ORDER>
    inline void solve_BrasCPaccel( const double AO_tol, const double MAX_MTTKRP, const size_t &R, Eigen::Tensor<double, 0> &frob_X, const VectorXi &tns_dims,
                                   const VectorXi &block_size, std::array<MatrixXd, TNS_ORDER> &Factors, double* Tensor_pointer)
    {   
        int AO_iter = 1;
        int mttkrp_counter = 0;
        int tns_order = Factors.size();                          //TNS_ORDER
        int current_mode;
        double L, beta_accel, lambda;							// NAG parameters
        const unsigned int threads_num = get_num_threads();
        
        
        std::array<MatrixXd, TNS_ORDER> Factors_prev = Factors;                    //Previous values of factors
        std::array<MatrixXd, TNS_ORDER> Y_Factors    = Factors;                    //Factors Y
        //-------------------------------------- Matrix Initializations ---------------------------------------------


        MatrixXd Hessian(R,R);

        //--------------------------> Begin Algorithm <-----------------------------------------------
        cout << " BEGIN ALGORITHM " << endl;
	    high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
        while(1)
        {   
            //Select the current mode
            symmetric::Sample_mode(tns_order, current_mode);
            
            //struct for current mode. contains the matrices for each mode
            symmetric::struct_mode current_mode_struct;
            current_mode_struct.struct_mode_init(block_size(current_mode), tns_order,  tns_dims(current_mode), R);

            //Sample the fibers and take the sampled matricization and the idxs used for the sampling of khatri-rao
            symmetric::Sample_Fibers( Tensor_pointer,  tns_dims,  block_size,  current_mode,
                             current_mode_struct.idxs, current_mode_struct.factor_idxs, current_mode_struct.T_s);

            //Compute the sampled Khatri Rao
            symmetric::Sample_KhatriRao( current_mode, R, current_mode_struct.idxs, Factors, current_mode_struct.KR_s);
            
            //Compute Hessian
            Hessian = current_mode_struct.KR_s.transpose()*current_mode_struct.KR_s;
            
            //Compute Nesterov Parameters
            Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
            
            std::cout << "MPIKE" << endl; 
            //Calculate Gradient
            Calc_gradient( tns_dims, current_mode, threads_num, lambda, Factors_prev[current_mode], Y_Factors[current_mode], Hessian, current_mode_struct.KR_s, current_mode_struct.T_s, current_mode_struct.Grad);
            
           
            //Update factor
            Factors[current_mode] = Y_Factors[current_mode] - current_mode_struct.Grad / (L + lambda);
            Factors[current_mode] = Factors[current_mode].cwiseMax(current_mode_struct.Zero_Matrix);

            if( int(AO_iter % (tns_dims.prod()/block_size(current_mode))) == 0)
            {   
                //Here we will calculate the measure of performance, either CPD_GEN or calculate the norm of a tensor using the factors
                mttkrp_counter++;
            }

            if(mttkrp_counter >= MAX_MTTKRP)
            {
                cout << "Exit Algorithm" << endl;
                break; 
            }

            Factors[current_mode] = Factors_prev[current_mode];
            AO_iter++;
            //delete current_mode_struct;
            current_mode_struct.destruct_struct_mode();
        }

        high_resolution_clock::time_point t2 = high_resolution_clock::now();

	    duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
	    cout << " CPU time = " << stop_t.count() << endl; 
	    cout << " AO_iter = " << AO_iter << endl;
	    cout << " number of threads = " << threads_num << endl << endl;

    }

    
} //end namespace symmetric

#endif //end if