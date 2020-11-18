#ifndef SOLVE_BRASCPACCEL_HPP
#define SOLVE_BRASCPACCEL_HPP

#include "master_library.hpp"
#include "mttkrp.hpp"
#include "sampling_funs.hpp"
#include "omp_lib.hpp"
#include "calc_gradient.hpp"
#include "cpdgen.hpp"
#include "compute_fval.hpp"

using namespace std::chrono_literals;

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

        void struct_mode_init(int m, int n, int k, int r)     //m bs(mode), n order, k dim, r rank
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


    template <std::size_t  TNS_ORDER>
    inline void solve_BrasCPaccel( const double AO_tol, const double MAX_MTTKRP, const int &R, const Eigen::Tensor<double, 0> &frob_X, Eigen::Tensor<double, 0> &f_value, const VectorXi &tns_dims,
                                   const VectorXi &block_size, std::array<MatrixXd, TNS_ORDER> &Factors, double* Tensor_pointer, 
                                   const Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > &True_Tensor)
    {   
        int AO_iter = 1;
        int print_iter = 1;
        int mttkrp_counter = 0;
        int tns_order = Factors.size();                          //TNS_ORDER
        int current_mode;
        double L, beta_accel, lambda;							// NAG parameters
        const unsigned int threads_num = get_num_threads();

        // Timers
        nanoseconds stop_t_cpdgen = 0ns;
        nanoseconds stop_t_MTTKRP = 0ns;
        nanoseconds stop_t_Ts = 0ns;
        nanoseconds stop_t_KRs = 0ns;
        nanoseconds stop_t_fval = 0ns;
        nanoseconds stop_t_cal_grad = 0ns;
        nanoseconds stop_t_struct = 0ns;
        nanoseconds stop_t_NAG = 0ns;
        
        Eigen::Tensor< double, static_cast<int>(TNS_ORDER) >  Est_Tensor_from_factors;                // with no dims, to calculate cost fun
        std::array<MatrixXd, TNS_ORDER> Factors_prev = Factors;                                       //Previous values of factors
        std::array<MatrixXd, TNS_ORDER> Y_Factors    = Factors;                                       //Factors Y

        #if USE_COST_FUN
            Eigen::Tensor< double, 2>::Dimensions two_dim (tns_dims(0), tns_dims.prod()/tns_dims(0));
            Eigen::Tensor< double, 2 > Mat_Tensor = True_Tensor.reshape(two_dim);                         // Does not modify true tensor
            MatrixXd X_mat_0(tns_dims(0), tns_dims.prod()/tns_dims(0));
            X_mat_0 = Eigen::Map<Eigen::MatrixXd>(Mat_Tensor.data(), tns_dims(0), tns_dims.prod()/tns_dims(0));
            MatrixXd MTTKRP_0(tns_dims(0), R);
            MatrixXd gram_cwise_prod(R, R);
        #endif
        //-------------------------------------- Matrix Initializations ---------------------------------------------
        MatrixXd Hessian(R,R);
        
        // Eigen::Tensor< double, 0 > frob_X_sq = frob_X.square();
        double frob_X_sq_d = frob_X(0);
        //--------------------------> Begin Algorithm <-----------------------------------------------
        cout << "--------------------------- BEGIN ALGORITHM --------------------------- " << endl;
        cout << AO_iter << "   " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X <<  endl;
        // Begin Timer for the While
	    auto t1 = high_resolution_clock::now();
        
        while(1)
        {   
            //Select the current mode
            symmetric::Sample_mode(tns_order, current_mode);

            //struct for current mode. contains the matrices for each mode
            auto t1_struct  = high_resolution_clock::now();
            symmetric::struct_mode current_mode_struct;
            current_mode_struct.struct_mode_init(block_size(current_mode), tns_order,  tns_dims(current_mode), R);
            auto t2_struct = high_resolution_clock::now();
            stop_t_struct += duration_cast<nanoseconds>(t2_struct - t1_struct);


            //Sample the fibers and take the sampled matricization and the idxs used for the sampling of khatri-rao
            auto t1_Ts = high_resolution_clock::now();
            symmetric::Sample_Fibers( Tensor_pointer,  tns_dims,  block_size,  current_mode,
                             current_mode_struct.idxs, current_mode_struct.factor_idxs, current_mode_struct.T_s);
            auto t2_Ts = high_resolution_clock::now();
            stop_t_Ts += duration_cast<nanoseconds>(t2_Ts - t1_Ts);

            //Compute the sampled Khatri Rao
            auto t1_KRs = high_resolution_clock::now();
            symmetric::Sample_KhatriRao( current_mode, R, current_mode_struct.idxs, Factors_prev, current_mode_struct.KR_s);
            auto t2_KRs = high_resolution_clock::now();
            stop_t_KRs += duration_cast<nanoseconds>(t2_KRs - t1_KRs);

            //Compute Hessian
            Hessian.noalias() = current_mode_struct.KR_s.transpose()*current_mode_struct.KR_s;

            //Compute Nesterov Parameters
            auto t1_NAG = high_resolution_clock::now();
            Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
            auto t2_NAG = high_resolution_clock::now();
            stop_t_NAG += duration_cast<nanoseconds>(t2_NAG - t1_NAG);

            //Calculate Gradient
            auto t1_cal_grad = high_resolution_clock::now();
            // This is by default to parallel you can change it inside
            parallel_MTTKRP::Calc_gradient( tns_dims, current_mode, threads_num, lambda, Factors_prev[current_mode], Y_Factors[current_mode], Hessian, current_mode_struct.KR_s, current_mode_struct.T_s, current_mode_struct.Grad, stop_t_MTTKRP);
            auto t2_cal_grad = high_resolution_clock::now();
            stop_t_cal_grad += duration_cast<nanoseconds>(t2_cal_grad - t1_cal_grad);

            //Update factor
            Factors[current_mode]   = Y_Factors[current_mode] - current_mode_struct.Grad /(L + lambda);
            Factors[current_mode]   = Factors[current_mode].cwiseMax(current_mode_struct.Zero_Matrix);

            //Update Y
            Y_Factors[current_mode] = Factors[current_mode] + beta_accel*(Factors[current_mode] - Factors_prev[current_mode]);

            Factors_prev[current_mode] = Factors[current_mode];


            if( int(AO_iter % ((TNS_ORDER + 1)*(((tns_dims.prod()/tns_dims(current_mode))/block_size(current_mode))))) == 0)
            {                   
                print_iter++;
                #if USE_COST_FUN
                    auto t1_fval = high_resolution_clock::now();
                    partial::mttpartialkrp<TNS_ORDER>(tns_order, tns_dims, R, 0, Factors_prev, X_mat_0, MTTKRP_0, threads_num);
                    gram_cwise_prod.setOnes();
                    compute_gram_cwise_prod(Factors_prev, gram_cwise_prod);
                    compute_fval(frob_X_sq_d, MTTKRP_0, gram_cwise_prod, Factors_prev[0], f_value(0));
                    f_value(0) = f_value(0) / frob_X_sq_d;
                    auto t2_fval = high_resolution_clock::now();
                    stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                    cout << print_iter<< "  " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                #else
                    // Here we calculate the measure of performance, either CPD_GEN or calculate the norm of a tensor using the factors
                    auto t1_cpdgen = high_resolution_clock::now();
                    CpdGen( tns_dims, Factors_prev, R, Est_Tensor_from_factors);
                    auto t2_cpdgen = high_resolution_clock::now();
                    stop_t_cpdgen += duration_cast<nanoseconds>(t2_cpdgen-t1_cpdgen);

                    // f_value computation
                    auto t1_fval = high_resolution_clock::now();
                    f_value = (True_Tensor - Est_Tensor_from_factors).square().sum();  
                    auto t2_fval = high_resolution_clock::now();
                    stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                    cout << print_iter<< "  " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                #endif

                mttkrp_counter += (TNS_ORDER + 1); // Increase counter according to printing (after TNS_ORDER + 1 (Accel. step)).
            }

            if(mttkrp_counter >= MAX_MTTKRP)
            {
                cout << "---------------------------- EXIT ALGORITHM --------------------------- " << endl;
                break; 
            }
            

            AO_iter++;
            //delete current_mode_struct;
            auto t1_struct_d  = high_resolution_clock::now();
            current_mode_struct.destruct_struct_mode();
            auto t2_struct_d  = high_resolution_clock::now();
            stop_t_struct += duration_cast<nanoseconds>(t2_struct_d - t1_struct_d);


        }

        auto t2 = high_resolution_clock::now();
        // End Timer for the While

	    duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
        // seconds stop_t = duration_cast<std::chrono::seconds>(t2 - t1);
	    cout << "CPU time WHILE = " << stop_t.count() << "s" <<endl; 
        cout << "Time CPDGEN    = " << stop_t_cpdgen.count() * (1e-9) << "s" << endl; 
        cout << "Time MTTKRP    = " << stop_t_MTTKRP.count() * (1e-9) << "s" << endl;
        cout << "Time sample T  = " << stop_t_Ts.count() * (1e-9) << "s" << endl;
        cout << "Time sample KR = " << stop_t_KRs.count() * (1e-9) << "s" << endl;
        cout << "Time f_value   = " << stop_t_fval.count() * (1e-9) << "s" << endl;
        cout << "Time grad      = " << (stop_t_cal_grad.count() - stop_t_MTTKRP.count()) * (1e-9) << "s" << endl;
        cout << "Time NAG par   = " << stop_t_NAG.count() * (1e-9) << "s" << endl;
        cout << "Time struct    = " << stop_t_struct.count() * (1e-9) << "s" << endl;
	    cout << "AO_iter        = " << AO_iter << endl;
	    cout << "num of threads = " << threads_num << endl << endl;

    }

    
} //end namespace symmetric

namespace sorted
{   
    
    struct struct_mode
    {   

        MatrixXi idxs;
        MatrixXd T_s;
        MatrixXd KR_s; 
        MatrixXd Grad;
        MatrixXd Zero_Matrix;	 

        void struct_mode_init(int m, int n, int k, int r)     //m bs(mode), n order, k dim, r rank
        {
             idxs = MatrixXi::Zero(m, n); 
             T_s = MatrixXd::Zero(k, m);
             KR_s = MatrixXd::Zero(m, r);     
             Grad = MatrixXd::Zero(k, r);
             Zero_Matrix = MatrixXd::Zero(k, r);
        }

        void destruct_struct_mode()
        {
             idxs.resize(0,0); 
             T_s.resize(0,0);
             KR_s.resize(0,0);  
             Grad.resize(0,0);
             Zero_Matrix.resize(0,0);

        }

    };


    template <std::size_t  TNS_ORDER>
    inline void solve_BrasCPaccel( const double AO_tol, const double MAX_MTTKRP, const int &R, const Eigen::Tensor<double, 0> &frob_X, Eigen::Tensor<double, 0> &f_value, const VectorXi &tns_dims,
                                   const VectorXi &block_size, std::array<MatrixXd, TNS_ORDER> &Factors, double* Tensor_pointer, 
                                   const Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > &True_Tensor)
    {   
        int AO_iter = 1;
        int print_iter = 1;
        int mttkrp_counter = 0;
        int tns_order = Factors.size();                          //TNS_ORDER
        int current_mode;
        double L, beta_accel, lambda;							// NAG parameters
        const unsigned int threads_num = get_num_threads();

        // Timers
        nanoseconds stop_t_cpdgen = 0ns;
        nanoseconds stop_t_MTTKRP = 0ns;
        nanoseconds stop_t_Ts = 0ns;
        nanoseconds stop_t_KRs = 0ns;
        nanoseconds stop_t_fval = 0ns;
        nanoseconds stop_t_cal_grad = 0ns;
        nanoseconds stop_t_struct = 0ns;
        nanoseconds stop_t_NAG = 0ns;
        
        Eigen::Tensor< double, static_cast<int>(TNS_ORDER) >  Est_Tensor_from_factors;                // with no dims, to calculate cost fun
        std::array<MatrixXd, TNS_ORDER> Factors_prev = Factors;                                       //Previous values of factors
        std::array<MatrixXd, TNS_ORDER> Y_Factors    = Factors;                                       //Factors Y

        #if USE_COST_FUN
            Eigen::Tensor< double, 2>::Dimensions two_dim (tns_dims(0), tns_dims.prod()/tns_dims(0));
            Eigen::Tensor< double, 2 > Mat_Tensor = True_Tensor.reshape(two_dim);                         // Does not modify true tensor
            MatrixXd X_mat_0(tns_dims(0), tns_dims.prod()/tns_dims(0));
            X_mat_0 = Eigen::Map<Eigen::MatrixXd>(Mat_Tensor.data(), tns_dims(0), tns_dims.prod()/tns_dims(0));
            MatrixXd MTTKRP_0(tns_dims(0), R);
            MatrixXd gram_cwise_prod(R, R);
        #endif
        //-------------------------------------- Matrix Initializations ---------------------------------------------
        MatrixXd Hessian(R,R);
        
        // Eigen::Tensor< double, 0 > frob_X_sq = frob_X.square();
        double frob_X_sq_d = frob_X(0);
        //--------------------------> Begin Algorithm <-----------------------------------------------
        cout << "--------------------------- BEGIN ALGORITHM --------------------------- " << endl;
        cout << AO_iter << "   " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X <<  endl;
        // Begin Timer for the While
	    auto t1 = high_resolution_clock::now();
        
        while(1)
        {   
            //Select the current mode
            sorted::Sample_mode(tns_order, current_mode);

            //struct for current mode. contains the matrices for each mode
            auto t1_struct  = high_resolution_clock::now();
            sorted::struct_mode current_mode_struct;
            current_mode_struct.struct_mode_init(block_size(current_mode), tns_order,  tns_dims(current_mode), R);
            auto t2_struct = high_resolution_clock::now();
            stop_t_struct += duration_cast<nanoseconds>(t2_struct - t1_struct);


            //Sample the fibers and take the sampled matricization and the idxs used for the sampling of khatri-rao
            auto t1_Ts = high_resolution_clock::now();
            sorted::Sample_fibers<TNS_ORDER>( Tensor_pointer,  tns_dims,  block_size,  current_mode,
                             current_mode_struct.idxs, current_mode_struct.T_s);
            auto t2_Ts = high_resolution_clock::now();
            stop_t_Ts += duration_cast<nanoseconds>(t2_Ts - t1_Ts);

            //Compute the sampled Khatri Rao
            auto t1_KRs = high_resolution_clock::now();
            symmetric::Sample_KhatriRao( current_mode, R, current_mode_struct.idxs, Factors_prev, current_mode_struct.KR_s); // !!!! THIS SHOULD BE SORTED !!!
            auto t2_KRs = high_resolution_clock::now();
            stop_t_KRs += duration_cast<nanoseconds>(t2_KRs - t1_KRs);

            //Compute Hessian
            Hessian.noalias() = current_mode_struct.KR_s.transpose()*current_mode_struct.KR_s;

            //Compute Nesterov Parameters
            auto t1_NAG = high_resolution_clock::now();
            Compute_NAG_parameters(Hessian, L, beta_accel, lambda);
            auto t2_NAG = high_resolution_clock::now();
            stop_t_NAG += duration_cast<nanoseconds>(t2_NAG - t1_NAG);

            //Calculate Gradient
            auto t1_cal_grad = high_resolution_clock::now();
            parallel_MTTKRP::Calc_gradient( tns_dims, current_mode, threads_num, lambda, Factors_prev[current_mode], Y_Factors[current_mode], Hessian, current_mode_struct.KR_s, current_mode_struct.T_s, current_mode_struct.Grad, stop_t_MTTKRP);
            auto t2_cal_grad = high_resolution_clock::now();
            stop_t_cal_grad += duration_cast<nanoseconds>(t2_cal_grad - t1_cal_grad);

            //Update factor
            Factors[current_mode]   = Y_Factors[current_mode] - current_mode_struct.Grad /(L + lambda);
            Factors[current_mode]   = Factors[current_mode].cwiseMax(current_mode_struct.Zero_Matrix);

            //Update Y
            Y_Factors[current_mode] = Factors[current_mode] + beta_accel*(Factors[current_mode] - Factors_prev[current_mode]);

            Factors_prev[current_mode] = Factors[current_mode];


            if( int(AO_iter % ((TNS_ORDER + 1)*(((tns_dims.prod()/tns_dims(current_mode))/block_size(current_mode))))) == 0)
            {                   
                print_iter++;
                #if USE_COST_FUN
                    auto t1_fval = high_resolution_clock::now();
                    partial::mttpartialkrp<TNS_ORDER>(tns_order, tns_dims, R, 0, Factors_prev, X_mat_0, MTTKRP_0, threads_num);
                    gram_cwise_prod.setOnes();
                    compute_gram_cwise_prod(Factors_prev, gram_cwise_prod);
                    compute_fval(frob_X_sq_d, MTTKRP_0, gram_cwise_prod, Factors_prev[0], f_value(0));
                    f_value(0) = f_value(0) / frob_X_sq_d;
                    auto t2_fval = high_resolution_clock::now();
                    stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                    cout << print_iter << "  " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                #else
                    // Here we calculate the measure of performance, either CPD_GEN or calculate the norm of a tensor using the factors
                    auto t1_cpdgen = high_resolution_clock::now();
                    CpdGen( tns_dims, Factors_prev, R, Est_Tensor_from_factors);
                    auto t2_cpdgen = high_resolution_clock::now();
                    stop_t_cpdgen += duration_cast<nanoseconds>(t2_cpdgen-t1_cpdgen);

                    // f_value computation
                    auto t1_fval = high_resolution_clock::now();
                    f_value = (True_Tensor - Est_Tensor_from_factors).square().sum();  
                    auto t2_fval = high_resolution_clock::now();
                    stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                    cout << print_iter<< "  " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                #endif

                mttkrp_counter += (TNS_ORDER + 1); // Increase counter according to printing (after TNS_ORDER + 1 (Accel. step)).
            }

            if(mttkrp_counter >= MAX_MTTKRP)
            {
                cout << "---------------------------- EXIT ALGORITHM --------------------------- " << endl;
                break; 
            }
            

            AO_iter++;
            //delete current_mode_struct;
            auto t1_struct_d  = high_resolution_clock::now();
            current_mode_struct.destruct_struct_mode();
            auto t2_struct_d  = high_resolution_clock::now();
            stop_t_struct += duration_cast<nanoseconds>(t2_struct_d - t1_struct_d);


        }

        auto t2 = high_resolution_clock::now();
        // End Timer for the While

	    duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
        // seconds stop_t = duration_cast<std::chrono::seconds>(t2 - t1);
	    cout << "CPU time WHILE = " << stop_t.count() << "s" <<endl; 
        cout << "Time CPDGEN    = " << stop_t_cpdgen.count() * (1e-9) << "s" << endl; 
        cout << "Time MTTKRP    = " << stop_t_MTTKRP.count() * (1e-9) << "s" << endl;
        cout << "Time sample T  = " << stop_t_Ts.count() * (1e-9) << "s" << endl;
        cout << "Time sample KR = " << stop_t_KRs.count() * (1e-9) << "s" << endl;
        cout << "Time f_value   = " << stop_t_fval.count() * (1e-9) << "s" << endl;
        cout << "Time grad      = " << (stop_t_cal_grad.count() - stop_t_MTTKRP.count()) * (1e-9) << "s" << endl;
        cout << "Time NAG par   = " << stop_t_NAG.count() * (1e-9) << "s" << endl;
        cout << "Time struct    = " << stop_t_struct.count() * (1e-9) << "s" << endl;
	    cout << "AO_iter        = " << AO_iter << endl;
	    cout << "num of threads = " << threads_num << endl << endl;

    }

    
}

namespace parallel
{   
    
    struct struct_mode
    {   

        MatrixXi idxs;
        MatrixXi factor_idxs;
        MatrixXd T_s;
        MatrixXd KR_s; 
        MatrixXd Grad;
        MatrixXd Zero_Matrix;	 

        void struct_mode_init(int m, int n, int k, int r)     //m bs(mode), n order, k dim, r rank
        {
             idxs = MatrixXi::Zero(m, n); 
             T_s = MatrixXd::Zero(k, m);
             KR_s = MatrixXd::Zero(m, r);     
             Grad = MatrixXd::Zero(k, r);
             Zero_Matrix = MatrixXd::Zero(k, r);
        }

        void destruct_struct_mode()
        {
             idxs.resize(0,0); 
             T_s.resize(0,0);
             KR_s.resize(0,0);  
             Grad.resize(0,0);
             Zero_Matrix.resize(0,0);

        }

    };


    template <std::size_t  TNS_ORDER>
    inline void solve_BrasCPaccel( const double AO_tol, const double MAX_MTTKRP, const int R, const Eigen::Tensor<double, 0> frob_X, Eigen::Tensor<double, 0> &f_value, const VectorXi &tns_dims,
                                   const VectorXi &block_size, std::array<MatrixXd, TNS_ORDER> &Factors, double* Tensor_pointer, 
                                   const Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > &True_Tensor, const int p)
    {   
        int AO_iter = 1;
        int print_iter = 1;
        int mttkrp_counter = 0;
        int tns_order = Factors.size();                          //TNS_ORDER
        int current_mode;
        const unsigned int threads_num = get_num_threads();

        // Timers
        nanoseconds stop_t_cpdgen = 0ns;
        nanoseconds stop_t_MTTKRP = 0ns;
        nanoseconds stop_t_Ts = 0ns;
        nanoseconds stop_t_KRs = 0ns;
        nanoseconds stop_t_fval = 0ns;
        nanoseconds stop_t_cal_grad = 0ns;
        nanoseconds stop_t_struct = 0ns;
        nanoseconds stop_t_NAG = 0ns;
        auto t1_struct  = high_resolution_clock::now();
        auto t1_Ts = high_resolution_clock::now();
        auto t1_KRs = high_resolution_clock::now();
        auto t1_NAG = high_resolution_clock::now();
        auto t1_cal_grad = high_resolution_clock::now();
        auto t1_cpdgen = high_resolution_clock::now();
        auto t1_fval = high_resolution_clock::now();
        auto t1_struct_d  = high_resolution_clock::now();

        Eigen::Tensor< double, static_cast<int>(TNS_ORDER) >  Est_Tensor_from_factors;                // with no dims, to calculate cost fun
        std::array<MatrixXd, TNS_ORDER> Factors_prev  = Factors;                                      //Previous values of factors
        std::array<MatrixXd, TNS_ORDER> Y_Factors     = Factors;                                      //Factors Y

        #if USE_COST_FUN
            Eigen::Tensor< double, 2>::Dimensions two_dim (tns_dims(0), tns_dims.prod()/tns_dims(0));
            Eigen::Tensor< double, 2 > Mat_Tensor = True_Tensor.reshape(two_dim);                         // Does not modify true tensor
            MatrixXd X_mat_0(tns_dims(0), tns_dims.prod()/tns_dims(0));
            X_mat_0 = Eigen::Map<Eigen::MatrixXd>(Mat_Tensor.data(), tns_dims(0), tns_dims.prod()/tns_dims(0));
            MatrixXd MTTKRP_0(tns_dims(0), R);
            MatrixXd gram_cwise_prod(R, R);
        #endif
        //-------------------------------------- Matrix Initializations ---------------------------------------------
        
        // Eigen::Tensor< double, 0 > frob_X_sq = frob_X.square();
        double frob_X_sq_d = frob_X(0);
        //--------------------------> Begin Algorithm <-----------------------------------------------
        cout << "--------------------------- BEGIN ALGORITHM --------------------------- " << endl;
        cout << AO_iter << "   " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X <<  endl;
        // Begin Timer for the While
	    auto t1 = high_resolution_clock::now();
        omp_set_nested(0);
        #if USE_COST_FUN
                    #pragma omp parallel \
                            num_threads(threads_num) \
                            proc_bind(spread)\
                            default(none)\
                            private(current_mode, Est_Tensor_from_factors)\
                            shared(Factors_prev, Y_Factors, cout, X_mat_0, MTTKRP_0, gram_cwise_prod,\
                            t1_struct, t1_Ts, t1_KRs, t1_cal_grad, t1_cpdgen, t1_struct_d, frob_X_sq_d,\
                            t1_NAG, t1_fval, stop_t_cal_grad, stop_t_cpdgen, stop_t_fval, stop_t_KRs, stop_t_MTTKRP,\
                            stop_t_NAG, stop_t_struct, stop_t_Ts, Factors, Tensor_pointer, tns_order,\
                            tns_dims, block_size, AO_iter, print_iter, True_Tensor, f_value, mttkrp_counter)
        #else
                    #pragma omp parallel\
                        num_threads(threads_num) \
                        proc_bind(spread)\
                        default(none)\
                        private(current_mode, Est_Tensor_from_factors)\
                        shared(Factors_prev, Y_Factors, cout,\
                        t1_struct, t1_Ts, t1_KRs, t1_cal_grad, t1_cpdgen, t1_struct_d, frob_X_sq_d,\
                        t1_NAG, t1_fval, stop_t_cal_grad, stop_t_cpdgen, stop_t_fval, stop_t_KRs, stop_t_MTTKRP,\
                        stop_t_NAG, stop_t_struct, stop_t_Ts, Factors, Tensor_pointer, tns_order,\
                        tns_dims, block_size, AO_iter, print_iter, True_Tensor, f_value, mttkrp_counter)
        #endif                    
        {
            std::array<MatrixXd, TNS_ORDER> local_Factors       = Factors;
            std::array<MatrixXd, TNS_ORDER> local_Factors_prev  = Factors;                                       //Previous values of factors
            std::array<MatrixXd, TNS_ORDER> local_Y_Factors     = Factors;
            double L, beta_accel, lambda;							// NAG parameters
            MatrixXd local_Hessian(R,R);

            double* local_Tensor_pointer = Tensor_pointer;
            double tmp_th = threads_num;
            double ratio = double(1.0/threads_num);
            while(1)
            {   
                //Select the current mode
                #pragma omp master
                sorted::Sample_mode(tns_order, current_mode);
                #pragma omp barrier
                #pragma omp flush(current_mode, AO_iter)
                //struct for current mode. contains the matrices for each mode
                #pragma omp master
                t1_struct  = high_resolution_clock::now();

                // std::cout << "Current mode: " << current_mode << std::endl;
                sorted::struct_mode current_mode_struct;
                current_mode_struct.struct_mode_init(block_size(current_mode), tns_order,  tns_dims(current_mode), R);
                #pragma omp master
                {
                    auto t2_struct = high_resolution_clock::now();
                    stop_t_struct += duration_cast<nanoseconds>(t2_struct - t1_struct);
                }


                //Sample the fibers and take the sampled matricization and the idxs used for the sampling of khatri-rao
                #pragma omp master
                t1_Ts = high_resolution_clock::now();

                sorted::Sample_fibers<TNS_ORDER>(Tensor_pointer,  tns_dims,  block_size,  current_mode,
                                current_mode_struct.idxs, current_mode_struct.T_s);
                // std::cout << "T_s norm is : " << current_mode_struct.T_s.norm() << std::endl; 
                #pragma omp master
                {
                    auto t2_Ts = high_resolution_clock::now();
                    stop_t_Ts += duration_cast<nanoseconds>(t2_Ts - t1_Ts);
                }                 


                //Compute the sampled Khatri Rao
                #pragma omp master
                t1_KRs = high_resolution_clock::now();

                symmetric::Sample_KhatriRao( current_mode, R, current_mode_struct.idxs, local_Factors_prev, current_mode_struct.KR_s);
                // std::cout << "KR_s norm is : " << current_mode_struct.KR_s.norm() << std::endl; 

                #pragma omp master
                {
                    auto t2_KRs = high_resolution_clock::now();
                    stop_t_KRs += duration_cast<nanoseconds>(t2_KRs - t1_KRs);
                }

                //Compute Hessian
                local_Hessian.noalias() = current_mode_struct.KR_s.transpose()*current_mode_struct.KR_s;

                //Compute Nesterov Parameters
                #pragma omp master
                t1_NAG = high_resolution_clock::now();

                Compute_NAG_parameters(local_Hessian, L, beta_accel, lambda);
                #pragma omp master
                {
                    auto t2_NAG = high_resolution_clock::now();
                    stop_t_NAG += duration_cast<nanoseconds>(t2_NAG - t1_NAG);
                }


                //Calculate local Gradient
                #pragma omp master
                t1_cal_grad = high_resolution_clock::now();

                parallel_with_p::Calc_gradient( tns_dims, current_mode, threads_num, lambda, local_Factors_prev[current_mode], local_Y_Factors[current_mode], local_Hessian, current_mode_struct.KR_s, current_mode_struct.T_s, current_mode_struct.Grad, stop_t_MTTKRP);
                #pragma omp master
                {
                    auto t2_cal_grad = high_resolution_clock::now();
                    stop_t_cal_grad += duration_cast<nanoseconds>(t2_cal_grad - t1_cal_grad);
                }


                //Update local factor
                local_Factors[current_mode]   = local_Y_Factors[current_mode] - current_mode_struct.Grad /(L + lambda);
                local_Factors[current_mode]   = local_Factors[current_mode].cwiseMax(current_mode_struct.Zero_Matrix);

                //Update local Y
                local_Y_Factors[current_mode] = local_Factors[current_mode] + beta_accel*(local_Factors[current_mode] - local_Factors_prev[current_mode]);

                local_Factors_prev[current_mode] = local_Factors[current_mode];
                if(AO_iter % p == 0)
                {   
                    #pragma omp barrier
                    #pragma omp master
                    {
                        for(int mode = 0; mode < TNS_ORDER; mode++)
                        {
                            Factors[mode].setZero();
                            Y_Factors[mode].setZero();
                            Factors_prev[mode].setZero();
                        }
                        
                    }
                    #pragma omp barrier
                    #pragma omp flush(Factors, Y_Factors, Factors_prev)
                    #pragma omp critical
                    {
                        for (int mode = 0; mode < TNS_ORDER; mode++)
                        {
                            Factors[mode] += ratio*local_Factors[mode];
                            Y_Factors[mode] += ratio*local_Y_Factors[mode];
                            Factors_prev[mode] += ratio*local_Factors_prev[mode];
                        }
                        
                    }
                    
                    #pragma omp barrier
                    #pragma omp flush(Factors, Y_Factors, Factors_prev)
                    for (int mode = 0; mode < TNS_ORDER; mode++)
                    {
                        local_Factors[mode] = Factors[mode];
                        local_Y_Factors[mode] = Factors[mode]; 
                        local_Factors_prev[mode] = Factors[mode];
                    }

                    #pragma omp barrier

                }
                // #pragma omp flush(Factors, Y_Factors, Factors_prev)               
                #pragma omp master
                {   
                    
                    if( int(AO_iter % ((TNS_ORDER + 1)*(((tns_dims.prod()/tns_dims(current_mode))/(threads_num * block_size(current_mode))))) == 0))
                    {    
                        #pragma omp flush(Factors, Y_Factors, Factors_prev)               
                        print_iter++;
                        #if USE_COST_FUN
                            auto t1_fval = high_resolution_clock::now();
                            partial::mttpartialkrp<TNS_ORDER>(tns_order, tns_dims, R, 0, Factors, X_mat_0, MTTKRP_0, threads_num);
                            gram_cwise_prod.setOnes();
                            compute_gram_cwise_prod(Factors, gram_cwise_prod);
                            compute_fval(frob_X_sq_d, MTTKRP_0, gram_cwise_prod, Factors[0], f_value(0));
                            f_value(0) = f_value(0) / frob_X_sq_d;
                            auto t2_fval = high_resolution_clock::now();
                            stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                            cout << print_iter << "  " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                        #else
                            // Here we calculate the measure of performance, either CPD_GEN or calculate the norm of a tensor using the factors
                            t1_cpdgen = high_resolution_clock::now();
                            // for(int i = 0; i < TNS_ORDER; i++)
                            // {
                            //     std:: cout << Factors_prev[i].norm() << std:: endl;
                                
                            // }
                            CpdGen( tns_dims, Factors_prev, R, Est_Tensor_from_factors);
                            auto t2_cpdgen = high_resolution_clock::now();
                            stop_t_cpdgen += duration_cast<nanoseconds>(t2_cpdgen-t1_cpdgen);

                            // f_value computation
                            t1_fval = high_resolution_clock::now();
                            f_value = (True_Tensor - Est_Tensor_from_factors).square().sum();  
                            auto t2_fval = high_resolution_clock::now();
                            stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                            cout << print_iter << "  " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X << endl;
                        #endif

                        mttkrp_counter += (TNS_ORDER + 1); // Increase counter according to printing (after TNS_ORDER + 1 (Accel. step)).
                    }
                }

                #pragma omp barrier
                if(mttkrp_counter >= MAX_MTTKRP)
                {
                    break; 
                }

                
                #pragma omp master
                {
                    AO_iter++;
                    //delete current_mode_struct;
                    t1_struct_d  = high_resolution_clock::now();
                }
                current_mode_struct.destruct_struct_mode();
                #pragma omp master
                {
                    auto t2_struct_d  = high_resolution_clock::now();
                    stop_t_struct += duration_cast<nanoseconds>(t2_struct_d - t1_struct_d);
                }



            }
        } // End of parallel region
        cout << "---------------------------- EXIT ALGORITHM --------------------------- " << endl;
        auto t2 = high_resolution_clock::now();
        // End Timer for the While

        duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
        // seconds stop_t = duration_cast<std::chrono::seconds>(t2 - t1);
        cout << "CPU time WHILE = " << stop_t.count() << "s" <<endl; 
        cout << "Time CPDGEN    = " << stop_t_cpdgen.count() * (1e-9) << "s" << endl; 
        cout << "Time MTTKRP    = " << stop_t_MTTKRP.count() * (1e-9) << "s" << endl;
        cout << "Time sample T  = " << stop_t_Ts.count() * (1e-9) << "s" << endl;
        cout << "Time sample KR = " << stop_t_KRs.count() * (1e-9) << "s" << endl;
        cout << "Time f_value   = " << stop_t_fval.count() * (1e-9) << "s" << endl;
        cout << "Time grad      = " << (stop_t_cal_grad.count() - stop_t_MTTKRP.count()) * (1e-9) << "s" << endl;
        cout << "Time NAG par   = " << stop_t_NAG.count() * (1e-9) << "s" << endl;
        cout << "Time struct    = " << stop_t_struct.count() * (1e-9) << "s" << endl;
        cout << "AO_iter        = " << AO_iter << endl;
        cout << "num of threads = " << threads_num << endl << endl;
        
    }

    
}

namespace parallel_asychronous
{   
    
    struct struct_mode
    {   

        MatrixXi idxs;
        MatrixXd T_s;
        MatrixXd KR_s; 
        MatrixXd Grad;
        MatrixXd Zero_Matrix;	 

        void struct_mode_init(int m, int n, int k, int r)     //m bs(mode), n order, k dim, r rank
        {
             idxs = MatrixXi::Zero(m, n); 
             T_s = MatrixXd::Zero(k, m);
             KR_s = MatrixXd::Zero(m, r);     
             Grad = MatrixXd::Zero(k, r);
             Zero_Matrix = MatrixXd::Zero(k, r);
        }

        void destruct_struct_mode()
        {
             idxs.resize(0,0); 
             T_s.resize(0,0);
             KR_s.resize(0,0);  
             Grad.resize(0,0);
             Zero_Matrix.resize(0,0);

        }

    };


    template <std::size_t  TNS_ORDER>
    inline void solve_BrasCPaccel( const double AO_tol, const double MAX_MTTKRP, const int R, const Eigen::Tensor<double, 0> frob_X, Eigen::Tensor<double, 0> &f_value, const VectorXi &tns_dims,
                                   const VectorXi &block_size, std::array<MatrixXd, TNS_ORDER> &Factors, double* Tensor_pointer, 
                                   const Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > &True_Tensor, const int p)
    {   
        int AO_iter = 1;
        int print_iter = 1;
        int mttkrp_counter = 0;
        int tns_order = Factors.size();                          //TNS_ORDER
        int current_mode;
        const unsigned int threads_num = get_num_threads();

        // Timers
        nanoseconds stop_t_cpdgen = 0ns;
        nanoseconds stop_t_MTTKRP = 0ns;
        nanoseconds stop_t_Ts = 0ns;
        nanoseconds stop_t_KRs = 0ns;
        nanoseconds stop_t_fval = 0ns;
        nanoseconds stop_t_cal_grad = 0ns;
        nanoseconds stop_t_struct = 0ns;
        nanoseconds stop_t_NAG = 0ns;
        auto t1_struct  = high_resolution_clock::now();
        auto t1_Ts = high_resolution_clock::now();
        auto t1_KRs = high_resolution_clock::now();
        auto t1_NAG = high_resolution_clock::now();
        auto t1_cal_grad = high_resolution_clock::now();
        auto t1_cpdgen = high_resolution_clock::now();
        auto t1_fval = high_resolution_clock::now();
        auto t1_struct_d  = high_resolution_clock::now();


        Eigen::Tensor< double, static_cast<int>(TNS_ORDER) >  Est_Tensor_from_factors;                // with no dims, to calculate cost fun
        std::array<MatrixXd, TNS_ORDER> Factors_prev  = Factors;                                      //Previous values of factors
        std::array<MatrixXd, TNS_ORDER> Y_Factors     = Factors;                                      //Factors Y

        #if USE_COST_FUN
            Eigen::Tensor< double, 2>::Dimensions two_dim (tns_dims(0), tns_dims.prod()/tns_dims(0));
            Eigen::Tensor< double, 2 > Mat_Tensor = True_Tensor.reshape(two_dim);                         // Does not modify true tensor
            MatrixXd X_mat_0(tns_dims(0), tns_dims.prod()/tns_dims(0));
            X_mat_0 = Eigen::Map<Eigen::MatrixXd>(Mat_Tensor.data(), tns_dims(0), tns_dims.prod()/tns_dims(0));
            MatrixXd MTTKRP_0(tns_dims(0), R);
            MatrixXd gram_cwise_prod(R, R);
        #endif
        //-------------------------------------- Matrix Initializations ---------------------------------------------
        
        // Eigen::Tensor< double, 0 > frob_X_sq = frob_X.square();
        double frob_X_sq_d = frob_X(0);
        //--------------------------> Begin Algorithm <-----------------------------------------------
        cout << "--------------------------- BEGIN ALGORITHM --------------------------- " << endl;
        cout << AO_iter << "   " << " --- " << f_value/frob_X << "   " << " --- " << f_value << "   " << " --- " << frob_X <<  endl;
        // Begin Timer for the While
	    auto t1 = high_resolution_clock::now();
        omp_set_nested(1);

        #if USE_COST_FUN
        #pragma omp parallel \
                        num_threads(threads_num) \
                        proc_bind(spread)\
                        default(none)\
                        private(current_mode, Est_Tensor_from_factors)\
                        shared(Factors_prev, Y_Factors, cout, X_mat_0, MTTKRP_0, gram_cwise_prod,\
                        t1_struct, t1_Ts, t1_KRs, t1_cal_grad, t1_cpdgen, t1_struct_d, frob_X_sq_d,\
                        t1_NAG, t1_fval, stop_t_cal_grad, stop_t_cpdgen, stop_t_fval, stop_t_KRs, stop_t_MTTKRP,\
                        stop_t_NAG, stop_t_struct, stop_t_Ts, Factors, Tensor_pointer, tns_order,\
                        tns_dims, block_size, AO_iter, print_iter, True_Tensor, f_value, mttkrp_counter)
        #else
        #pragma omp parallel\
                        num_threads(threads_num) \
                        proc_bind(spread)\
                        default(none)\
                        private(current_mode, Est_Tensor_from_factors)\
                        shared(Factors_prev, Y_Factors, cout,\
                        t1_struct, t1_Ts, t1_KRs, t1_cal_grad, t1_cpdgen, t1_struct_d, frob_X_sq_d,\
                        t1_NAG, t1_fval, stop_t_cal_grad, stop_t_cpdgen, stop_t_fval, stop_t_KRs, stop_t_MTTKRP,\
                        stop_t_NAG, stop_t_struct, stop_t_Ts, Factors, Tensor_pointer, tns_order,\
                        tns_dims, block_size, AO_iter, print_iter, True_Tensor, f_value, mttkrp_counter)
        #endif                
        {
            std::array<MatrixXd, TNS_ORDER> local_Factors       = Factors;
            std::array<MatrixXd, TNS_ORDER> local_Factors_prev  = Factors;                                       //Previous values of factors
            std::array<MatrixXd, TNS_ORDER> local_Y_Factors     = Factors;
            double L, beta_accel, lambda;							// NAG parameters
            MatrixXd local_Hessian(R,R);

            double* local_Tensor_pointer = Tensor_pointer;
            double tmp_th = threads_num;
            double ratio = double(1.0/threads_num);
            
            while(1)
            {   
                //Select the current mode
                // #pragma omp master
                // sorted::Sample_mode(tns_order, current_mode);
                // #pragma omp barrier
                // #pragma omp flush(current_mode, AO_iter)    
                current_mode = AO_iter % TNS_ORDER; // In a cyclical manner

                //struct for current mode. contains the matrices for each mode
                #pragma omp master
                t1_struct  = high_resolution_clock::now();

                // std::cout << "Current mode: " << current_mode << std::endl;
                sorted::struct_mode current_mode_struct;
                current_mode_struct.struct_mode_init(block_size(current_mode), tns_order,  tns_dims(current_mode), R);
                #pragma omp master
                {
                    auto t2_struct = high_resolution_clock::now();
                    stop_t_struct += duration_cast<nanoseconds>(t2_struct - t1_struct);
                }


                //Sample the fibers and take the sampled matricization and the idxs used for the sampling of khatri-rao
                #pragma omp master
                t1_Ts = high_resolution_clock::now();

                sorted::Sample_fibers<TNS_ORDER>(local_Tensor_pointer,  tns_dims,  block_size,  current_mode,
                                current_mode_struct.idxs, current_mode_struct.T_s);
                // std::cout << "T_s norm is : " << current_mode_struct.T_s.norm() << std::endl; 
                #pragma omp master
                {
                    auto t2_Ts = high_resolution_clock::now();
                    stop_t_Ts += duration_cast<nanoseconds>(t2_Ts - t1_Ts);
                }                 


                //Compute the sampled Khatri Rao
                #pragma omp master
                t1_KRs = high_resolution_clock::now();

                symmetric::Sample_KhatriRao( current_mode, R, current_mode_struct.idxs, local_Factors_prev, current_mode_struct.KR_s);
                // std::cout << "KR_s norm is : " << current_mode_struct.KR_s.norm() << std::endl; 

                #pragma omp master
                {
                    auto t2_KRs = high_resolution_clock::now();
                    stop_t_KRs += duration_cast<nanoseconds>(t2_KRs - t1_KRs);
                }

                //Compute Hessian
                local_Hessian.noalias() = current_mode_struct.KR_s.transpose()*current_mode_struct.KR_s;

                //Compute Nesterov Parameters
                #pragma omp master
                t1_NAG = high_resolution_clock::now();

                Compute_NAG_parameters(local_Hessian, L, beta_accel, lambda);
                #pragma omp master
                {
                    auto t2_NAG = high_resolution_clock::now();
                    stop_t_NAG += duration_cast<nanoseconds>(t2_NAG - t1_NAG);
                }


                //Calculate local Gradient
                #pragma omp master
                t1_cal_grad = high_resolution_clock::now();

                parallel_with_p::Calc_gradient( tns_dims, current_mode, threads_num, lambda, local_Factors_prev[current_mode], local_Y_Factors[current_mode], local_Hessian, current_mode_struct.KR_s, current_mode_struct.T_s, current_mode_struct.Grad, stop_t_MTTKRP);
                #pragma omp master
                {
                    auto t2_cal_grad = high_resolution_clock::now();
                    stop_t_cal_grad += duration_cast<nanoseconds>(t2_cal_grad - t1_cal_grad);
                }


                //Update local factor
                local_Factors[current_mode]   = local_Y_Factors[current_mode] - current_mode_struct.Grad /(L + lambda);
                local_Factors[current_mode]   = local_Factors[current_mode].cwiseMax(current_mode_struct.Zero_Matrix);

                //Update local Y
                local_Y_Factors[current_mode] = local_Factors[current_mode] + beta_accel*(local_Factors[current_mode] - local_Factors_prev[current_mode]);

                local_Factors_prev[current_mode] = local_Factors[current_mode];

                if(AO_iter % p == 0)
                {   
                    #pragma omp barrier
                    #pragma omp master
                    {
                        for(int mode = 0; mode < TNS_ORDER; mode++)
                        {
                            Factors[mode].setZero();
                            Y_Factors[mode].setZero();
                            Factors_prev[mode].setZero();
                        }
                        
                    }
                    #pragma omp barrier
                    #pragma omp flush(Factors, Y_Factors, Factors_prev)
                    #pragma omp critical
                    {
                        for (int mode = 0; mode < TNS_ORDER; mode++)
                        {
                            Factors[mode] += ratio*local_Factors[mode];
                            Y_Factors[mode] += ratio*local_Y_Factors[mode];
                            Factors_prev[mode] += ratio*local_Factors_prev[mode];
                        }
                        
                    }
                    
                    #pragma omp barrier
                    #pragma omp flush(Factors, Y_Factors, Factors_prev)
                    for (int mode = 0; mode < TNS_ORDER; mode++)
                    {
                        local_Factors[mode] = Factors[mode];
                        local_Y_Factors[mode] = Factors[mode]; // maybe factors
                        local_Factors_prev[mode] = Factors[mode];
                    }

                    #pragma omp barrier

                }
                // #pragma omp flush(Factors, Y_Factors, Factors_prev)     
                if( int(AO_iter % ((TNS_ORDER + 1)*(((tns_dims.prod()/tns_dims(current_mode))/(threads_num * block_size(current_mode))))) == 0))
                {   
                    #pragma omp master
                    {    
                        #pragma omp flush(Factors, Y_Factors, Factors_prev)               
                        print_iter++;
                        #if USE_COST_FUN
                            auto t1_fval = high_resolution_clock::now();
                            partial::mttpartialkrp<TNS_ORDER>(tns_order, tns_dims, R, 0, Factors, X_mat_0, MTTKRP_0, threads_num);
                            gram_cwise_prod.setOnes();
                            compute_gram_cwise_prod(Factors, gram_cwise_prod);
                            compute_fval(frob_X_sq_d, MTTKRP_0, gram_cwise_prod, Factors[0], f_value(0));
                            f_value(0) = f_value(0) / frob_X_sq_d;
                            auto t2_fval = high_resolution_clock::now();
                            stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                            cout << print_iter<< " " << " --- " << f_value << " " << " --- " << frob_X << endl;
                        #else
                            // Here we calculate the measure of performance, either CPD_GEN or calculate the norm of a tensor using the factors
                            t1_cpdgen = high_resolution_clock::now();
                            // for(int i = 0; i < TNS_ORDER; i++)
                            // {
                            //     std:: cout << Factors_prev[i].norm() << std:: endl;
                                
                            // }
                            CpdGen( tns_dims, Factors_prev, R, Est_Tensor_from_factors);
                            auto t2_cpdgen = high_resolution_clock::now();
                            stop_t_cpdgen += duration_cast<nanoseconds>(t2_cpdgen-t1_cpdgen);

                            // f_value computation
                            t1_fval = high_resolution_clock::now();
                            f_value = (True_Tensor - Est_Tensor_from_factors).square().sum();  
                            auto t2_fval = high_resolution_clock::now();
                            stop_t_fval += duration_cast<nanoseconds>(t2_fval - t1_fval);
                            cout << print_iter<< " " << " --- " << f_value/frob_X << " " << " --- " << f_value << " " << " --- " << frob_X << endl;
                        #endif

                        mttkrp_counter += (TNS_ORDER + 1); // Increase counter according to printing (after TNS_ORDER + 1 (Accel. step)).
                        
                    }
                    #pragma omp barrier
                    if(mttkrp_counter >= MAX_MTTKRP)
                    {
                        break; 
                    }

                }

         
                #pragma omp master
                {
                    AO_iter++;
                    //delete current_mode_struct;
                    t1_struct_d  = high_resolution_clock::now();
                }
                current_mode_struct.destruct_struct_mode();
                #pragma omp master
                {
                    auto t2_struct_d  = high_resolution_clock::now();
                    stop_t_struct += duration_cast<nanoseconds>(t2_struct_d - t1_struct_d);
                }



            }
        } // End of parallel region
        cout << "---------------------------- EXIT ALGORITHM --------------------------- " << endl;
        auto t2 = high_resolution_clock::now();
        // End Timer for the While

        duration<double> stop_t = duration_cast<duration<double>>(t2-t1);
        // seconds stop_t = duration_cast<std::chrono::seconds>(t2 - t1);
        cout << "CPU time WHILE = " << stop_t.count() << "s" <<endl; 
        cout << "Time CPDGEN    = " << stop_t_cpdgen.count() * (1e-9) << "s" << endl; 
        cout << "Time MTTKRP    = " << stop_t_MTTKRP.count() * (1e-9) << "s" << endl;
        cout << "Time sample T  = " << stop_t_Ts.count() * (1e-9) << "s" << endl;
        cout << "Time sample KR = " << stop_t_KRs.count() * (1e-9) << "s" << endl;
        cout << "Time f_value   = " << stop_t_fval.count() * (1e-9) << "s" << endl;
        cout << "Time grad      = " << (stop_t_cal_grad.count() - stop_t_MTTKRP.count()) * (1e-9) << "s" << endl;
        cout << "Time NAG par   = " << stop_t_NAG.count() * (1e-9) << "s" << endl;
        cout << "Time struct    = " << stop_t_struct.count() * (1e-9) << "s" << endl;
        cout << "AO_iter        = " << AO_iter << endl;
        cout << "num of threads = " << threads_num << endl << endl;
        
    }

    
}
#endif //end if