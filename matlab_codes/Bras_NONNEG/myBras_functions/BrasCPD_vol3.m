function [A_est, error, F_value] = BrasCPD_vol3(T,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BrasCPD accel Algorithm (CPD using randomized block coordiante 
% + accelerated stochastic gradient)
% --input
% T      : the data tensor
% options: algorithm parameters
%    'cyclical'            - Update in a cyclical manner
%    'dims'                - Tensor dimensions
%    'constraint'          - Latent factor constraints
%    'alpha0'              - Initial stepsize (for non accelerated case)
%    'block_size'          - Number of samples
%    'MAX_ITER'            - Maximum number of iterations
%    'A_init'              - Latent factor initializations
%    'A_true'              - Ground truth latent factors (for MSE computation only)
%    'tol'                 - stopping criterion (cost function)
%    'tol_rel'             - stopping criterion (relative factor change)
% --------------------------------------------------------------------------------
% --output
% A_est     : the estimated factors
% F_value   : the cost function at different iterations
% error     : error at different iterations
% --------------------------------------------------------------------------------
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%Initialize parameters
    A_true      = options.A_true;
    A_init      = options.A_init;
    MAX_ITER    = options.max_iter; 
    block_size  = options.bz;
    tol         = options.tol;
    tol_rel     = options.tol_rel;
    alpha0      = options.alpha0;
    dims        = options.dims;
    cyclical    = options.cyclical;
    display     = 1;
    %Acceleration variable 
    accel_var = 0;
    
    %Cyclical vector
    cyclical_vec = [1 2 3];
    
    %Lambda parameter
    lambda_prox = 0;
     
    %% Calculate required quantities  
    order = size(A_true,2);
    frob_X = frob(T);
    F_value = [frob(T - cpdgen(A_init))];                                  % initial f_value
    error_init = F_value(1)/frob_X;                                        % initial error
    iter_per_epoch = floor(dims(1)*dims(2)/block_size(1));                 % epochs
    
    A_est = A_init;
    A_next = A_init;
    error = error_init;
    
%     MSE = (1/3)*(MSE_measure(A_est{1},A_true{1})+MSE_measure(A_est{2},A_true{2})+ ...
%         MSE_measure(A_est{3},A_true{3}));
    
    %Check if acceleration mode is on 
    if strcmp('on',options.acceleration)
         Y = A_init;
         accel_var = 1;
    end

    
    for iter = 1:MAX_ITER 
        
        n = select_factor(cyclical,iter,order,cyclical_vec);               % select factor

                                                                           % choose factor to update
        kr_idx = find([1:order] - n);                                      % factors for Khatri-Rao 
        
        [ ~, factor_idxs, T_s ] = sample_fbrs(T, n, dims, block_size );    % sample fibers                 
        
        A_kr = A_est(kr_idx);
        
        H = sample_khatri_rao(A_kr, factor_idxs);                          % khatri-rao product
             
        if accel_var == 1
            
            Hessian = H'*H;
            [L, beta_accel, lambda_prox] = NAG_parameters_prox(Hessian,options);
            
            G = Y{n}*(Hessian + lambda_prox*eye(size(Hessian,1)))- ...
                (T_s'*H + lambda_prox*A_est{n});
           
            A_next{n} = Y{n} - (1/(L + lambda_prox)).*(G);
            A_next{n} = proxr(A_next{n}, options, n);                      % add constraint (proximal operator)

            Y{n} = A_next{n} + beta_accel*(A_next{n} - A_est{n});
            
        else
            
            alpha = alpha0/(block_size*(iter)^(1e-6));       
            A_next{n} = A_est{n} - alpha*(A_est{n}*(H'*H)-T_s'*H);
            A_next{n} = proxr(A_next{n}, options, n);
            
        end

        RFC = rel_measure(A_next, A_est);                                  % relative factor change
        A_est{n} = A_next{n};
        
%         if(mod(iter,3*iter_per_epoch) == 0 && iter > 0)
%             figure(2)
%             subplot(311)
%             stem(A_next{1} - A_est{1});
%             subplot(312)
%             stem(A_next{2} - A_est{2});
%             subplot(313)
%             stem(A_next{3} - A_est{3});
%             pause(0.1)
%         end
        
          % Terminating condition
%         if RFC < tol_rel 
%             iter
%             break;
%         end

        if(mod(iter,3*iter_per_epoch) == 0 && iter > 0)
            i = iter/(3*iter_per_epoch);
            f_val = frob(T - cpdgen(A_est));
                                                 
            F_value = [F_value f_val];
            error(i+1) = f_val/frob_X;                                     % error for accel scheme
%             MSE(i + 1) = (1/3)*(MSE_measure(A_est{1},A_true{1})+MSE_measure(A_est{2},A_true{2})...
%                        + MSE_measure(A_est{3},A_true{3}));      

              if(display == 1)
%               disp(['BrasCPD accel at iteration ',num2str(i+1),' and the MSE is ',num2str(MSE(i+1))])
%               disp(['at iteration ',num2str(i+1),' and the NRE is ',num2str(error(i+1))])
%               disp('====')   
              end
              
              % Terminating condition
%             if abs(error(i+1) )<= tol
%                  break;
%             end
        end
    end
end

