function [A_est, MSE, error] = BrasCPD_vol3(T,options)
    
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
    
    %Check if acceleration mode is on 
    accel_var = 0;
    cyclical_vec = [1 2 3];
    lambda_prox = 0;
    %% Calculate required quantities  
    order = size(A_true,2);
    error_init = frob(T - cpdgen(A_init))/frob(T);                         % initial error
    iter_per_epoch = floor(dims(1)*dims(2)/block_size(1));                 % epochs
    
    A_est = A_init;
    A_next = A_init;
    error = error_init;
    MSE = (1/3)*(MSE_measure(A_est{1},A_true{1})+MSE_measure(A_est{2},A_true{2})+ ...
        MSE_measure(A_est{3},A_true{3}));
    
    if strcmp('on',options.acceleration)
         Y      = A_init;
         accel_var = 1;
    end

    
    for iter = 1:MAX_ITER 
        
        n = select_factor(cyclical,iter,order,cyclical_vec);

                                                                           % choose factor to update
        kr_idx = find([1:order] - n);                                      % factors for Khatri-Rao 
        
        [ ~, factor_idxs, T_s ] = sample_fbrs(T, n, dims, block_size );                        
        
        A_kr = A_est(kr_idx);
        H = sample_khatri_rao(A_kr, factor_idxs);
        
        %H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});                         % khatri rao product
        
        if accel_var == 1
            Hessian = H'*H;
            [L, beta_accel, lambda_prox] = NAG_parameters_prox(Hessian,options);
            G = Y{n}*(Hessian + lambda_prox*eye(size(Hessian,1)))-(T_s'*H + lambda_prox*A_est{n});
            A_next{n} = Y{n} - (1/(L + lambda_prox)).*(G);
            A_next{n} = proxr(A_next{n}, options, n);

            Y{n} = A_next{n} + beta_accel*(A_next{n} - A_est{n});
            

        else
            alpha = alpha0/(block_size*(iter)^(1e-6));        
            A_next{n} = A_est{n} - alpha*(A_est{ n }*H'*H-T_s'*H);
            A_next{n} = proxr(A_next{n}, options, n);
            
        end
        
        RFC = rel_measure(A_next, A_est);
        A_est{n} = A_next{n};
        
%         if RFC < tol_rel 
%             break;
%         end
        if(mod(iter,iter_per_epoch) == 0 && iter > 0)
            i = iter/iter_per_epoch;
            error(i+1) = frob(T - cpdgen(A_est))/frob(T);                      % error for accel scheme
            MSE(i + 1) = (1/3)*(MSE_measure(A_est{1},A_true{1})+MSE_measure(A_est{2},A_true{2})...
                       +MSE_measure(A_est{3},A_true{3}));
            disp(['BrasCPD accel at iteration ',num2str(i+1),' and the MSE is ',num2str(MSE(i+1))])
            disp(['at iteration ',num2str(i+1),' and the NRE is ',num2str(error(i+1))])
            disp('====')   
            if abs(error(i+1) )<= tol
                 break;
            end
        end
    end
end

