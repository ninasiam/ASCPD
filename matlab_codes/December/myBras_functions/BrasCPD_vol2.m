function [A_est, MSE, error] = BrasCPD_vol2(T,options)
    
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
    %% Calculate required quantities (matricizations) 

    T_mat{1} = tens2mat(T,1,[2 3])';
    T_mat{2} = tens2mat(T,2,[1 3])';
    T_mat{3} = tens2mat(T,3,[1 2])';

    
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
    
    if strcmp('on',options.proximal)
        lambda_prox = 10^(1);
    end
    
    for iter = 1:MAX_ITER 
        
        n = select_factor(cyclical,iter,order,cyclical_vec);

                                                                           % choose factor to update
        kr_idx = find([1:order] - n);                                      % factors for Khatri-Rao 
        J_n = dims(kr_idx(1))*dims(kr_idx(2));                             % khatri rao dimension
        idx = randperm(J_n,block_size);                                    % choose the sample 
        F_n = sort(idx);                                                   % sorted sample of row fibers

        T_s = T_mat{n};  
        T_s = T_s(F_n,:);
        
        H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});                         % khatri rao product
        
        if accel_var == 1
            Hessian = H(F_n,:)'*H(F_n,:);
            [L, beta_accel] = NAG_parameters(Hessian,lambda_prox);
            G = Y{n}*(Hessian + lambda_prox*eye(size(Hessian,1)))-(T_s'*H(F_n,:) + lambda_prox*A_est{n});
            A_next{n} = Y{n} - (1/(L + lambda_prox)).*(G);
            A_next{n} = proxr(A_next{n}, options, n);

            Y{n} = A_next{n} + beta_accel*(A_next{n} - A_est{n});
            

        else
            alpha = alpha0/(block_size*(iter)^(1e-6));        
            A_next{n} = A_est{n} - alpha*(A_est{ n }*H(F_n,:)'*H(F_n,:)-T_s'*H(F_n,:));
            A_next{n} = proxr(A_next{n}, options, n);
            
        end
        
        RFC = rel_measure(A_next, A_est);
        A_est{n} = A_next{n};
        
        if RFC < tol_rel 
            break;
        end
        if(mod(iter,iter_per_epoch) == 0 && iter > 0)
            i = iter/iter_per_epoch;
            error(i+1) = frob(T - cpdgen(A_est))/frob(T);                      % error for accel scheme
            MSE(i + 1) = (1/3)*(MSE_measure(A_est{1},A_true{1})+MSE_measure(A_est{2},A_true{2})...
                       +MSE_measure(A_est{3},A_true{3}));
                  
            if abs(error(i+1) )<= tol
                 break;
            end
        end
    end
end

