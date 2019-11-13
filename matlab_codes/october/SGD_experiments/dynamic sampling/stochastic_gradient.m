function [ f_val_sd,fval_merged, x_sd_aver, norm_sd ] = stochastic_gradient( A, b, eta_sd_init, x_sd, f_val_sd, Options )
    
    n = size(A,1);

    p = inputParser;
    p.KeepUnmatched = true;
    p.addOptional('epsilon', 10^(-8));
    p.addOptional('MaxIter', 20000);
    p.addOptional('StepSize', 'constant');
    p.addOptional('Function','smooth');
    p.addOptional('mu', 1);
    p.addOptional('gamma', 10);
    p.addOptional('mini_batch','false');
    p.addOptional('batch_size', 1);
    p.addOptional('accelerate','false');
    p.addOptional('averaging', 'false');
    p.addOptional('maximal_res',1)
    p.addOptional('f_star',0.01);
    p.addOptional('adaptive_sample','false');
    p.parse(Options);
    options = p.Results

    iter = 1;
    if(strcmp(options.Function,'strongly-convex'))
        strong_conv_on = 1;
        q = cond(A'*A);
        b_par = (1-sqrt(1/q))/(1+sqrt(1/q));
    else
        strong_conv_on = 0;
        q = 0;
    end
    
    
    fval_merged = [f_val_sd(:,iter)];
    x_sd(:,iter) = x_sd;
    y_sd(:,iter) = x_sd;
    f_val_sd(iter) = f_val_sd;
    norm_sd(iter) =  norm(x_sd(:,iter));
    a_par(iter) = 1;
    variance = [];
    sq_grad_full = [];
    x_sd_aver = [];
    stoch_grad_cov = [];
    
    if(strcmp(options.mini_batch,'true'))
        batch_s = options.batch_size;
    else
        batch_s = 1;
    end
    
    chi = 0.5; %in (0,1)
    tau = 1.1;
    
    while(1)

        batch = randperm(n,batch_s);
        if(strcmp(options.accelerate,'true'))
            stoch_grad = (1/batch_s).*(A(batch,:)'*A(batch,:)*y_sd(:,iter) - A(batch,:)'*b(batch));
        else
            stoch_grad = (1/batch_s).*(A(batch,:)'*A(batch,:)*x_sd(:,iter) - A(batch,:)'*b(batch));
            
            %||g(x_k,ksi_k)||_2^2
            stoch_grad_sq_norm = norm(stoch_grad)^2;
            %create an approximation for the variance
            for ii=1:batch_s
                stoch_grad_cov = [stoch_grad_cov (A(batch(ii),:)'*A(batch(ii),:)*x_sd(:,iter) - A(batch(ii),:)'*b(batch(ii)))];
            end
            appr_var = trace(cov(stoch_grad_cov));
        end
        
        if(strcmp(options.StepSize,'variant'))
            if(strong_conv_on == 0)  %only smoothness
                eta_sd = (1/sqrt(iter))*eta_sd_init;
            else                     %strongly convex
                eta_sd = (1/iter)*eta_sd_init;
            end
        elseif(strcmp(options.StepSize,'variant-method2'))
            if iter > 0.5*options.MaxIter
                eta_sd = 2/(options.mu*(options.gamma + iter));
            else
                eta_sd = eta_sd_init;
            end
        elseif(strcmp(options.StepSize,'constant'))
            eta_sd = eta_sd_init;
        end
        
        if(strcmp(options.accelerate,'true'))
            %accelerate
            x_sd(:,iter+1) = y_sd(:,iter) - eta_sd*stoch_grad;
            if(strcmp(options.Function, 'smooth'))
                a_par(iter+1) = update_alpha(a_par(iter), q);
                b_par(iter+1) = (a_par(iter)*(1 - a_par(iter)))/(a_par(iter)^2 + a_par(iter+1));
                y_sd(:,iter+1) = x_sd(:,iter+1) + b_par(iter+1)*(x_sd(:,iter+1) - x_sd(:,iter));
            else
                y_sd(:,iter+1) = x_sd(:,iter+1) + b_par*(x_sd(:,iter+1) - x_sd(:,iter));
            end
        else
            x_sd(:,iter+1) = x_sd(:,iter) - eta_sd*stoch_grad;
        end
        
        if strcmp(options.averaging, 'true')
            x_sd_aver(:,iter+1) = (1/(iter + 1))*(sum(x_sd,2));
        end
        
        norm_sd(iter+1) =  norm(x_sd(:,iter+1));
        
        f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;
        
        %for mini-batch plot
        fval_merged = [fval_merged f_val_sd(:,iter+1)*ones(1,batch_s)];
        
        crit_sg = norm(x_sd(:,iter+1)-x_sd(:,iter))/norm(x_sd(:,iter));
        
        %adaptive sample size
        if(strcmp(options.adaptive_sample,'true'))   
            if(appr_var/batch_s >= (chi^2)*stoch_grad_sq_norm) && (batch_s < 0.7*n)
                batch_s = floor(tau*batch_s);
            end    
        end
        
        if(crit_sg < options.epsilon || iter > options.MaxIter) 
            break;
        end
        
        iter = iter + 1;
        stoch_grad_cov = [];
    end
   

end
