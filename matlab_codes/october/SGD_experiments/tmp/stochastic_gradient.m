function [ f_val_sd,fval_merged, norm_sd ] = stochastic_gradient( A, b, eta_sd_init, x_sd, f_val_sd, Options )
    
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
    p.addOptional('sampling_rate',0);
    p.addOptional('maximal_res',1)
    p.addOptional('exp_chunk',1);
    p.addOptional('f_star',0.01);
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
    
    if(strcmp(options.mini_batch,'true'))
        batch_s = options.batch_size;
    else
        batch_s = 1;
    end
    
    while(1)

        batch = randperm(n,batch_s);
        if(strcmp(options.accelerate,'true'))
            stoch_grad = (1/batch_s).*(A(batch,:)'*A(batch,:)*y_sd(:,iter) - A(batch,:)'*b(batch));
        else
            stoch_grad = (1/batch_s).*(A(batch,:)'*A(batch,:)*x_sd(:,iter) - A(batch,:)'*b(batch));
        end
        if(options.sampling_rate ~= 0)
            if mod(iter, options.sampling_rate) == 0
                exp_chunk_idx = randperm(n,options.exp_chunk);
                [expected_value_g_squared,~] = compute_squared_mean_g(exp_chunk_idx, x_sd(:,iter), A, b);
                sq_grad_full = [sq_grad_full (1/(n^2))*norm(A'*(A*x_sd(:,iter) - b))^2];
                variance = [variance (expected_value_g_squared - sq_grad_full(end))];
            end
            
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
            x_sd(:,iter+1) = (1/(iter + 1))*(sum(x_sd,2));
        end
        norm_sd(iter+1) =  norm(x_sd(:,iter+1));
        
        f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;
        
        %for mini-batch plot
        fval_merged = [fval_merged f_val_sd(:,iter+1)*ones(1,batch_s)];
        
        crit_sg = norm(x_sd(:,iter+1)-x_sd(:,iter))/norm(x_sd(:,iter));

        if(crit_sg < options.epsilon || iter > options.MaxIter) 
            break;
        end

        iter = iter + 1;
    end
    
    M = 3%options.f_star*norm(A(n/2,:))^2;   %for big dimensionallity of x norms tend to be equal
    M_v = 3*n-1%n/2 -1;
    if(options.sampling_rate ~= 0)
        figure(1)
        semilogy(variance);
        hold on;
        semilogy(M + M_v*sq_grad_full);
        hold off;
        legend('variance','bound');
        grid on;
    end
end

