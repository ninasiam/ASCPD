function [x_ada, J_ada_end, x_list] = solve_AdaGrad(n, m, A, y, J_init, x_init, sigm_x, alpha, lambda, MAX_ITERS)
    
    fig5 = figure();
    epsilon = 10^(-4);
    J_ada = [J_init];
    J_ada_plot = [J_init];
    grad_accum = zeros(n,1);
    x_ada = x_init;
    iter = 1;
    eta = alpha;
    zeta = 10^(-10);
    x_sum = zeros(n,1);
    x_epoch = ones(n,1);
    x_list = [x_init];
    RFC = 1;
    
    while(1)

        % Stochastic Gradient Step
        i = randi(m, 1, 1);
        grad_J_x = A(i,:)'*(sigm_x(i) - y(i)) + (lambda).*x_ada;
        
        % Accumulate the Gradient
        grad_J_x_sq = grad_J_x*grad_J_x';
        grad_accum = grad_accum  + grad_J_x_sq;
        G_iter = sqrt(diag(grad_accum));
        
        % Update step
        x_new = x_ada - (eta./(zeta + G_iter)).*grad_J_x;
        
        if(mod(iter,50)==0)
            plot_boundary_2(A, y, x_new,'',fig5);
            pause(0.1);
        end
        
        % Compute the new z and hypothesis h_x(z)
        x_ada = x_new;
        x_sum = x_sum + x_ada;
        param = A*x_new;
        sigm_x = (1./(1 + exp(-param)));
        
        
        % Compute cost function value
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J_val = sum(log_loss)/m + (lambda/(2))*norm(x_ada)^2;

        % Update cost function vector
        J_ada = [J_ada J_val];

        if iter > MAX_ITERS || RFC < epsilon
            break;
        end

        if(mod(iter,m) == 0)
            J_ada_plot = [J_ada_plot J_val];
            x_epoch_prev = x_epoch;
            x_epoch = (1/m)*x_sum;
            x_list = [x_list x_epoch];
            x_sum = zeros(n,1);
            RFC = norm(x_epoch - x_epoch_prev)/norm(x_epoch_prev);
        end
        iter = iter + 1;    
        
    end
    J_ada_end = J_ada(end);
    iter
    RFC
    
end

