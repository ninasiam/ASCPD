function [x_sgd, J_sgd_end, x_list] = solve_SGD(n, m, A, y, J_init, x_init, sigm_x_init, alpha, lambda, MAX_ITERS);
    
    fig4 = figure();
    
    % Initial values
    J_sgd = [J_init];
    J_sgd_plot = [J_init];
    x_sgd = x_init;
    sigm_x = sigm_x_init;
    
    % Initializations
    L_true_i = [];
    x_list = [x_init];
    h_theta = [];
    x_sum = zeros(n,1);
    x_epoch = ones(n,1);
    
    epsilon = 10^(-4);                                                     % RFC tol
    iter = 1;
    RFC = 1;
    
    while(1)

        % Stochastic Gradient Step
        i = randi(m, 1, 1);
        grad_J_x = A(i,:)'*(sigm_x(i) - y(i)) + (lambda).*x_sgd;
        
        if alpha == 0                                                      % non constant step size
            % Compute L
            h_theta = [h_theta sigm_x(i)];
            L_true_i = [L_true_i (h_theta(end)*(1-h_theta(end)))*norm(A(i,:),2)^2+lambda];    
            %L_true_i = [L_true_i (1/2)*norm(A(i,:),2)^2+lambda];

            alpha = 1/(m*L_true_i(end));                                   % (edit: in all stoch algorithms!!! I put m in the denominator)
            
        end
        
        % Update 
        x_new = x_sgd - alpha*grad_J_x;
        
        % Plot the boundary
        if(mod(iter,50)==0)
            plot_boundary_2(A, y, x_new,'',fig4);
            pause(0.1);
        end
        
        % Compute the new z and hypothesis h_x(z)
        x_sgd = x_new;
        x_sum = x_sum + x_sgd;
        param = A*x_sgd;
        sigm_x = (1./(1 + exp(-param)));

        % Compute cost function value
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J_val = sum(log_loss)/m + (lambda/(2))*norm(x_sgd)^2;

        % Update cost function vector
        J_sgd = [J_sgd J_val];


        if(mod(iter,m) == 0)
            J_sgd_plot = [J_sgd_plot J_val];
            x_epoch_prev = x_epoch;
            x_epoch = (1/m)*x_sum;
            x_list = [x_list x_epoch];
            x_sum = zeros(n,1);
            RFC = norm(x_epoch - x_epoch_prev)/norm(x_epoch_prev);
            
        end
        
        if iter > MAX_ITERS || RFC < epsilon
            break;
        end

        alpha = 0;
        iter = iter + 1;  
    end
    J_sgd_end = J_sgd(end);
    iter
    RFC
    
end

