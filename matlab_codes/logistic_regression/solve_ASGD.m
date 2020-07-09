function [x_asgd, J_asgd_end, x_list] = solve_ASGD(n, m, A, y, J_init, x_init, sigm_x_init, alpha, lambda, MAX_ITERS);
    
    fig6 = figure();
    
    % Initial values
    J_asgd = [J_init];
    J_asgd_plot = [J_init];
    x_asgd = x_init;
    y_asgd = x_init;
    
    % Cost function at point y
    param = A*y_asgd;
    sigm_x = (1./(1 + exp(-param)));
    
    % Initializations
    h_theta = [];
    L_true_i = [];
    x_list = [x_init];
    x_sum = zeros(n,1);
    x_epoch = ones(n,1);
    
    iter = 1;
    alpha = alpha;
    beta = 0.09;
    epsilon = 10^(-4);                                                     % RFC tol
    RFC = 1;

    while(1)

        % Stochastic Gradient Step at point y
        i = randi(m, 1, 1);
        grad_J_y = A(i,:)'*(sigm_x(i) - y(i)) + (lambda).*y_asgd;
        
        if alpha == 0                                                      % non constant step size
            % Compute L via an upper bound 
            h_theta = [h_theta sigm_x(i)];
            L_true_i = [L_true_i (h_theta(end)*(1-h_theta(end)))*norm(A(i,:),2)^2+lambda];    
            %L_true_i = [L_true_i (1/2*m)*norm(A(i,:),2)^2+lambda];

            alpha = 1/(m*L_true_i(end));
            
        end
        
        beta = (1 - (L_true_i(end)/lambda))/(1 + (L_true_i(end)/lambda));  % momentum parameter (estimation)
        x_new = y_asgd - alpha*grad_J_y;
        y_new = x_new + beta*(x_new - x_asgd);                             % interpolation

        % Compute the new z and hypothesis h_x(z)
        x_asgd = x_new;
        x_sum = x_sum + x_asgd;
        param = A*x_asgd;
        sigm_x = (1./(1 + exp(-param)));
        y_asgd = y_new;
        
        % Plot the boundary 
        if(mod(iter,50)==0)
            plot_boundary_2(A, y, x_new,'',fig6);
            pause(0.1);
        end
        
        % Compute cost function value
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J_val = sum(log_loss)/m + (lambda/(2))*norm(x_asgd)^2;

        % Update cost function vector
        J_asgd = [J_asgd J_val];

        if iter > MAX_ITERS || RFC < epsilon
            break;
        end

        if(mod(iter,m) == 0)
            J_asgd_plot = [J_asgd_plot J_val];
            x_epoch_prev = x_epoch;                                        % for RFC
            x_epoch = (1/m)*x_sum;                                         % x_epoch                                       
            x_list = [x_list x_epoch];
            x_sum = zeros(n,1);
            RFC = norm(x_epoch - x_epoch_prev)/norm(x_epoch_prev);
        end
        
        iter = iter + 1;    
        alpha = 0;
    end
    
    J_asgd_end = J_asgd(end);
    iter
    RFC
    
end

