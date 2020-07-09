function [x_gd, J_gd, x_list] = solve_GD(n, m, A, y, J_init, x_init, sigm_x_init, alpha, lambda, MAX_ITER)
    
    fig2 = figure();
    
    % Initial values
    J = [J_init];
    x = x_init;
    sigm_x = sigm_x_init;  
    
    % Initializations
    Hessian_true = zeros(n, n);
    L_true = [];
    x_list = [x_init];   
    
    iter = 1;
    epsilon = 10^(-4);                                                     % RFC tol
    RFC = 1;
    
    while(1)
        
        if alpha == 0
            % Compute true L
            % Compute the Hessian of logistic cost function at every
            % iteration (see Chapter 2, https://engineering.purdue.edu/ChanGroup/ECE595/files/chapter2.pdf )
            for i=1:m
                Hessian_true =  Hessian_true + ((sigm_x(i)*(1 - sigm_x(i)))*(A(i,:)'*A(i,:)));
            end
            % Save values of L
            L_true = [L_true max(svd(Hessian_true))/m + lambda];
            % Compute step
            alpha = 1/L_true(end);
        end

        % Gradient Step
        grad_J_x = (1/m)*(A'*(sigm_x - y)) + (lambda).*x;
        x_prev = x;                                                        % save previous value for RFC
        x_new = x - alpha*grad_J_x;

        % Compute the new z and hypothesis h_x(z)
        x = x_new;
        x_list = [x_list x];
        
        % Plot the boundary
        if(mod(iter,1)==0)
            plot_boundary_2(A, y, x,'',fig2);
            pause(0.1);
        end
        
        % Compute cost function value
        param = A*x;
        sigm_x = (1./(1 + exp(-param)));
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J_val = sum(log_loss)/m + (lambda/(2))*norm(x)^2;

        % Update cost function vector
        J = [J J_val];
        
        % Compute RFC
        RFC = norm(x- x_prev)/norm(x_prev);
        if iter > MAX_ITER || RFC < epsilon
            break;
        end

        % Set Hessian to zero
        alpha = 0;
        Hessian_true = zeros(n, n);
        iter = iter + 1; 
        
    end
    
    x_gd = x;
    J_gd = J(end);
    
    iter
    RFC
end

