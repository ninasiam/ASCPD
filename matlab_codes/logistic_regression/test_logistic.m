%% Logistic Regression (GD/SGD/AgaGrad)
%  Ioanna Siaminou
%  23/06/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
clear ; close all; clc

addpath('Libraries/cvx')
%% Create Data
m = 500;
n = 100;

% Ground truth
x_star = randn(n + 1, 1);				
%x_star = x_star./norm(x_star);

A = randn(m,n);
A = [ones(m, 1) A];

% Measurements in y
y = (rand(m,1) < exp(A*x_star)./(1+exp(A*x_star)));

%% Initial values
x_init = randn(n + 1, 1);
%% Solution via cvx
% cvx_begin
%     variable x(n+1)
%     minimize (1/m * (-y'*log(1/(1 + exp(-A*x))) - (1 - y)'*(log(1 - 1/(1 + exp(-A*x))))) )
% cvx_end

cvx_expert true
cvx_begin
variables x(n+1)
    minimize((1/m)*(-y'*A*x+sum(log_sum_exp([zeros(1,m); x'*A']))))
cvx_end
%% Initial values


% log loss
param = A*x_init;
sigm_x = (1./(1 + exp(-param)));
log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));

% initial cost function value
J_init = sum(log_loss)/m;

%% Solution via Gradient
fprintf('Solution via GD...\n');
J = [J_init];
x = x_init;
iter = 1;
% Upper bound for L
Hessian_est = (A'*A)/m;
L = max(svd(Hessian_est));

Hessian_true = zeros(n+1, n+1);
alpha = 1/L;%0.1;

L_true = [];
while(1)
    
    % Commpute an estimate for L
    for i=1:m
        Hessian_true =  Hessian_true + ((sigm_x(i)*(1 - sigm_x(i)))*(A(i,:)'*A(i,:)));
    end
    
    %Hessian_true = (sigm_x'*(1 - sigm_x))*Hessian_est;
    L_true = [L_true max(svd(Hessian_true))/m];
    alpha = 1/L_true(end);
    
    % Gradient Step
    grad_J_x = (1/m)*(A'*(sigm_x - y));
    x_new = x - alpha*grad_J_x;
    
    % Compute the new z and hypothesis h_x(z)
    x = x_new;
    figure(1)
    plot(x)
    param = A*x_new;
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J = [J J_val];
    
    if iter > 2000
        break;
    end
    
    Hessian_true = zeros(n+1, n+1);
    iter = iter + 1;    
end

%% Solution via Stochast Gradient
fprintf('Solution via SGD...\n');
J_sgd = [J_init];
J_sgd_plot = [J_init];
x_sgd = x_init;
iter = 1;
L_true_i = [];
%alpha = 0.01;
% alpha = 1;
while(1)
    
    % Stochastic Gradient Step
    i = randi(m, 1, 1);
    grad_J_x = A(i,:)'*(sigm_x(i) - y(i));
    
    % Compute L
    %L_true_i = [L_true_i (sigm_x(i)*(1 - sigm_x(i)))*norm(A(i,:),2)^2];
    L_true_i = [L_true_i (1/2)*norm(A(i,:),2)^2];

    alpha = 1/L_true_i(end);
    
    % Update 
    x_new = x_sgd - alpha*grad_J_x;
    
    % Compute the new z and hypothesis h_x(z)
    x_sgd = x_new;
    param = A*x_new;
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J_sgd = [J_sgd J_val];
   
    if iter > 2000*m
        break;
    end
    
    if(mod(iter,100) == 0)
        J_sgd_plot = [J_sgd_plot J_val];
    end
%     alpha = 1/iter;
    iter = iter + 1;    
    %alpha = alpha/(iter);
end

%% Solution via Stochast Gradient (Accelerated)
fprintf('Solution via SGD (with momentum)...\n');
J_asgd = [J_init];
J_asgd_plot = [J_init];
x_asgd = x_init;
y_asgd = x_init;
% At point y
param = A*y_asgd;
sigm_x = (1./(1 + exp(-param)));
iter = 1;
alpha = 0.01;
beta = 0.5;
% alpha = 1;
while(1)
    
    % Stochastic Gradient Step
    i = randi(m, 1, 1);
    grad_J_x = A(i,:)'*(sigm_x(i) - y(i));
    x_new = y_asgd - alpha*grad_J_x;
    y_new = x_new + beta*(x_new - x_asgd);
    
    % Compute the new z and hypothesis h_x(z)
    x_asgd = x_new;
    param = A*x_new;
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J_asgd = [J_asgd J_val];
   
    if iter > 200000
        break;
    end
    
    if(mod(iter,100) == 0)
        J_asgd_plot = [J_asgd_plot J_val];
    end
%     alpha = 1/iter;
    iter = iter + 1;    
    %alpha = alpha/(iter);
end

%% Solution via AdaGrad
fprintf('Solution via AdaGrad...\n');
J_ada = [J_init];
J_ada_plot = [J_init];
grad_accum = zeros(n + 1);
x_ada = x_init;
iter = 1;
eta = 0.1;
zeta = 10^(-10);
while(1)
    
    % Stochastic Gradient Step
    i = randi(m, 1, 1);
    grad_J_x = A(i,:)'*(sigm_x(i) - y(i));
    % Accumulate the Gradient
    grad_J_x_sq = grad_J_x*grad_J_x';
    grad_accum = grad_accum  + grad_J_x_sq;
    G_iter = sqrt(diag(grad_accum));
    % Update step
    x_new = x_ada - (eta./(zeta + G_iter)).*grad_J_x;
    
    % Compute the new z and hypothesis h_x(z)
    x_ada = x_new;
    param = A*x_new;
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J_ada = [J_ada J_val];
   
    if iter > 200000
        break;
    end
    
    if(mod(iter,100) == 0)
        J_ada_plot = [J_ada_plot J_val];
    end
    iter = iter + 1;    
end

grid on;
semilogy(J); 
hold on;
semilogy(J_sgd_plot);
hold on;
semilogy(J_asgd_plot);
hold on;
semilogy(J_ada_plot);
leg = legend('Gradient Descent', 'Stochastic Gradient Descent','Stochastic Gradient Descent (Acceleration)', 'AdaGrad')
xlabel('iterations');
ylabel('cost fun value')