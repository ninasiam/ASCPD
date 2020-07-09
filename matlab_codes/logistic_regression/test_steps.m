%% Logistic Regression (GD/SGD) optimal steps
%  Ioanna Siaminou
%  30/06/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all;

% Problem variables
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
Hessian_est = A'*A/m;
L_est = max(svd(Hessian_est));

% True Hessian (Computed at each step)
Hessian_true = zeros(n+1, n+1);
alpha = 1/L_est; %0.1;

L_true = [];
while(1)
    
    % Commpute the Hessian 
  %  for i=1:m
   %     Hessian_true =  Hessian_true + ((sigm_x(i)*(1 - sigm_x(i)))*(A(i,:)'*A(i,:)));
  %  end
    
   % L_true = [L_true max(svd(Hessian_true))];
   % L_true = L_true/m;
   % alpha = 1/L_true(end);
    
    % Gradient Step
    grad_J_x = (1/m)*(A'*(sigm_x - y));
    x_new = x - alpha*grad_J_x;
    
    % Compute the new z and hypothesis h_x(z)
    x = x_new;
    param = A*x_new;
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J = [J J_val];
    
    if iter > 500
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

%% Initial values

% log loss
param = A*x_sgd;
sigm_x = (1./(1 + exp(-param)));
log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));

iter = 1;
L_true_i = [];
alpha = 0.01;
h_theta = [];
X = [];
% alpha = 1;
while(1)
    
    % Stochastic Gradient Step
    i = randi(m, 1, 1);
    grad_J_x = A(i,:)'*(sigm_x(i) - y(i));
    
    % Compute L
    h_theta = [h_theta sigm_x(i)];
    %L_true_i = [L_true_i (h_theta(end)*(1-h_theta(end)))*norm(A(i,:),2)^2];
    L_true_i = [L_true_i (1/2)*norm(A(i,:),2)^2];

    alpha = 1/L_true_i(end);
    
    % Update 
    x_new = x_sgd - alpha*grad_J_x;
  %  X = [X x_new]
    
    % Compute the new z and hypothesis h_x(z)
    x_sgd = x_new;
    param = A*x_sgd;
   % figure(1)
  %  plot(X)
  %  hold on
  %  pause()
    sigm_x = (1./(1 + exp(-param)));
    
    % Compute cost function value
    log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
    J_val = sum(log_loss)/m;
    
    % Update cost function vector
    J_sgd = [J_sgd J_val];
   
    if iter > 50000
        break;
    end
    
    if(mod(iter,n) == 0)
        J_sgd_plot = [J_sgd_plot J_val];
    end
%     alpha = 1/iter;
    iter = iter + 1  
    %alpha = alpha/(iter);
end


figure(2);
semilogy(J); 
hold on;
semilogy(J_sgd_plot);
leg = legend('Gradient Descent', 'Stochastic Gradient Descent')
xlabel('iterations');
ylabel('cost fun value')