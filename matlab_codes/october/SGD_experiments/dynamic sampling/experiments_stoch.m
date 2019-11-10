clc, clear all, close all

%creation of data
m = 2000;
n = 100;
MaxIter = 20000;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 1;
lambda_max = 10;
eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);
eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);
A = U*Sigma*V';

%Normalize the rows of A to unit norm 
%A = normalize(A);

%vector of parameters
b = rand(m,1);
%or
%b = A*randn(n,1);

%Problem parameters
Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;

%initial point
x_init_all = randn(n,1);

%Optimal
x_star = inv(Hessian)*A'*b;
f_star = (1/(2*m))*norm(A*x_star - b)^2;
Beta = norm(x_star);
%residuals to estimate M
residuals = (1/2)*(A*x_star - b).^2;
max_res = max(residuals);
%computation of expected value E_i[||g(w,i)||^2] at optimum
% [expected_value_g_square_opt,stoch_squared] = compute_squared_mean_g(randperm(m,m), x_star, A, b);
full_grad_opt = (1/(m^2))*norm(A'*(A*x_star - b))^2;


% %%%%%%%%%%%%% Stochastic Gradients Different Steps%%%%%%%
%% %i) Nesterov (full problem) 
x_sd_sc_full_accel = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_sd_sc_full_accel = m/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'true';
opts.batch_size = m;
opts.MaxIter = 500;
[~, f_val_merged_full_nes, ~, ~] = stochastic_gradient(A, b, eta_sd_sc_full_accel, x_sd_sc_full_accel, f_val, opts);

%% %ii) Gradient (full batch)
x_gd_sc_full = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_gd_sc_full = m/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'false';
opts.batch_size = m;
opts.MaxIter = 500;
[~, f_val_merged_full_gd, ~, ~] = stochastic_gradient(A, b, eta_gd_sc_full, x_gd_sc_full, f_val, opts);

%% %ii) Mini-Batch (m/20)
x_sd_sc_mini = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_sd_sc_mini = m/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'false';
opts.batch_size = m/10;
opts.adaptive_sample ='true';
opts.MaxIter = 500;
[~, f_val_merged_mini, ~, ~] = stochastic_gradient(A, b, eta_sd_sc_mini, x_sd_sc_mini, f_val, opts);
 
figure(1)
semilogy(abs(f_val_merged_full_nes - f_star),'Linestyle',':','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_full_gd - f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini - f_star),'Linestyle',':','Linewidth',3);
grid on;
xlabel('Stochastic Gradients');
ylabel('|f_t - f^*|');
title('Mini Batch Stongly Convex')
legend('full Nesterov','full Gradient','mini Batch Adaptive');

