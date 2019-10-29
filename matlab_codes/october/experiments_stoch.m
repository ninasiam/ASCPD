clc, clear all, close all

%creation of data
n = 500;

A_tmp = randn(n,n);
[U,S,V] = svd(A_tmp);

lambda_min = 10;
lambda_max = 100;

eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);

eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);

A = U*Sigma*V';
b = rand(n,1);

Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;

x_init_all = zeros(n,1);

%Optimal
x_star = inv(Hessian)*A'*b;
f_star = (1/(2*n))*norm(A*x_star - b)^2;
Beta = norm(x_star);

%%%%%%%%%%%%% Stochastic Gradients Different Steps%%%%%%%
%i) constant rho-Lipschitz 
x_sd = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd = Beta/(L*Beta + norm(A'*b));

opts = struct;
opts.StepSize = 'constant';
opts.epsilon = 10^(-14);
opts.MaxIter = 10000;

[fval_sd,~,norm_sd] = stochastic_gradient(A, b, eta_sd, x_sd, f_val, opts); 

%ii)constant rho-Lipschitz 
x_sd_v = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_v = Beta/(L*Beta + norm(A'*b));

opts = struct;
opts.StepSize = 'variant';
opts.epsilon = 10^(-14);
opts.MaxIter = 10000;

[fval_sd_v,~,norm_sd_v] = stochastic_gradient(A, b, eta_sd_v, x_sd_v, f_val, opts);

%iii)strongly convex variant step
x_sd_sc_v = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_v = 1/mu; %we use the mu of A'*A;

opts = struct;
opts.StepSize = 'variant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-14);
opts.MaxIter = 10000;

[fval_sd_sc_v,~,norm_sd_sc_v] = stochastic_gradient(A, b, eta_sd_sc_v, x_sd_sc_v, f_val, opts);

%)iv Bottou strongly convex diminishing step sizes
x_sd_sc_bt = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt = 1/L;

opts = struct;
opts.StepSize = 'variant-method2';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-14);
opts.MaxIter = 10000;
opts.mu = mu;
opts.gamma = 5;
[f_val_sd_sc_bt,~,norm_sd_sc_bt] = stochastic_gradient(A, b, eta_sd_sc_bt, x_sd_sc_bt, f_val, opts); 


figure(1)
semilogy(abs(fval_sd -f_star),'Linestyle',':','Linewidth',3)
hold on;
semilogy(abs(fval_sd_v -f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(fval_sd_sc_v -f_star),'Linestyle',':','Linewidth',3);
hold on;
semilogy(abs(f_val_sd_sc_bt - f_star),'Linestyle','--','Linewidth',3)
grid on;
xlabel('iterations');
ylabel('|f_t - f^*|');
title('Stochastic gradients with different step sizes')
legend('Beta/(L*Beta + norm(A^T*b))',...
       'Beta/(L*Beta + norm(A^T*b))*sqrt(t)','1/(mu*sqrt(t))', ...
       '1/L or 2/(mu*(gamma + t))');
%%%%%%%%%%%%%%%%%%%%%%%%Mini-Batch%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%i) Nesterov (full problem) 
x_sd_sc_full_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_full_accel = n/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'true';
opts.batch_size = n;
opts.MaxIter = 10000;
[f_val_sd_sc_full_accel, f_val_merged_full, norm_sd_sc_full_accel] = stochastic_gradient(A, b, eta_sd_sc_full_accel, x_sd_sc_full_accel, f_val, opts); 

%) n/5 acceleration
x_sd_sc_mini_accel1 = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_mini_accel1 = (n/5)/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'true';
opts.batch_size = n/5;
opts.MaxIter = 10000;
[f_val_sd_sc_mini_accel1, f_val_merged_mini_n_5, norm_sd_sc_mini_accel1] = stochastic_gradient(A, b, eta_sd_sc_mini_accel1, x_sd_sc_mini_accel1, f_val, opts); 

% n/4 acceleration
x_sd_sc_mini_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_mini_accel = (n/4)/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'true';
opts.batch_size = n/4;
opts.MaxIter = 10000;
[f_val_sd_sc_mini_accel,f_val_merged_mini_n_4,norm_sd_sc_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_mini_accel, x_sd_sc_mini_accel, f_val, opts); 

%) n/10 acceleration
x_sd_sc_mini_accel10 = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_mini_accel10 = (n/10)/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-4);
opts.mini_batch = 'true';
opts.accelerate = 'true';
opts.batch_size = n/10;
opts.MaxIter = 10000;
[f_val_sd_sc_mini_accel10, f_val_merged_mini_n_10, norm_sd_sc_mini_accel10] = stochastic_gradient(A, b, eta_sd_sc_mini_accel10, x_sd_sc_mini_accel10, f_val, opts); 

% Stochastic batch = 1
x_sd_sc_1_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_1_accel = 1/L; %!!!

opts = struct;
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-6);
opts.mini_batch = 'true';
opts.accelerate = 'false';
opts.batch_size = 1;
opts.MaxIter = 10000;
[f_val_sd_sc_1_accel,f_val_merged_mini_n_1_merged,norm_sd_sc_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_1_accel, x_sd_sc_1_accel, f_val, opts); 

figure(2)
semilogy(abs(f_val_merged_full -f_star),'Linestyle',':','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini_n_5 -f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini_n_4 -f_star),'Linestyle',':','Linewidth',3);
hold on;
semilogy(abs(f_val_merged_mini_n_10 - f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini_n_1_merged - f_star),'Linestyle',':','Linewidth',3)
grid on;
xlabel('iterations');
ylabel('|f_t - f^*|');
title('Mini Batch Stongly Convex')
legend('n/L',...
       '(n/5)/L', '(n/4)/L', '(n/10)/L',...
       '1/L');