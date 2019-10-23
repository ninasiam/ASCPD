clc, clear all, close all

%creation of data
m = 2000;
n = 200;
MaxIter = 20000;

A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');

lambda_min = 10;
lambda_max = 1000;

eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);

eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);

A = U*Sigma*V';

%another_x
b = rand(m,1);

Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;
%A = normalize(A);
x_init_all = zeros(n,1);

%Optimal
x_star = inv(Hessian)*A'*b;
f_star = (1/(2*m))*norm(A*x_star - b)^2;
Beta = norm(x_star);
residuals = (1/2)*(A*x_star - b).^2
max_res = max(residuals);

[expected_value_g_square_opt,mtp] = compute_squared_mean_g(randperm(m,m), x_star, A, b);
full_grad_opt = (1/(m^2))*norm(A'*(A*x_star - b))^2;



% %%%%%%%%%%%%% Stochastic Gradients Different Steps%%%%%%%
% %%
%i) constant rho-Lipschitz 
x_sd = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_sd = Beta/(L*Beta + norm(A'*b)); %use a constant

opts = struct;
opts.StepSize = 'constant';
opts.epsilon = 10^(-14);
opts.MaxIter = 20000;

[fval_sd,~,norm_sd] = stochastic_gradient(A, b, eta_sd, x_sd, f_val, opts); 

%ii)constant rho-Lipschitz 
x_sd_v = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_sd_v = 100*Beta/(L*Beta + norm(A'*b)); %use a variant and a constant to enforce a bigger step size

opts = struct;
opts.StepSize = 'variant';
opts.epsilon = 10^(-14);
opts.MaxIter = 20000;

[fval_sd_v,~,norm_sd_v] = stochastic_gradient(A, b, eta_sd_v, x_sd_v, f_val, opts);

% %iii)strongly convex variant step
% x_sd_sc_v = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_v = 1/mu; %we use the mu of A'*A;
% 
% opts = struct;
% opts.average = 'true';
% opts.StepSize = 'variant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-14);
% opts.MaxIter = 20000;
% 
% [fval_sd_sc_v,~,norm_sd_sc_v] = stochastic_gradient(A, b, eta_sd_sc_v, x_sd_sc_v, f_val, opts);

%)iv Bottou strongly convex diminishing step sizes
x_sd_sc_bt = x_init_all;
f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt = 1/(L);

opts = struct;
opts.average = 'false';
opts.StepSize = 'constant';
opts.Function ='strongly-convex';
opts.epsilon = 10^(-14);
opts.MaxIter = 20000;
opts.mu = mu;
opts.gamma = 100;
opts.sampling_rate = 1000;
opts.maximal_res = max_res;
opts.batch = m/4;
[f_val_sd_sc_bt,~,norm_sd_sc_bt] = stochastic_gradient(A, b, eta_sd_sc_bt, x_sd_sc_bt, f_val, opts); 


figure(2)
semilogy(abs(fval_sd -f_star),'Linestyle',':','Linewidth',3)
hold on;
semilogy(abs(fval_sd_v -f_star),'Linestyle','--','Linewidth',3)
hold on;
% semilogy(abs(fval_sd_sc_v -f_star),'Linestyle',':','Linewidth',3);
% hold on;
semilogy(abs(f_val_sd_sc_bt - f_star),'Linestyle','--','Linewidth',3)
hold on;
mean_value = (max_res)/(2*mu).*ones(1,MaxIter);
semilogy(mean_value,'Linestyle','-','Linewidth',3)
hold off;
grid on;
xlabel('iterations');
ylabel('|f_t - f^*|');
title('Stochastic gradients with different step sizes')
legend('Beta/(L*Beta + norm(A^T*b))',...
       'Beta/(L*Beta + norm(A^T*b))*sqrt(t)', ...
       '1/L', 'bound');

% %%%%%%%%%%%%%%%%%%%%%%%%Mini-Batch%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
% %i) Nesterov (full problem) 
% x_sd_sc_full_accel = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_full_accel = m/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m;
% opts.MaxIter = 10000;
% [f_val_sd_sc_full_accel, f_val_merged_full, norm_sd_sc_full_accel] = stochastic_gradient(A, b, eta_sd_sc_full_accel, x_sd_sc_full_accel, f_val, opts); 
% 
% %) m/5 acceleration
% x_sd_sc_mini_accel1 = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel1 = (1)/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/5;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel1, f_val_merged_mini_n_5, norm_sd_sc_mini_accel1] = stochastic_gradient(A, b, eta_sd_sc_mini_accel1, x_sd_sc_mini_accel1, f_val, opts); 
% 
% % m/4 acceleration
% x_sd_sc_mini_accel = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel = (1)/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/4;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel,f_val_merged_mini_n_4,norm_sd_sc_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_mini_accel, x_sd_sc_mini_accel, f_val, opts); 
% 
% %) m/10 acceleration
% x_sd_sc_mini_accel10 = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel10 = (1)/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/10;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel10, f_val_merged_mini_n_10, norm_sd_sc_mini_accel10] = stochastic_gradient(A, b, eta_sd_sc_mini_accel10, x_sd_sc_mini_accel10, f_val, opts); 
% 
% % Stochastic batch = 1
% x_sd_sc_1_accel = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_1_accel = 1/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.Function ='strongly-convex';
% opts.epsilon = 10^(-6);
% opts.mini_batch = 'true';
% opts.accelerate = 'false';
% opts.batch_size = 1;
% opts.MaxIter = 10000;
% [f_val_sd_sc_1_accel,f_val_merged_mini_n_1_merged,norm_sd_sc_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_1_accel, x_sd_sc_1_accel, f_val, opts); 
% 
% figure(2)
% semilogy(abs(f_val_merged_full -f_star),'Linestyle',':','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_5 -f_star),'Linestyle','--','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_4 -f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_merged_mini_n_10 - f_star),'Linestyle','--','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_1_merged - f_star),'Linestyle',':','Linewidth',3)
% grid on;
% xlabel('Stochastic Gradients');
% ylabel('|f_t - f^*|');
% title('Mini Batch Stongly Convex')
% legend('m/L',...
%        '(m/5)/L', '(m/4)/L', '(m/10)/L',...
%        '1/L');
%    
% 
% %%%%%%%%%%%%%%%%%%%%%%%%Mini-Batch ~Smooth%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
% %i) Nesterov (full problem) 
% x_sd_sc_full_accel_s = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_full_accel_s = m/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m;
% opts.MaxIter = 10000;
% [f_val_sd_sc_full_accel_s, f_val_merged_full_s, norm_sd_sc_full_accel_s] = stochastic_gradient(A, b, eta_sd_sc_full_accel_s, x_sd_sc_full_accel_s, f_val, opts); 
% 
% %) m/5 acceleration
% x_sd_sc_mini_accel1_s = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel1_s = (1)/(L); %!!!
% 
% opts = struct;
% opts.averaging = 'true';
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/5;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel1_s, f_val_merged_mini_n_5_s, norm_sd_sc_mini_accel1_s] = stochastic_gradient(A, b, eta_sd_sc_mini_accel1_s, x_sd_sc_mini_accel1_s, f_val, opts); 
% 
% % m/4 acceleration
% x_sd_sc_mini_accel_s = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel_s = (1)/(L); %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/4;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel_s,f_val_merged_mini_n_4_s,norm_sd_sc_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_mini_accel_s, x_sd_sc_mini_accel_s, f_val, opts); 
% 
% %) m/10 acceleration
% x_sd_sc_mini_accel10_s = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_mini_accel10_s = (1)/(L); %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-4);
% opts.mini_batch = 'true';
% opts.accelerate = 'true';
% opts.batch_size = m/10;
% opts.MaxIter = 10000;
% [f_val_sd_sc_mini_accel10_s, f_val_merged_mini_n_10_s, norm_sd_sc_mini_accel10] = stochastic_gradient(A, b, eta_sd_sc_mini_accel10_s, x_sd_sc_mini_accel10_s, f_val, opts); 
% 
% % Stochastic batch = 1
% x_sd_sc_1_accel_s = x_init_all;
% f_val = (1/(2*m))*norm(A*x_init_all - b)^2;
% eta_sd_sc_1_accel_s = 1/L; %!!!
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-6);
% opts.mini_batch = 'true';
% opts.accelerate = 'false';
% opts.batch_size = 1;
% opts.MaxIter = 10000;
% [f_val_sd_sc_1_accel_s,f_val_merged_mini_n_1_merged_s,norm_sd_sc_mini_accel_s] = stochastic_gradient(A, b, eta_sd_sc_1_accel_s, x_sd_sc_1_accel_s, f_val, opts); 
% 
% figure(3)
% semilogy(abs(f_val_merged_full_s -f_star),'Linestyle',':','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_5_s -f_star),'Linestyle','--','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_4_s -f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_merged_mini_n_10_s - f_star),'Linestyle','--','Linewidth',3)
% hold on;
% semilogy(abs(f_val_merged_mini_n_1_merged_s - f_star),'Linestyle',':','Linewidth',3)
% grid on;
% xlabel('Stochastic Gradients');
% ylabel('|f_t - f^*|');
% title('Mini Batch Smooth')
% legend('m/L',...
%        '(m/5)/(L)', '(m/4)/(L)', '(m/10)/(L)',...
%        '1/L');