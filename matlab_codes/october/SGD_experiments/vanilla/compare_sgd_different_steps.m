%Compare stochastic methods

%Stochastic Gradient
%       --> i)  rho-Lipschitz ~constant step
%       --> ii) rho-Lipschitz ~variant step
%       --> iii) mu-strongly convex
%Nesterov Accelerated Gradient
%       --> i)  L-smooth
%       --> ii) L-smooth, mu-strongly convex
clc, clear all, close all

%creation of data
n = 1000;

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

% %%%%%%%%%%%%%%%%%%%% step Ben-David, Shalev - Shwartz %%%%%%%%%%%%%%%%%%%%
% %i) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz ~ constant step Ben-David,
% %Shalev - Shwartz
% x_sd = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd = Beta/(L*Beta + norm(A'*b));
% 
% opts = struct;
% opts.StepSize = 'constant';
% opts.epsilon = 10^(-14);
% %opts.MaxIter = 10000;
% 
% [fval_sd,~,norm_sd] = stochastic_gradient(A, b, eta_sd, x_sd, f_val, opts); 
% 
% %ii) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz ~ variant step Ben-David,
% %Shalev - Shwartz
% x_sd_v = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd = Beta/(L*Beta + norm(A'*b));
% 
% opts3 = struct;
% opts3.StepSize = 'variant';
% opts3.epsilon = 10^(-14);
% %opts3.MaxIter = 10000;
% 
% [fval_sd_v,~,norm_sd_v] = stochastic_gradient(A, b, eta_sd, x_sd_v, f_val, opts3); 
% 
% %iii) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz, mu strongly
% %convex
% 
% x_sd_sc = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd_sc = 1/mu;
% 
% opts2 = struct;
% opts2.StepSize = 'variant';
% opts2.Function ='strongly-convex';
% opts2.epsilon = 10^(-14);
% %opts2.MaxIter = 10000;
% [f_val_sd_sc,~,norm_sd_sc] = stochastic_gradient(A, b, eta_sd_sc, x_sd_sc, f_val, opts2); 
% 
% %iv) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz, mu strongly
% %convex than l(0,z) <= 1
% 
% eps = 10^(1);
% x_sd_sm_learn = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd_sm_learn = 1/(L*(1+(3/eps)));
% 
% opts6 = struct;
% opts6.StepSize = 'constant';
% opts6.Function ='smooth';
% opts6.epsilon = 10^(-14);
% %opts6.MaxIter = 10000;
% [f_val_sd_sm_learn,~,norm_sd_sm_learn] = stochastic_gradient(A, b, eta_sd_sm_learn, x_sd_sm_learn, f_val, opts6); 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%% Empirical steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %v) Solve minf(x) = (1/2n)||Ax - b||_2, f - smooth constant
% 
% x_sd_sm = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd_sm = 1/L;
% 
% opts4 = struct;
% opts4.StepSize = 'constant';
% opts4.Function ='smooth';
% opts4.epsilon = 10^(-14);
% %opts4.MaxIter = 10000;
% [f_val_sd_sm,~,norm_sd_sm] = stochastic_gradient(A, b, eta_sd_sm, x_sd_sm, f_val, opts4); 
% 
% %vi) Solve minf(x) = (1/2n)||Ax - b||_2, f - smooth variant
% x_sd_sm_v = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd_sm_v = 1/L;
% 
% opts5 = struct;
% opts5.StepSize = 'variant';
% opts5.Function ='smooth';
% opts5.epsilon = 10^(-14);
% %opts5.MaxIter = 10000;
% [f_val_sd_sm_v,~,norm_sd_sm_v] = stochastic_gradient(A, b, eta_sd_sm_v, x_sd_sm_v, f_val, opts5); 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%Bottou%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %vii) Solve minf(x) = (1/2n)||Ax - b||_2, f - smooth, mu strongly
% %convex 
% x_sd_sc_bt = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% eta_sd_sc_bt = 1/L;
% 
% opts7 = struct;
% opts7.StepSize = 'variant-method2';
% opts7.Function ='strongly-convex';
% opts7.epsilon = 10^(-14);
% %opts7.MaxIter = 10000;
% opts7.mu = mu;
% opts7.gamma = 5;
% [f_val_sd_sc_bt,~,norm_sd_sc_bt] = stochastic_gradient(A, b, eta_sd_sc_bt, x_sd_sc_bt, f_val, opts7); 

%%%Bottou mini-batch
%mini batch without accel
x_sd_sc_bt_mini = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt_mini = (n/4)/(L); %!!!


opts9 = struct;
opts9.StepSize = 'constant';
opts9.Function ='strongly-convex';
opts9.epsilon = 10^(-4);
opts9.mini_batch = 'true';
opts9.batch_size = n/4;
opts9.MaxIter = 10000;
[f_val_sd_sc_bt_mini,f_val_merged_mini,norm_sd_sc_bt_mini] = stochastic_gradient(A, b, eta_sd_sc_bt_mini, x_sd_sc_bt_mini, f_val, opts9); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Accelerated Mini-Batch%%%%%%%%%%%%%%%%%%%%%
x_sd_sc_bt_full_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt_full_accel = n/L; %!!!

opts11 = struct;
opts11.StepSize = 'constant';
opts11.Function ='strongly-convex';
opts11.epsilon = 10^(-4);
opts11.mini_batch = 'true';
opts11.accelerate = 'true';
opts11.batch_size = n;
opts11.MaxIter = 10000;
[f_val_sd_sc_bt_full_accel,f_val_merged_full_merged,norm_sd_sc_bt_full_accel] = stochastic_gradient(A, b, eta_sd_sc_bt_full_accel, x_sd_sc_bt_full_accel, f_val, opts11); 

x_sd_sc_bt_mini_accel1 = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt_mini_accel1 = (n/5)/L; %!!!

opts12 = struct;
opts12.StepSize = 'constant';
opts12.Function ='strongly-convex';
opts12.epsilon = 10^(-4);
opts12.mini_batch = 'true';
opts12.accelerate = 'true';
opts12.batch_size = n/5;
opts12.MaxIter = 10000;
[f_val_sd_sc_bt_mini_accel1,f_val_merged_mini_n_5_merged,norm_sd_sc_bt_mini_accel1] = stochastic_gradient(A, b, eta_sd_sc_bt_mini_accel1, x_sd_sc_bt_mini_accel1, f_val, opts12); 

x_sd_sc_bt_mini_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt_mini_accel = (n/4)/L; %!!!

opts10 = struct;
opts10.StepSize = 'constant';
opts10.Function ='strongly-convex';
opts10.epsilon = 10^(-4);
opts10.mini_batch = 'true';
opts10.accelerate = 'true';
opts10.batch_size = n/4;
opts10.MaxIter = 10000;
[f_val_sd_sc_bt_mini_accel,f_val_merged_mini_n_4_merged,norm_sd_sc_bt_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_bt_mini_accel, x_sd_sc_bt_mini_accel, f_val, opts10); 

x_sd_sc_bt_1_accel = x_init_all;
f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
eta_sd_sc_bt_1_accel = 1/L; %!!!

opts13 = struct;
opts13.StepSize = 'constant';
opts13.Function ='strongly-convex';
opts13.epsilon = 10^(-6);
opts13.mini_batch = 'true';
opts13.accelerate = 'false';
opts13.batch_size = 1;
opts13.MaxIter = 10000;
[f_val_sd_sc_bt_1_accel,f_val_merged_mini_n_1_merged,norm_sd_sc_bt_mini_accel] = stochastic_gradient(A, b, eta_sd_sc_bt_1_accel, x_sd_sc_bt_1_accel, f_val, opts13); 

% %%%%%%%%%%%%%%%%https://icml.cc/2012/papers/261.pdf paper%%%%%%%%%%%%%%%%%
% %iii) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz, mu strongly
% %convex
% x_sd_sc_pp = x_init_all;
% f_val = (1/(2*n))*norm(A*x_init_all - b)^2;
% c = 0.8;
% eta_sd_sc_pp = c/mu;
% 
% opts8 = struct;
% opts8.StepSize = 'variant';
% opts8.Function ='strongly-convex';
% opts8.epsilon = 10^(-14);
% %opts8.MaxIter = 10000;
% [f_val_sd_sc_pp,~,norm_sd_sc_pp] = stochastic_gradient(A, b, eta_sd_sc_pp, x_sd_sc_pp, f_val, opts8); 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%PLOTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% semilogy(abs(fval_sd -f_star),'Linestyle','--','Linewidth',3);
% hold on;
% semilogy(abs(fval_sd_v - f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sc - f_star),'Linestyle','--','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sm - f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sm_v - f_star),'Linestyle','--','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sm_learn - f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sc_bt - f_star),'Linestyle','--','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sc_pp - f_star),'Linestyle',':','Linewidth',3);
% grid on;
% xlabel('iterations');
% ylabel('|f_t - f^*|');
% legend('SGD rho-Lip constant step', 'SGD rho-Lip  variant step',...
%     'SGD rho-Lip, mu-strong convex','SGD smooth function empirical',...
%     'SGD smooth function empirical variant','SGD smooth constant step (learning)',...
%     'SGD strongly convex, variant step bottou', 'SGD strongly convex using 1>c>1/2');
% hold off;
% 
% figure(2)
% plot(norm_sd - Beta);
% hold on;
% plot(norm_sd_v - Beta);
% hold on;
% plot(norm_sd_sc - Beta);
% hold on;
% plot(norm_sd_sm - Beta);
% hold on;
% plot(norm_sd_sm_v - Beta);
% hold on;
% plot(norm_sd_sm_learn - Beta);
% hold on;
% plot(norm_sd_sc_bt - Beta);
% hold on;
% plot(norm_sd_sc_pp - Beta);
% grid on;
% xlabel('iterations');
% ylabel('$\|{\bf x }\|_2 ~ - ~ \|{\bf x}_* \|_2$','fontsize',14,'interpreter','latex');
% legend('SGD rho-Lip constant step', 'SGD rho-Lip variant step',...
% 'SGD rho-Lip, mu-strong convex','SGD smooth function empirical',...
% 'SGD smooth function empirical variant', 'SGD smooth constant step (learning)',...
% 'SGD strongly convex, variant step bottou','SGD strongly convex using 1>c>1/2');
% hold off
% 
% figure(3)
% semilogy(abs(f_val_sd_sc_bt_full_accel -f_star),'Linestyle',':','Linewidth',3)
% hold on;
% semilogy(abs(f_val_sd_sc_bt_mini_accel -f_star),'Linestyle','--','Linewidth',3)
% hold on;
% semilogy(abs(f_val_sd_sc_bt_mini_accel1 -f_star),'Linestyle',':','Linewidth',3);
% hold on;
% semilogy(abs(f_val_sd_sc_bt_mini -f_star),'Linestyle',':','Linewidth',3);
% grid on;
% xlabel('iterations');
% ylabel('|f_t - f^*|');
% title('Mini-batch')
% legend('Accelerated Full-batch (n/4) constant step strongly convex',...
%        'Accelerated Mini-batch (n/4) constant step strongly convex','Accelerated Mini-batch (n/5) constant step strongly convex', ...
%        'Mini-batch (n/4) constant step strongly convex');
   
figure(4)
semilogy(abs(f_val_merged_full_merged -f_star),'Linestyle',':','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini_n_4_merged -f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini_n_5_merged -f_star),'Linestyle',':','Linewidth',3);
hold on;
semilogy(abs(f_val_merged_mini_n_1_merged - f_star),'Linestyle','--','Linewidth',3)
hold on;
semilogy(abs(f_val_merged_mini -f_star),'Linestyle',':','Linewidth',3);
grid on;
xlabel('stochastic gradients');
ylabel('|f_t - f^*|');
title('Mini-batch')
legend('Accelerated Full-batch constant step strongly convex',...
       'Accelerated Mini-batch (n/4) constant step strongly convex','Accelerated Mini-batch (n/5) constant step strongly convex', ...
       'Accelerated Mini-batch (1) constant step strongly convex','Mini-batch (n/4) constant step strongly convex');