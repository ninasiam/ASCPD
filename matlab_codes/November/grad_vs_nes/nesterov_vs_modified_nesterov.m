clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 10000;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 1;
lambda_max = 100;
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

A_t_b = A'*b;
%initial point
x_init_all = zeros(n,1);

%Optimal
x_star = inv(Hessian)*A_t_b;
f_star1 = (1/(2))*norm(A*x_star - b)^2;
f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);


%%Accelerated Gradient Descend
tic();
iter = 1;
x_nes(:,iter) = x_init_all;
y(:,iter) = x_init_all;

f_val_nes_init = (1/(2))*norm(A*x_nes(:,iter) - b)^2;
f_val_nes(iter) = f_val_nes_init;

Q= mu/L;
b_par = (1-sqrt(Q))/(1+sqrt(Q));

%Nesterov main loop 

while(1)

    x_nes(:,iter+1) = y(:,iter) - (1/(L))*(Hessian*y(:,iter) - A_t_b);
    
    y(:,iter+1) = x_nes(:,iter+1) + b_par*(x_nes(:,iter+1) - x_nes(:,iter));

    f_val_nes(iter+1) = (1/(2))*norm(A*x_nes(:,iter+1) - b)^2;
    
    crit_nes = norm(x_nes(:,iter+1) - x_nes(:,iter))/norm(x_nes(:,iter));
    
    if((f_val_nes(iter+1) - f_star1 < ep) ||  iter > MAX_OUTER_ITER)%crit_nes < ep || iter > MAX_OUTER_ITER)%abs(f_val_nes(k) - f_star) < ep)
        break;
    end
    
    iter = iter + 1;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nesterov');
time_nes = toc 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         Modified Nesterov                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic();
iter = 1;

alpha(iter) = 1;
alpha(iter + 1) = 1;

beta(iter) = (1/(2*L));
beta(iter + 1) = (1/(2*L));

lambda(iter) = (2/L);
lambda(iter + 1) = (2/L);

x_nes_ag(:,iter) = x_init_all;
x_nes_mod(:,iter) = x_init_all;
while(1)
    
    x_nes_md(:,iter+1) = (1 - alpha(iter+1))*x_nes_ag(:,iter) + alpha(iter+1)*x_nes_mod(:,iter);
    
    grad_md = Hessian*x_nes_md(:,iter+1) - A_t_b;
    
    x_nes_mod(:,iter + 1) = x_nes_mod(:, iter) - lambda(iter + 1)*grad_md;
    
    x_nes_ag(:,iter + 1) = x_nes_md(:,iter+1) - beta(iter + 1)*grad_md;
    
    crit_nes = norm(x_nes_mod(:,iter+1) - x_nes_mod(:,iter))/norm(x_nes_mod(:,iter));
    
    f_val_nes_mod(iter+1) = (1/(2))*norm(A*x_nes_mod(:,iter+1) - b)^2;
    if((f_val_nes_mod(iter+1) - f_star1 < ep) ||  iter > MAX_OUTER_ITER)%crit_nes < ep || iter > MAX_OUTER_ITER)%abs(f_val_nes(k) - f_star) < ep)
        break;
    end
    iter = iter + 1;
    alpha(iter + 1) = 0.99;
    beta(iter + 1) = 1/(2*L);
    lambda(iter + 1) = 2/L;
end
fprintf('Nesterov Modified');
time_nes_mod = toc 

figure(1)
semilogy(f_val_nes_mod - f_star1);
hold on;
semilogy(f_val_nes - f_star1);
hold off
%xlim([0 10^4])
ylim([10^(-4) 10^6])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('Nesterov Modified', 'Nesterov');
xlabel('iterations');
ylabel('f_k - f_*');
title(['condition number:', num2str(condition)])
grid on;

