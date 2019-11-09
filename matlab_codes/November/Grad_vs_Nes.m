clc, clear all, close all

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 5000;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 10;
lambda_max = 1000;
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
x_init_all = randn(n,1);

%Optimal
x_star = inv(Hessian)*A'*b;
f_star = (1/(2))*norm(A*x_star - b)^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ep = 10^(-4);

%closed form solution
x_star = inv((A'*A))*A'*b;
f_star = (1/(2))*norm(A*x_star - b)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          NON STOCHASTIC METHODS

tic();
x_gd = [];
x_gd(:,1) = x_init_all;


iter = 1;
f_val_gd_init = (1/(2))*norm(A*x_gd(:,1) - b)^2;
f_val_gd(iter) = f_val_gd_init;

eta_gd = 1/(L);

while(1)

    grad_f = Hessian*x_gd(:,iter) - A_t_b;
    x_gd(:,iter+1) = x_gd(:,iter) - eta_gd*grad_f;
    
    f_val_gd(iter+1) = (1/(2))*norm(A*x_gd(:,iter+1) - b)^2;
    
    crit_gd = norm(x_gd(:,iter+1) - x_gd(:,iter))/norm(x_gd(:,iter));
    
    if( crit_gd < ep || iter > MAX_OUTER_ITER)
        break;
    end
    
    iter = iter + 1;
        
end

fprintf('Gradient');
time_gd = toc 


%%Accelerated Gradient Descend
tic();
iter = 1;
x_nes(:,iter) = x_init_all;
y(:,iter) = x_init_all;

f_val_nes_init = (1/(2))*norm(A*x_nes(:,iter) - b)^2;

Q= mu/L;
b_par = (1-sqrt(Q))/(1+sqrt(Q));

%Nesterov main loop 

while(1)

    x_nes(:,iter+1) = y(:,iter) - (1/(L))*(Hessian*y(:,iter) - A_t_b);
    
    y(:,iter+1) = x_nes(:,iter+1) + b_par*(x_nes(:,iter+1) - x_nes(:,iter));

    f_val_nes(iter+1) = (1/(2))*norm(A*x_nes(:,iter+1) - b)^2;
    
    crit_nes = norm(x_nes(:,iter+1) - x_nes(:,iter))/norm(x_nes(:,iter));
    
    if(crit_nes < ep || iter > MAX_OUTER_ITER)%abs(f_val_nes(k) - f_star) < ep)
        break;
    end
    
    iter = iter + 1;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nesterov');
time_gd = toc 



figure(1)
semilogy(f_val_gd - f_star);
hold on;
semilogy(f_val_nes - f_star);
hold off;
legend('Gradient', 'Nesterov');
grid on;