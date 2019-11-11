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

A_t_b = A'*b;
%initial point
x_init_all = zeros(n,1);

%Optimal
x_star = inv(Hessian)*A_t_b;
f_star1 = (1/(2))*norm(A*x_star - b)^2;
f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              without 1/m

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
    
    if((f_val_gd(iter+1) - f_star1 < ep) ||  iter > MAX_OUTER_ITER)% crit_gd < ep || iter > MAX_OUTER_ITER)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          with 1/m

tic();
x_gd_w = [];
x_gd_w(:,1) = x_init_all;


iter = 1;
f_val_gd_init_w = (1/(2*m))*norm(A*x_gd_w(:,1) - b)^2;
f_val_gd_w(iter) = f_val_gd_init_w;

eta_gd = m/(L);

while(1)

    grad_f = (1/m)*(Hessian*x_gd_w(:,iter) - A_t_b);
    x_gd_w(:,iter+1) = x_gd_w(:,iter) - eta_gd*grad_f;
    
    f_val_gd_w(iter+1) = (1/(2*m))*norm(A*x_gd_w(:,iter+1) - b)^2;
    
    crit_gd_w = norm(x_gd_w(:,iter+1) - x_gd_w(:,iter))/norm(x_gd_w(:,iter));
    
    if((f_val_gd_w(iter+1) - f_star2 < ep) ||  iter > MAX_OUTER_ITER)% crit_gd < ep || iter > MAX_OUTER_ITER)
        break;
    end
    
    iter = iter + 1;
        
end

fprintf('Gradient');
time_gd_w = toc 


%%Accelerated Gradient Descend
tic();
iter = 1;
x_nes_w(:,iter) = x_init_all;
y_w(:,iter) = x_init_all;

f_val_nes_init_w = (1/(2*m))*norm(A*x_nes_w(:,iter) - b)^2;
f_val_nes_w(iter) = f_val_nes_init_w;

Q= mu/L;
b_par = (1-sqrt(Q))/(1+sqrt(Q));

%Nesterov main loop 

while(1)

    x_nes_w(:,iter+1) = y_w(:,iter) - (m/(L))*(1/m)*(Hessian*y_w(:,iter) - A_t_b);
    
    y_w(:,iter+1) = x_nes_w(:,iter+1) + b_par*(x_nes_w(:,iter+1) - x_nes_w(:,iter));

    f_val_nes_w(iter+1) = (1/(2*m))*norm(A*x_nes_w(:,iter+1) - b)^2;
    
    crit_nes_w = norm(x_nes_w(:,iter+1) - x_nes_w(:,iter))/norm(x_nes_w(:,iter));
    
    if((f_val_nes_w(iter+1) - f_star2 < ep) ||  iter > MAX_OUTER_ITER)%crit_nes < ep || iter > MAX_OUTER_ITER)%abs(f_val_nes(k) - f_star) < ep)
        break;
    end
    
    iter = iter + 1;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nesterov');
time_nes_w = toc 

bound_iter_gd = condition*log(1/ep)
bound_iter_nes = sqrt(condition)*log(1/ep)

figure(1)
semilogy(f_val_gd - f_star1);
hold on;
semilogy(f_val_nes - f_star1);
hold off
%xlim([0 10^4])
ylim([10^(-4) 10^6])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('Gradient', 'Nesterov');
xlabel('iterations');
ylabel('f_k - f_*');
title(['condition number:', num2str(condition)])
grid on;

figure(2)
semilogy(f_val_gd_w - f_star2);
hold on;
semilogy(f_val_nes_w - f_star2);
hold off
%xlim([0 10^4])
ylim([10^(-4) 10^(-1)])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('Gradient', 'Nesterov');
xlabel('iterations');
ylabel('f_k - f_*');
title(['Using 1/m condition number:', num2str(condition)])
grid on;