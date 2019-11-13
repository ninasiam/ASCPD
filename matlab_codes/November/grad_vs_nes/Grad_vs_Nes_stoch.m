clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 1000;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 1;
lambda_max = 1000;
eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);
eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);
A = U*Sigma*V';

%Normalize the rows of A to unit norm 
%A = normalize(A);

%vector of parameters
%b = rand(m,1);
%or
b = A*randn(n,1) + 0.1*randn(m,1);

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Stochastic gradient descent
tic();
iter = 1;
x_sgd(:,iter) = x_init_all;
f_val_sgd_init_w = (1/(2*m))*norm(A*x_sgd(:,iter) - b)^2;
f_val_sgd_w(iter) = f_val_sgd_init_w;
cache = zeros(n,1);
while (1)
   
   ind = randi(m,1,1);
   grad_f_SGD = ( A(ind,:) * x_sgd(:,iter) - b(ind) ) * A(ind,:)';
   cache = cache + grad_f_SGD;
   x_sgd(:,iter + 1) = x_sgd(:,iter) - (1/(L+mu))*(1/iter)*cache;%2/(L+mu)*grad_f_SGD;
   %new_x_SGD = x_SGD -  (theta/iter) * grad_f_SGD;
   if (mod(iter, n)==0) 
      f_val_sgd_w = [f_val_sgd_w 1/(2*m) * norm(A * x_sgd(:,iter + 1) - b)^2];
   end
   if ( (f_val_sgd_w(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER)
       break;
   end
   
   iter = iter + 1;
    
end
fprintf('SGD');
time_sgd = toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic();
iter = 1;
x_mini(:,iter) = x_init_all;
f_val_mini_init = (1/(2*m))*norm(A*x_mini(:,iter) - b)^2;
f_val_mini(iter) = f_val_mini_init;

batch_s = 20;
while(1)
    
    batch = randperm(m, batch_s);
    grad_f_mini = A(batch,:)'*(A(batch,:)*x_mini(:,iter) - b(batch));
    
    
    x_mini(:,iter + 1) = x_mini(:,iter) - 2*batch_s/((L+mu))*grad_f_mini;
    if (mod(iter, n/batch_s)==0) 
      f_val_mini = [f_val_mini 1/(2*m) * norm(A * x_mini(:,iter + 1) - b)^2];
    end
    if ( (f_val_mini(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER/batch_s)
       break;
    end
    
    iter = iter + 1;
end

fprintf('Mini-batch');
time_mini = toc
bound_iter_gd = condition*log(1/ep)
bound_iter_nes = sqrt(condition)*log(1/ep)

figure(2)
semilogy(f_val_gd_w - f_star2);
hold on;
semilogy(f_val_nes_w - f_star2);
hold on;
semilogy(f_val_sgd_w - f_star2);
hold on;
semilogy(f_val_mini - f_star2);
hold off
%xlim([0 10^4])
ylim([10^(-5) 10^(4)])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('Gradient', 'Nesterov','SGD','Mini');
xlabel('iterations');
ylabel('f_k - f_*');
title(['Using 1/m condition number:', num2str(condition)])
grid on;