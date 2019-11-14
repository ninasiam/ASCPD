clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 1;

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
%b = rand(m,1);
%or
b = A*randn(n,1) + 0.01*randn(m,1);

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
f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);

%%  Stochastic gradient descent 
tic();
iter = 1;
x_sgd(:,iter) = x_init_all;
f_val_sgd_init = (1/(2*m))*norm(A*x_sgd(:,iter) - b)^2;
f_val_sgd(iter) = f_val_sgd_init;

eta_sgd = 100/L;
while (1)
   
   grad = (1/m)*A'*(A*x_sgd(:,iter) - b);
   ind = randi(m,1,1);
   res = (A(ind,:) * x_sgd(:,iter) - b(ind));
   grad_f_SGD = res* A(ind,:)';
   error_sgd(iter) = res.^2;
   
   costheta = dot(grad,grad_f_SGD)/(norm(grad_f_SGD)*norm(grad))
   x_sgd(:,iter + 1) = x_sgd(:,iter) - eta_sgd*grad_f_SGD;
   %new_x_SGD = x_SGD -  (theta/iter) * grad_f_SGD;
   if (mod(iter, n)==0) 
      f_val_sgd = [f_val_sgd 1/(2*m) * norm(A * x_sgd(:,iter + 1) - b)^2];
   end
   if ( (f_val_sgd(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER)
       break;
   end

   iter = iter + 1;
    
end
fprintf('SGD');
time_sgd = toc
time_per_iter = time_sgd/iter