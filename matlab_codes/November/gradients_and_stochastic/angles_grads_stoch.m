clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 200;
n = 10;
MAX_OUTER_ITER = 1;

%creation of matrix A with specific eigenvalues
rng('default');
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
f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);

%%  Stochastic gradient descent 
tic();
iter = 1;
x_sgd(:,iter) = x_init_all;


eta_sgd = 100/L;

while (iter < 10)
   
   grad = (1/m)*A'*(A*x_sgd(:,iter) - b);
   
   for ii = 1 : m
       res = (A(ii,:) * x_sgd(:,iter) - b(ii));
       grad_f_SGD = res* A(ii,:)';
       costheta(ii,iter) = dot(grad,grad_f_SGD)/(norm(grad_f_SGD)*norm(grad));
       
   end
%    
%    signed_costheta = sign( costheta(:,iter) );
%    positive_angles = sum(signed_costheta>0);
%    negative_angles = m - positive_angles;
%    
%    prob_positive = positive_angles/m;
%    prob_negative = 1 - prob_positive;
%    
   figure(1)
   hist(costheta(:,iter))
   pause;
   x_sgd(:,iter + 1) = x_sgd(:,iter) - eta_sgd*grad;

   iter = iter + 1;
    
end
fprintf('SGD');
time_sgd = toc
time_per_iter = time_sgd/iter