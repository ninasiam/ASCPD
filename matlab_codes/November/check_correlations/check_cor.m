clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 20;
n = 10;
MAX_OUTER_ITER = 100;

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

[a_i_a_j, a_jx_bj] = check_correlations(A, b);
a_i_sq = diag(a_i_a_j);
a_b_i_sq = diag(a_jx_bj);

figure(1)
stem(a_i_sq, 'rx');
hold on
stem(a_i_a_j - eye(m)*a_i_sq);
hold off;
grid on;

figure(2)
stem(a_b_i_sq, 'mx');
hold on
stem(a_jx_bj - eye(m)*a_b_i_sq);
hold off;
grid on;

min_values_a_i_a_j = min(a_i_a_j(1,:));
max_values_a_i_a_j = max(a_i_a_j(1,:));
figure(3)
edges = min_values_a_i_a_j:50:max_values_a_i_a_j;
h1 = histogram(a_i_a_j(1,:),edges)

