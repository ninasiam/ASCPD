%testing stochastic algorithms and acceleration
clear, clc, clf

% Define dimensions and condition number of LS problems
m = 1000;
n = 100;

tmp = randn(m,n);
[U_tmp,S_tmp,V_tmp]=svd(tmp, 'econ');

min_s_A = 1; max_s_A = 10;
S_A = diag([min_s_A; max_s_A; min_s_A+max_s_A*rand(n-2,1)]);

A = U_tmp * S_A * V_tmp'; 
x = randn(n,1); 
b = A*x + 0.0001 * randn(m,1); 

% Optimal solution
x_star = pinv(A) * b;
f_star = 1/(2*m) * norm(A * x_star - b)^2;

tol = 10^(-4);
maxiters = 1000;

fprintf('\nSolution using gradient...')
[x_GD, f_GD] = algo_gradient(A, b, tol, maxiters);

fprintf('\nSolution using Nesterov...')
[x_N, f_N] = algo_nesterov(A, b, tol, maxiters);

fprintf('\nSolution using plain SGD...')
[x_SGD, f_SGD] = algo_SGD(A, b, tol, f_star, maxiters*m);

fprintf('\nSolution using plain SGD with acceleration...')
[x_SGD_accel, f_SGD_accel] = algo_SGD_accel(A, b, tol, f_star, maxiters*m);

fprintf('\nSolution using mini-batch SGD...')
size_batch = 10;
[x_SGD_mb, f_SGD_mb] = algo_SGD_mini_batch(A, b, size_batch, tol, f_star, maxiters*m/size_batch);

fprintf('\nSolution using mini-batch SGD acceleration...')
size_batch = 10;
[x_SGD_mb_accel, f_SGD_mb_accel] = algo_SGD_mini_batch_accel(A, b, size_batch, tol, f_star, maxiters*m/size_batch);

figure(1)
semilogy((f_GD-f_star),'c')
hold on
semilogy((f_N-f_star), 'r'),
hold on;
semilogy( (f_SGD-f_star),  'g')
hold on 
semilogy(f_SGD_accel - f_star, 'k', 'Linewidth', 5);
hold on;
semilogy( (f_SGD_mb-f_star),  'm')
hold on;
semilogy( (f_SGD_mb_accel-f_star),  'y')
hold off;
legend('gradient','nesterov','sgd','sgd accel','mini-batch sgd', 'mini-batch sgd accel')
grid on;
fprintf('\n')
