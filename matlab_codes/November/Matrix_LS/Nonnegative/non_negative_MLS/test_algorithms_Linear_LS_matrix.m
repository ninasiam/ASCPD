clear, clc, clf

% Define dimensions and condition number of LS problems
m = 200;
n = 2000;
r = 20;

A = rand(m,r);
B = rand(n,r);
X = A*B' + 0.001*randn(m,n);

Z = B' * B;
s = svd(Z);
L = max(s);
mu = min(s);

A_init = rand(m,r);
maxiters = 1000;

fprintf('\nSolution using gradient...')
[A_GD, f_GD] = algo_gradient_matrix(B, X, A_init, L, mu, maxiters);

fprintf('\nSolution using Nesterov...')
[A_N, f_N] = algo_nesterov_matrix(B, X, A_init, L, mu, maxiters);

fprintf('\nSolution using Nesterov 2...')
[A_N2, ~, f_N2] = Nesterov_MNLS_proximal_adapt(B, X', A_init', 10^(-8));

fprintf('\nSolution using plain SGD...')
[A_SGD, f_SGD] = algo_SGD_matrix(B, X, A_init, L, mu, maxiters*n);

fprintf('\nSolution using mini-batch SGD...')
batch_s = 50;
[A_SGD_mb, f_SGD_mb] = algo_SGD_mini_batch_matrix(B, X, A_init, L, mu, batch_s, maxiters*n/batch_s);

figure(1)
semilogy((f_GD))
hold on
semilogy((f_N), 'r'),
hold on;
semilogy((f_N2), 'y'),
hold on;
semilogy( (f_SGD(1:1:end)),  'g')
hold on;
semilogy( (f_SGD_mb(1:1:end)),  'm')
hold off;
legend('gradient','nesterov','nesterov2','sgd','mini-batch sgd')
grid on;
ylabel('f_val');
xlabel('epochs');
fprintf('\n')
