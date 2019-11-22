%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 10000;
tol = 10^(-4);
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

x_star = pinv(A) * b;
f_star = 1/(2*m) * norm(A*x_star-b)^2;
%Problem parameters
Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;

size_batch = 10;
[x_SGD, f_SGD_mini] = algo_SGD_mini_batch(A, b, size_batch, tol, f_star, MAX_OUTER_ITER);


[x_SGD, f_SGD] = algo_SGD(A, b, tol, f_star, MAX_OUTER_ITER);


figure(1)
semilogy(f_SGD - f_star);
hold on;
semilogy(f_SGD_mini - f_star);
hold off
%xlim([0 10^4])
ylim([10^(-5) 10^(4)])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('SGD','Mini batch');
xlabel('epochs');
ylabel('f_k - f_*');
title(['Using 1/m condition number:', num2str(condition)])
grid on;