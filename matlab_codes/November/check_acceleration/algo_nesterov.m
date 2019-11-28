function [x_N, f_N] = algo_nesterov(A, b, tol, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [x_N, f_N] = algo_nesterov(A, b, tol, maxiters)       %
%                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(A);

x_star = pinv(A) * b;
f_star = 1/(2*m) * norm(A*x_star-b)^2;

ATA = A' * A;
ATb = A' * b;

eig_ATA = 1/m * eig(ATA);
L = max(eig_ATA);
mu = min(eig_ATA);
q = mu/L;

% Initialization 
x_N = zeros(n,1);
y_N = zeros(n,1);
f_N(1) = 1/(2*m) * norm(b)^2;
alpha = 1;
b_par = (1-sqrt(q))/(1+sqrt(q));
iter = 1;
while (1)
   
    grad_f_N = (1/m) * (ATA * y_N - ATb);
    new_x_N = y_N - (1/L) * grad_f_N;
    %new_alpha = update_alpha(alpha, q);
    %beta = alpha * (1 - alpha) / ( alpha^2 + new_alpha );
    
    new_y_N = new_x_N + b_par * (new_x_N - x_N);
    
    f_N(iter+1,1) = 1/(2*m) * norm(A * new_x_N - b)^2;
   
   if ( (f_N(iter+1,1)-f_star )  < tol  || iter > maxiters)
       x_N = new_x_N;
       break;
   end
   
   x_N = new_x_N;
   y_N = new_y_N;
   %alpha = new_alpha;

   iter = iter + 1;
    
end