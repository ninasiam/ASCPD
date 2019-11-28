function [x_GD, f_GD] = algo_SGD(A, b, tol, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [x_GD, f_GD] = algo_gradient(A, b, tol, maxiters)       %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(A);

x_star = pinv(A) * b;
f_star = 1/(2*m) * norm(A*x_star-b)^2;

ATA = A' * A;
ATb = A' * b;

L = max(eig(ATA))/m;

% Initialization 
x_GD = zeros(n,1);
f_GD(1) = 1/(2*m) * norm(b)^2;

iter = 1;
while (1)
   
   grad_f_GD = 1/m * (ATA * x_GD - ATb);
   new_x_GD = x_GD - (1/L) * grad_f_GD;
   f_GD(iter+1,1) = 1/(2*m) * norm(A * new_x_GD - b)^2;
   
   if ( (f_GD(iter+1,1)-f_star ) < tol  || iter > maxiters)
       x_GD = new_x_GD;
       break;
   end
   
   x_GD = new_x_GD;
   
   iter = iter + 1;
    
end