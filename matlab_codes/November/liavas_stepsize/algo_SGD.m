function [x_SGD, f_SGD] = algo_SGD(A, b, tol, f_star, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [x_SGD, f_SGD] = algo_SGD(A, b, tol, maxiters)       %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(A);

ATA = A' * A;

L = max(eig(ATA));              %here no division with m
c =  min(eig(ATA));
theta = 1/(2*c) + 1;

% Initialization 
x_SGD = zeros(n,1);
f_SGD(1) = 1/(2*m) * norm(b)^2;

iter = 1;
step_size = (4*m)/(L*n)

while (1)
   
   ind = randi(m,1,1);
   grad_f_SGD = ( A(ind,:) * x_SGD - b(ind) ) * A(ind,:)';
   
   new_x_SGD = x_SGD - (4*m)/(L * n) * grad_f_SGD;
   %new_x_SGD = x_SGD -  0.1/sqrt(iter) * grad_f_SGD;
   %new_x_SGD = x_SGD -  300/iter * grad_f_SGD;
   
   if (mod(iter, n)==0) 
      f_SGD = [f_SGD 1/(2*m) * norm(A * new_x_SGD - b)^2];
   end
   
   if ( (f_SGD(end,1)-f_star )  / f_star < tol  || iter > maxiters)
       x_SGD = new_x_SGD;
       break;
   end
   
   x_SGD = new_x_SGD;
   
   iter = iter + 1;
    
end