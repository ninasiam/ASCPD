function [x_SGD, f_SGD] = algo_SGD_accel(A, b, tol, f_star, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [x_SGD, f_SGD] = algo_SGD(A, b, tol, maxiters)       %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(A);

ATA = A' * A;

L = max(eig(ATA))/m;
mu = min(eig(ATA))/m;
q = mu/L;

b_par = (1-sqrt(q))/(1+sqrt(q));
% Initialization 
x_SGD = zeros(n,1);
y_SGD = x_SGD;
f_SGD(1) = 1/(2*m) * norm(b)^2;

iter = 1;
step_size = 1/(L*m*n);%(4)/(L*n);

while (1)
   
   ind = randi(m,1,1);
   grad_f_SGD = ( A(ind,:) * y_SGD - b(ind) ) * A(ind,:)';
   
   new_x_SGD = y_SGD - step_size * grad_f_SGD;
   
   new_y_SGD = new_x_SGD + b_par * (new_x_SGD - x_SGD);
   
   if (mod(iter, m)==0) 
      f_SGD = [f_SGD 1/(2*m) * norm(A * new_x_SGD - b)^2];
   end
   
   if ( (f_SGD(end,1)-f_star )  < tol  || iter > maxiters)
       x_SGD = new_x_SGD;
       y_SGD = new_y_SGD;
       break;
   end
   
   x_SGD = new_x_SGD;
   y_SGD = new_y_SGD;
   iter = iter + 1;
    
end