function [x_SGD, f_SGD] = algo_SGD_mini_batch(A, b, size_batch, tol, f_star, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [x_SGD, f_SGD] = algo_SGD_mini_batch(A, b, size, tol, maxiters)       %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m, n] = size(A);

ATA = A' * A;

L = max(eig(ATA))/m;
c =  min(eig(ATA));


% Initialization 
x_SGD = zeros(n,1);
f_SGD(1) = 1/(2*m) * norm(b)^2;

step_size = (4*size_batch)/(L*n);
iter = 1;
while (1)
   
   ind = randperm(m,size_batch);
   A_local = A(ind, :); b_local = b(ind);
   grad_f_SGD = 1/size_batch * A_local' *( A_local * x_SGD - b_local );
   
   new_x_SGD = x_SGD - step_size * grad_f_SGD;

   if (mod(iter, m/size_batch) == 0)
        f_SGD = [f_SGD; 1/(2*m) * norm(A * new_x_SGD - b)^2];
   end
   if ( (f_SGD(end,1)-f_star )  < tol  || iter > maxiters)
       x_SGD = new_x_SGD;
       break;
   end
   
   x_SGD = new_x_SGD;
   
   iter = iter + 1;
    
end