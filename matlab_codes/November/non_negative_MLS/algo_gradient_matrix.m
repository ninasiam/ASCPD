function [A_GD, f_GD] = algo_gradient_matrix(B, X, A_init, L, mu, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [A, f_GD] = gradient_MNLS(B, X, A_init, lambda, tol_inner)               %
%                                                                           %
% Problem: min 1/2 * ||X - A * B^T||                                        %  
%               s.t. A >= 0                                                 %
%                                                                           %
% Algorithm: gradient                                                       %
%                                                                           %
% Inputs: Matrices B, A_init, and X                                         %
% Output: Optimal A                                                         %
%                                                                           %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gram = B' * B;
X_B_prod = X * B;
 
A = A_init;
f_val(1) = (1/2)*norm(X - A_init*B','fro')^2;
iters = 1;
while (1)
    
    grad_A = - X_B_prod + A*Gram;

    if(iters > maxiters)
        break
    else
        new_A = max(0, A - 1/L * grad_A);
        f_val(iters+1) = (1/2)*norm(X - new_A*B','fro')^2;
    end
    A = new_A;
    iters = iters + 1;
end

A_GD = A;
f_GD = f_val;
