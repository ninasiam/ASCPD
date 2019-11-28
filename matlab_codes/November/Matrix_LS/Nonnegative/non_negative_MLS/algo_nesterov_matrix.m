function [A_NAG, f_NAG] = algo_nesterov_matrix(B, X, A_init, L, mu, max_iters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [A_NAG, f_NAG] = Nesterov_MNLS_proximal(B, X, A_init, max_iters)          %
%                                                                           %
% Problem: min 1/2 * ||X - ABT||                                            %  
%               s.t. A >= 0                                                 %
%                                                                           %
% Algorithm: Optimal Nesterov                                               %
%                                                                           %
% Inputs: Matrices B, X, and A_init                                         %
% Output: Optimal A                                                         %
%                                                                           %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gram = B' * B;
X_B_prod = X * B;

A = A_init;
Y = A;
f_val(1) = (1/2)*norm(X - A_init*B','fro')^2;

q = L/mu;

beta = (1-sqrt(q))/(1+sqrt(q));
iters = 1;

while (1)
    
    grad_Y = -X_B_prod + Y*Gram;

    if(iters > max_iters)
        break
    else
        new_A = max(0, Y - (1/L) * grad_Y);
        Y = new_A + beta * (new_A - A);
        f_val(iters+1) = (1/2)*norm(X - new_A*B','fro')^2;
    end
    A = new_A;
    iters = iters + 1;
end

A_NAG = A;
f_NAG = f_val;
