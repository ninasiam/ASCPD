function [A_SGD, f_SGD] = algo_GD_matrix(B, X, A_init, L, mu, maxiters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [A, f_GD] = SGD_MNLS(B, X, A_init, L, mu, maxiters)                       %
%                                                                           %
% Problem: min 1/2 * ||X - A * B^T||                                        %  
%               s.t. A >= 0                                                 %
%                                                                           %
% Algorithm: SGD                                                            %
%                                                                           %
% Inputs: Matrices B, A_init, and X                                         %
% Output: Optimal A                                                         %
%                                                                           %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n] = size(X);
 
A = A_init;
fval(1) = (1/2)*norm(X - A_init*B','fro')^2;
iters = 1;
ii = 2;
while (1)
    
    idx = randi(n,1);
    grad_A = - X(:,idx)*B(idx,:) + A*(B(idx,:)' * B(idx,:));

    if(iters > maxiters)
        break
    else
        new_A = max(0, A - 1/L * grad_A);
        if(mod(iters,n) == 0)   %epoch is defined on dataset
            fval(ii) = (1/2)*norm(X - new_A*B','fro')^2;
            ii = ii + 1;
        end
    end
    A = new_A;
    iters = iters + 1;
end

A_SGD = A;
f_SGD = fval;
