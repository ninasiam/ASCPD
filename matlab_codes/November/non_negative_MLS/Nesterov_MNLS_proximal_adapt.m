function [H, iters, f_val] = Nesterov_MNLS_proximal_adapt(W, X_AT, H_init, tol_inner)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [H, iters] = Nesterov_MNLS_proximal(W, X_AT, H_init, lambda, tol_inner)   %
%                                                                           %
% Problem: min 1/2 * ||W * H - X_AT|| + lambda/2 * ||H - H_init||^2         %  
%               s.t. H >= 0                                                 %
%                                                                           %
% Algorithm: Optimal Nesterov                                               %
%                                                                           %
% Inputs: Matrices W, X_AT, and H_init                                      %
% Output: Optimal H                                                         %
%                                                                           %
% A. P. Liavas, May 4, 2016                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z = W' * W;
w = W' * X_AT;
 
s = svd(Z);
L = max(s);
mu = min(s);

q = L/mu;

% if (q < 10^4) 
%     lambda = 10^(-1.5);
% elseif (q < 10^6)
%     lambda = 10^(-1);
% else 
%     lambda = 10^0;
% end
% 
% Z = Z + lambda * eye(F);
% L = L + lambda;
% mu = mu + lambda;

H = H_init;
Y = H;
f_val(1) = (1/2)*norm(W*H_init - X_AT,'fro')^2;
alpha = 1;
iters = 1;
while (1)
    grad_Y = Z * Y - w;% - lambda * H_init;

    if(iters > 1000)
        break
    else
        new_H = max(0, Y - 1/L * grad_Y);
        new_alpha = update_alpha(alpha, q);
        beta = alpha * (1 - alpha) / ( alpha^2 + new_alpha );
        Y = new_H + beta * (new_H - H);
        f_val(iters+1) = (1/2)*norm(W*new_H - X_AT,'fro')^2;
    end
    H = new_H;
    alpha = new_alpha;
    iters = iters + 1;
end
