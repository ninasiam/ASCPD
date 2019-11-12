clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 100;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 1;
lambda_max = 10;
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

%Problem parameters
Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;

A_t_b = A'*b;
%initial point
x_init_all = zeros(n,1);

%Optimal
x_star = inv(Hessian)*A_t_b;

f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);

x_init_all = zeros(n,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%           For GOOD CONDITION NUMBER

tic();
iter = 1;
x_mini(:,iter) = x_init_all;
f_val_mini_init = (1/(2*m))*norm(A*x_mini(:,iter) - b)^2;
f_val_mini(iter) = f_val_mini_init;

while(1)
    
    if iter == 1
        full_grad = (1/m)*A'*(A*x_mini(:,iter) - b);
        eta = 1/L;
        grad_f_mini = full_grad;
    else
    
        while 1
            indx = randperm(m, batch_s);
            grad_f_mini = A(indx,:)'*(A(indx,:)*x_mini(:,iter) - b(indx));
            costheta = dot(grad_f_mini,0.4*full_grad)/(norm(grad_f_mini)*norm(full_grad))
            if costheta > 0% && costheta < sqrt(2)/2
                break;
            end
        end        
    end
    
    x_mini(:,iter + 1) = x_mini(:,iter) - eta*grad_f_mini;
    
    if (mod(iter, n)==0) 
      f_val_mini = [f_val_mini 1/(2*m) * norm(A * x_mini(:,iter + 1) - b)^2];
    end
    
    if ( (f_val_mini(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER)
       break;
    end
    
    if iter == 1 
        batch_s = 1;
        eta = 0.1/batch_s;
    end
    
    iter = iter + 1;
end

fprintf('Mini-batch');
time_mini = toc

figure(1)
semilogy(f_val_mini - f_star2);

%xlim([0 10^4])
ylim([10^(-5) 10^(4)])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
xlabel('epochs');
ylabel('f_k - f_*');
title(['Using 1/m condition number:', num2str(condition)])
grid on;
