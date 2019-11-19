clc, clear all, close all

%%Check gradient and nesterov for 1/2||Ax - b||2^2 and  1/(2m)||Ax - b||2^2 

%creation of data
m = 2000;
n = 100;
MAX_OUTER_ITER = 5000;

%creation of matrix A with specific eigenvalues
A_tmp = randn(m,n);
[U,S,V] = svd(A_tmp,'econ');
lambda_min = 1;
lambda_max = 100;
eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);
eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);
A = U*Sigma*V';

%Normalize the rows of A to unit norm 
%A = normalize(A);

%vector of parameters
%b = rand(m,1);
%or
b = A*randn(n,1) + 0.01*randn(m,1);

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
f_star1 = (1/(2))*norm(A*x_star - b)^2;
f_star2 = (1/(2*m))*norm(A*x_star - b)^2;

ep = 10^(-4);

%%  Stochastic gradient descent 
tic();
cpu_t1 = cputime;
iter = 1;
x_sgd(:,iter) = x_init_all;
f_val_sgd_init = (1/(2*m))*norm(A*x_sgd(:,iter) - b)^2;
f_val_sgd(iter) = f_val_sgd_init;

eta_sgd = (4*m)/(L*n);
rng('default');
while (1)
   
   ind = randi(m,1,1);
   res = ( A(ind,:) * x_sgd(:,iter) - b(ind) );
   grad_f_SGD = res* A(ind,:)';
   error_sgd(iter) = res.^2;
   x_sgd(:,iter + 1) = x_sgd(:,iter) - eta_sgd*grad_f_SGD;
   %new_x_SGD = x_SGD -  (theta/iter) * grad_f_SGD;
   if (mod(iter, n)==0) 
      f_val_sgd = [f_val_sgd 1/(2*m) * norm(A * x_sgd(:,iter + 1) - b)^2];
   end
   if ( (f_val_sgd(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER)
       break;
   end

   iter = iter + 1;
    
end
fprintf('SGD');
time_sgd = toc
cpu_time_sgd = cputime - cpu_t1
time_per_iter = time_sgd/iter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Stochastic gradient descent with window
tic();
cpu_t2 = cputime;
iter = 1;
x_sgd_win(:,iter) = x_init_all;
f_val_sgd_init_win = (1/(2*m))*norm(A*x_sgd_win(:,iter) - b)^2;
f_val_sgd_win(iter) = f_val_sgd_init_win;

window_length = 1000;

eta_sgd_win_init = (4*m)/(L*n);
eta_sgd_win = eta_sgd_win_init;
scaling_parameter = 0.005; % 1 +/- scaling_parameter
alert_counter = 0;
rng('default');
while (1)
   
   ind = randi(m,1,1);
   residual = ( A(ind,:) * x_sgd_win(:,iter) - b(ind) ); 
   grad_f_SGD = residual* A(ind,:)';
   error(iter) = residual.^2;
   if iter > window_length
      [eta_sgd_win, alert_counter ] = check_error(error, window_length, eta_sgd_win, scaling_parameter, alert_counter);
   end
   
   x_sgd_win(:,iter + 1) = x_sgd_win(:,iter) - eta_sgd_win*grad_f_SGD;
   %new_x_SGD = x_SGD -  (theta/iter) * grad_f_SGD;
   if (mod(iter, n)==0) 
      f_val_sgd_win = [f_val_sgd_win 1/(2*m) * norm(A * x_sgd_win(:,iter + 1) - b)^2];
   end
   if ( (f_val_sgd_win(end)-f_star2 ) < ep || iter > n*MAX_OUTER_ITER)
       break;
   end
   
%    eta_sgd = 100/L;
   iter = iter + 1;
    
end
fprintf('SGD window');
time_sgd_win = toc
cpu_time_sgd_win = cputime - cpu_t2
time_per_iter_win = time_sgd_win/iter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic();
cpu_t3 = cputime;
iter = 1;
x_mini(:,iter) = x_init_all;
f_val_mini_init = (1/(2*m))*norm(A*x_mini(:,iter) - b)^2;
f_val_mini(iter) = f_val_mini_init;

batch_s = 10;
beta = 0.8;
alpha = 0.3;
t_init = (norm(mu*randn(n,1) - 2*A_t_b))/norm(randn(n,1)); %norm(A_t_b);%better than t_init = 1
% t = t_init/norm(randn(n,1));
while(1)
    
    t = t_init;
    batch = randperm(m, batch_s);
    grad_f_mini = (1/batch_s)*A(batch,:)'*(A(batch,:)*x_mini(:,iter) - b(batch));
    x_mini(:,iter + 1) = x_mini(:,iter) - t*grad_f_mini;
    
    f_fixed = 1/(2*batch_s) * norm(A(batch,:) * x_mini(:,iter) - b(batch))^2;
    norm_g__sq_fixed = norm(grad_f_mini)^2;
    while((1/(2*batch_s) * norm(A(batch,:) * x_mini(:,iter + 1) - b(batch))^2)  >  f_fixed - alpha*t*norm_g__sq_fixed)
        t = t*beta;%backtracking..
        x_mini(:, iter + 1) = x_mini(:, iter) - t*grad_f_mini; 
    end
    t
%     t_init = t;
    %x_mini(:, iter + 1) = x_mini(:, iter) - t*grad_f_mini; 
    if (mod(iter, n/batch_s)==0) 
      f_val_mini = [f_val_mini 1/(2*m)*norm(A * x_mini(:,iter + 1) - b)^2];
    end
    if ( (f_val_mini(end)-f_star2 )   < ep  || iter > n*MAX_OUTER_ITER/batch_s)
        break;
    end
    
    iter = iter + 1;
end

fprintf('Mini-batch');
time_mini = toc
cpu_time_mini = cputime - cpu_t3
figure(1)
semilogy(error)

figure(2)
semilogy(f_val_sgd - f_star2);
hold on;
semilogy(f_val_sgd_win - f_star2);
hold on;
semilogy(f_val_mini - f_star2);
hold off
%xlim([0 10^4])
ylim([10^(-5) 10^(4)])
% set(gca,'XTick',[bound_iter_nes bound_iter_gd],'XTickLabel',{'upper bound nesterov','upper bound gradient'})
legend('SGD','SGD window','Mini batch back tracking');
xlabel('epochs');
ylabel('f_k - f_*');
title(['Using 1/m condition number:', num2str(condition)])
grid on;