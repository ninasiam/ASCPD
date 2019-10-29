%Stochastic optimization
%Set 9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%9.1
clc, clear, close all

MAX_OUTER_ITER = 3000;
MAX_INNER_ITER = 3000;

n = 100;
m = 200;

A = randn(m,n);
b = randn(m,1);

% cvx_begin
%     cvx_precision('best');
%     variable x_cvx(n,1)
%     minimize ((1/2)*(A*x_cvx-b)'*(A*x_cvx-b));
% cvx_end

%closed form solution
x_star = inv((A'*A))*A'*b;
f_star = (1/2)*norm(A*x_star - b)^2;

%Gradient descent with constant step size
tic();
x_gd = [];
x_gd(:,1) = zeros(n,1);

f_val = [];

eta = 1/(2*max(svd(A'*A)));
epsilon = 10^(-5);

iter = 1;
while(1)

    grad_f = A'*(A*x_gd(:,iter) - b);
    x_gd(:,iter+1) = x_gd(:,iter) - eta*grad_f;
    
    f_val(:,iter+1) = (1/2)*norm(A*x_gd(:,iter+1) - b)^2;
    
    iter = iter + 1;
    
    if(norm(grad_f) < epsilon)
        break;
    end
end
% display('GD');
fprintf('GD \t')
toc()
figure(1);
semilogy(abs(f_val(1:1:iter) - f_star));
hold on;

%Stochastic gradient descent type I
tic();
x_sd = [];
x_sd(:,1) = zeros(n,1);

f_val_sd = [];
epsilon = 10^(-3);

iter = 1;

while(1)
    
    r = randi(m,1);
    
    stoch_grad = (A(r,:)*x_sd(:,iter) - b(r))*A(r,:)';
    eta_sd = 1/(2*max(svd(A(r,:)'*A(r,:))));
    x_sd(:,iter+1) = x_sd(:,iter) - eta_sd*stoch_grad;
    
    f_val_sd(:,iter+1) = (1/2)*norm(A*x_sd(:,iter+1) - b)^2;
    
    if(norm(f_val_sd(:,iter+1) - f_star) < epsilon || iter > MAX_INNER_ITER )
        break;
    end
    
    iter = iter + 1;
end

fprintf('SD Type I\t')
toc();

semilogy(abs(f_val_sd(1:1:iter) - f_star));
hold on;

% %Stochastic gradient descent type II
% 
% tic();
% x_sd2 = [];
% x_sd2(:,1) = zeros(n,1);
% 
% 
% x_sd_prev = zeros(n,1);
% x_sd_tmp = zeros(n,1);
% 
% f_val_sd2 = [];
% epsilon = 10^(-3);
% 
% iter = 1;
% 
% while(1)
%     
%     x_sd_prev = x_sd2(:,iter);
%     
%     batch = randperm(n);
%     for i=1:size(batch,2)
%        stoch_grad2 = (A(batch(i),:)*x_sd_prev - b(batch(i)))*A(batch(i),:)';
%        eta_sd2 = 0.0001%1/(2*max(svd(A(batch(i),:)'*A(batch(i),:))));
%        x_sd_tmp = x_sd_prev - (eta_sd2)*stoch_grad2;
%        x_sd_prev = x_sd_tmp;
%     end
%     
%     x_sd2(:,iter+1) = x_sd_tmp;
%     f_val_sd2(:,iter+1) = (1/2)*norm(A*x_sd2(:,iter+1) - b)^2;
%     
%     if(norm(f_val_sd2(:,iter+1) - f_star) < epsilon || iter > MAX_INNER_ITER)
%         break;
%     end
%     
%     iter = iter + 1;
% end
% fprintf('SD Type II\t')
% toc();
% semilogy(abs(f_val_sd2(1:1:iter) - f_star));
% hold on;

%Adagrad
tic();
x_ad = [];
x_ad(:,1) = zeros(n,1);

f_val_ad = [];
epsilon = 10^(-5);
stoch_agrad = [];
eta = 200/max(svd(A'*A));
iter = 1;
outer_prod_t = zeros(n);
stoch_agrad_prod = [];

while(1)
    
    r = randi(m,1);
   
    stoch_agrad = (A(r,:)*x_ad(:,iter) - b(r))*A(r,:)';
    stoch_agrad_prod = stoch_agrad*stoch_agrad';
    
    %updated sum of outer products of the chosen gradient
    outer_prod_t = outer_prod_t + stoch_agrad_prod;
    
    %t is the current iteration
    B_t = sqrt(diag(outer_prod_t)) + 10^(-10);
    
    x_ad(:,iter+1) = x_ad(:,iter) - eta^(1/1)*(stoch_agrad./B_t);
    
    f_val_ad(:,iter+1) = (1/2)*norm(A*x_ad(:,iter+1) - b)^2;
    
    if(norm(x_ad(:,iter+1) - x_ad(:,iter)) < epsilon && iter > MAX_INNER_ITER )
        break;
    end
    
    iter = iter + 1;
end
fprintf('AdaGrad\t')
toc();
semilogy(abs(f_val_ad(1:1:iter) - f_star));
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%9.2

%compute condition number
cond_n = cond(A'*A);

%SVRG
tic();
iter = 1;
n_par = 2*m; 


x_tilda_prev = ones(n,1); 

f_val_svrg = [];
eta = 1/(2*max(svd(A'*A)));

while(1)
    
    if(iter > MAX_OUTER_ITER)
        break;
    end
    
    x_tilda = x_tilda_prev;
    
    mu = 1/m*(A'*(A*x_tilda - b)); 
    
    x_prev = x_tilda;
    
    t = 1;
    while(t < n_par)
        %loop that changes x
        i_t = randi(m,1);
        
        x = x_prev - eta*((A(i_t,:)*x_prev - b(i_t)).*A(i_t,:)' - (A(i_t,:)*x_tilda - b(i_t)).*A(i_t,:)' + mu); 
        
        x_prev = x;
        
        t = t + 1;
    end
    %option I
    
    x_tilda_prev = x;
    fval_x = (1/2)*norm(A*x - b)^2;  
    f_val_svrg = [f_val_svrg fval_x];
    iter = iter + 1;
    
end
fprintf('SVRG\t')
toc();
semilogy(abs(f_val_svrg - f_star));
title('Exer 9.2')
xlabel('iterations');
ylabel('|f_t - f^*|');
legend('GD','SGD-Type I','AdaGrad','SVRG');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%9.3
%2)
%Cyclic coordinate descent

x_ccd = zeros(n,1);

f_val_ccd = [];

iter = 1;

tic();
while(iter < MAX_OUTER_ITER)
    
    
    i = mod(iter,n) + 1;
    
    L_i = norm(A(:,i)'*A);
    
    grad_i = A(:,i)'*(A*x_ccd - b);
    
    x_ccd(i) = x_ccd(i) - (1/L_i)*grad_i;
    
    f_val_c = (1/2)*norm(A*x_ccd - b)^2;
    
    f_val_ccd = [f_val_ccd f_val_c];
    
    iter = iter + 1;
end
fprintf('CCD\t')
toc();

figure(2);
semilogy(abs(f_val - f_star));
hold on;
semilogy(abs(f_val_ccd - f_star));
hold on;

%3) 

x_urcd = zeros(n,1);

f_val_urcd = [];

iter = 1;

tic();
while(iter < MAX_OUTER_ITER)
    
    
    i = randi(n,1);
    
    L_i = norm(A(:,i)'*A);
    
    grad_i = A(:,i)'*(A*x_urcd - b);
    
    x_urcd(i) = x_urcd(i) - (1/L_i)*grad_i;
    
    f_val_ur = (1/2)*norm(A*x_urcd - b)^2;
    
    f_val_urcd = [f_val_urcd f_val_ur];
    
    iter = iter + 1;
end
fprintf('URCD\t')
toc();

semilogy(abs(f_val_urcd - f_star));
hold on;

%4)

x_nurcd = zeros(n,1);

f_val_nurcd = [];

iter = 1;

a = 1;

for(l=1:1:n)
    L(l) = norm(A(:,l)'*A);
end

sum_Li = sum(L.^(a));

p_a = (L.^(a))./sum_Li;

%cdf of p_a disrtibution
cdf = cumsum(p_a);

tic();
while(iter < MAX_OUTER_ITER)
    
    i = find(L == L(find(rand < cdf,1)));
    
    grad_i = A(:,i)'*(A*x_nurcd - b);
    
    x_nurcd(i) = x_nurcd(i) - (1/L(i))*grad_i;
    
    f_val_nur = (1/2)*norm(A*x_nurcd - b)^2;
    
    f_val_nurcd = [f_val_nurcd f_val_nur];
    
    iter = iter + 1;
end
fprintf('NURCD\t')
toc();

semilogy(abs(f_val_nurcd - f_star));
title('Exer 9.3');
legend('GD','CCD','Uniform Random CD','Non-Uniform Random CD');
xlabel('iterations');
ylabel('|f_t - f^*|');

