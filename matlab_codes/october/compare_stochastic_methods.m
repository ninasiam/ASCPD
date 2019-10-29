%Compare stochastic methods

%Stochastic Gradient
%       --> i)  rho-Lipschitz
%       --> ii) mu-strongly convex
%Nesterov Accelerated Gradient
%       --> i)  L-smooth
%       --> ii) L-smooth, mu-strongly convex
clc, clear all, close all

%creation of data
n = 500;

MAX_ITER = 10000;
epsilon = 10^(-8);

A_tmp = randn(n,n);
[U,S,V] = svd(A_tmp);

lambda_min = 10;
lambda_max = 100;

eigs = lambda_min + (lambda_max - lambda_min)*rand(n-2,1);

eig_A = [lambda_min; lambda_max; eigs];
Sigma = diag(eig_A);

A = U*Sigma*V';
b = randn(n,1);

Hessian = A'*A;
mu = min(svd(Hessian));
L = max(svd(Hessian));
condition = L/mu;

x_init_all = zeros(n,1);
%Optimal
x_star = (Hessian)\(A'*b);
f_star = (1/(2*n))*norm(A*x_star - b)^2;
Beta = norm(x_star);

%i) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz 
iter = 1;
x_sd(:,iter) = x_init_all;
norm_sd(iter) =  norm(x_sd(:,iter));
f_val_sd(:,iter) = (1/(2*n))*norm(A*x_init_all - b)^2;

while(1)

    r = randi(n,1);
    stoch_grad = (A(r,:)*x_sd(:,iter) - b(r))*A(r,:)';

    %eta_sd = Beta/(sqrt(iter)*L);
    %eta_sd = 1/(sqrt(iter)*(max(svd(A(r,:)'*A(r,:)))));
    eta_sd = Beta/(L*Beta + norm(A'*b));
    x_sd(:,iter+1) = x_sd(:,iter) - eta_sd*stoch_grad;
    norm_sd(iter+1) =  norm(x_sd(:,iter+1));
    f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;
    
    crit_sg = norm(x_sd(:,iter+1)-x_sd(:,iter))/norm(x_sd(:,iter));
    
    if(crit_sg < epsilon|| iter > MAX_ITER) 
        break;
    end

    iter = iter + 1;
end

%ii) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz, mu strongly
%convex

iter_sd_sc = 1;
x_sd_sc(:,iter_sd_sc) = x_init_all;
norm_sd_sc(iter_sd_sc) =  norm(x_sd_sc(:,iter_sd_sc));
f_val_sd_sc(:,iter_sd_sc) = (1/(2*n))*norm(A*x_init_all - b)^2;

while(1)

    r = randi(n,1);
    stoch_grad = (A(r,:)*x_sd_sc(:,iter_sd_sc) - b(r))*A(r,:)';

    eta_sd_sc = 1/(iter_sd_sc*mu);

    x_sd_sc(:,iter_sd_sc+1) = x_sd_sc(:,iter_sd_sc) - eta_sd_sc*stoch_grad;
    norm_sd_sc(iter_sd_sc + 1) =  norm(x_sd_sc(:,iter_sd_sc+1));
    f_val_sd_sc(:,iter_sd_sc+1) = (1/(2*n))*norm(A*x_sd_sc(:,iter_sd_sc+1) - b)^2;
    
    crit_sg_sc = norm(x_sd_sc(:,iter_sd_sc+1)-x_sd_sc(:,iter_sd_sc))/norm(x_sd_sc(:,iter_sd_sc));
    
    if(crit_sg_sc < epsilon|| iter_sd_sc > MAX_ITER) 
        break;
    end

    iter_sd_sc = iter_sd_sc + 1;
end

%i) Solve minf(x) = (1/2n)||Ax - b||_2, f - rho-Lipschitz via ASGD
iter_snes = 1;
alpha_par = 1;
beta_par = 1;
x_snes(:,iter_snes) = x_init_all;
norm_snes(iter_snes) = norm(x_snes(:,iter_snes));
f_val_snes(iter_snes) = (1/(2*n))*norm(A*x_snes(:,iter_snes) - b)^2;
y_snes(:,iter_snes) = x_snes(:,iter_snes);

q = 0;
while(1)
  
    r = randi(n,1);
    %eta_nes = Beta/(sqrt(iter_snes)*L);
    eta_nes = 1/(sqrt(iter_snes)*L);
    x_snes(:,iter_snes+1) = y_snes(:,iter_snes) - eta_nes*(A(r,:)*y_snes(:,iter_snes) - b(r))*A(r,:)'; 
    norm_snes(iter_snes+1) = norm(x_snes(:,iter_snes+1));
    
    a_poly = 1;
    b_poly = alpha_par(iter_snes)^2 - q;
    c_poly = -alpha_par(iter_snes)^2;
    D = b_poly^2 - 4 * a_poly * c_poly;

    alpha_par(iter_snes+1) = (-b_poly+sqrt(D))/2;

    beta_par(iter_snes+1) = (alpha_par(iter_snes)*(1 - alpha_par(iter_snes)))/(alpha_par(iter_snes)^2 + alpha_par(iter_snes+1));

    y_snes(:,iter_snes+1) = x_snes(:,iter_snes+1) + beta_par(iter_snes+1)*(x_snes(:,iter_snes+1) - x_snes(:,iter_snes));

    f_val_snes(iter_snes+1) = (1/(2*n))*norm(A*x_snes(:,iter_snes+1) - b)^2;

    crit_snes = norm(x_snes(:,iter_snes + 1) - x_snes(:,iter_snes))/norm(x_snes(:,iter_snes));
    if(crit_snes < epsilon || iter_snes > MAX_ITER)
        break;
    end
    
    iter_snes = iter_snes + 1;

end

%ii) Solve minf(x) = (1/2n)||Ax - b||_2, f - L-smoothness, mu - strongly convex via ASGD
iter_snes_sc = 1;
x_snes_sc(:,iter_snes_sc) = x_init_all;
norm_snes_sc(iter_snes_sc) = norm(x_snes_sc(:,iter_snes_sc));
f_val_snes_sc(iter_snes_sc) = (1/(2*n))*norm(A*x_snes_sc(:,iter_snes_sc) - b)^2;
y_snes_sc(:,iter_snes_sc) = x_snes_sc(:,iter_snes_sc);

beta_star = (1-sqrt(mu/L))/(1+sqrt(mu/L));

while(1)
    
    r = randi(n,1);
    eta_snes_sc = Beta/(iter_snes_sc*L);
    
    x_snes_sc(:,iter_snes_sc + 1 ) = y_snes_sc(:,iter_snes_sc) - eta_snes_sc*(A(r,:)*y_snes_sc(:,iter_snes_sc) - b(r))*A(r,:)';
    norm_snes_sc(iter_snes_sc + 1) = norm(x_snes_sc(:,iter_snes_sc +1));
    y_snes_sc(:,iter_snes_sc+1) = x_snes_sc(:,iter_snes_sc + 1 ) + beta_star*(x_snes_sc(:,iter_snes_sc + 1 ) - x_snes_sc(:,iter_snes_sc));
    
    f_val_snes_sc(iter_snes_sc + 1) = (1/(2*n))*norm(A*x_snes_sc(:,iter_snes_sc + 1) - b)^2;
    
    crit_snes_sc = norm(x_snes_sc(:,iter_snes_sc + 1) - x_snes_sc(:,iter_snes_sc))/norm(x_snes_sc(:,iter_snes_sc));
    if(crit_snes_sc < epsilon || iter_snes_sc > MAX_ITER)
        break;
    end
    
    iter_snes_sc = iter_snes_sc + 1;
end
figure(1)
semilogy(abs(f_val_sd -f_star));
hold on;
semilogy(abs(f_val_sd_sc - f_star));
hold on;
% semilogy(abs(f_val_snes - f_star));
% hold on;
semilogy(abs(f_val_snes_sc - f_star));
grid on;
xlabel('iterations');
ylabel('|f_t - f^*|');
legend('SGD rho-Lip','SGD rho-Lip, mu-strong convex','AGD L-smooth', 'AGD L-smooth, mu-strongly convex');
hold off;

figure(2)
plot(norm_sd - Beta);
hold on;
plot(norm_sd_sc - Beta);
hold on;
% plot(norm_snes - Beta);
% hold on;
plot(norm_snes_sc - Beta, 'Linewidth',3);
grid on;
xlabel('iterations');
ylabel('$\|{\bf x }\|_2 ~ vs ~ \|{\bf x}_* \|_2$','fontsize',14,'interpreter','latex');
legend('SGD rho-Lip norm','SGD rho-Lip, mu-strong convex norm', 'AGD L-smooth norm', 'AGD L-smooth, mu-strongly convex');
hold off