%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input data
n = 2000;
d = 500;

A = randn(n,d);
b = randn(n,1);

[U,Sig,V] = svd(A,'econ');
MIN = 1;
MAX = 200;

eig = MIN + (MAX-MIN).*rand(d,1);
A_tmp = U*diag(eig)*V;

A = A_tmp;

condition = rcond(A'*A);
sigma = min(svd(A'*A));
L = max(svd(A'*A));

%return;
stoch_step_counter_kat = 0;
stoch_step_counter_svrg = 0;
stoch_step_counter_sg = 0;
stoch_step_counter_asg = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MAX_OUTER_ITER = 50000;
ep = 10^(-3);

%closed form solution
x_star = inv((A'*A))*A'*b;
f_star = (1/(2*n))*norm(A*x_star - b)^2;

x_init_all = randn(d,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          NON STOCHASTIC METHODS
%%Gradient descent with constant step size
%Consider that I take the full batch

tic();
x_gd = [];
x_gd(:,1) = x_init_all;

f_val_gd_merged = [];
f_val_gd = [];
f_val_gd_init = (1/(2*n))*norm(A*x_gd(:,1) - b)^2;

f_val_gd_merged = f_val_gd_init;
eta_gd = 1/(2*L);

iter = 1;
while(1)

    grad_f = A'*(A*x_gd(:,iter) - b);
    x_gd(:,iter+1) = x_gd(:,iter) - eta_gd*grad_f;
    
    f_val_gd(:,iter+1) = (1/(2*n))*norm(A*x_gd(:,iter+1) - b)^2;
    
    iter = iter + 1;
    
    crit_gd = norm(x_gd(:,iter) - x_gd(:,iter-1))/norm(x_gd(:,iter-1));
    if( crit_gd < 10^(-6) || iter*n > MAX_OUTER_ITER)%abs(f_val_gd(:,iter) - f_star) < ep)
        break;
    end
    
    f_val_gd_merged = [f_val_gd_merged; ones(n,1)*f_val_gd(iter)];
    
end

fprintf('Gradient');
time_gd = toc 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Accelerated Gradient Descend
tic();
k = 1;
x_nes = [];
x_nes(:,k) = x_init_all;
y = [];
y(:,k) = x_init_all;

a = [];
a(1) = 0.3;
b_n = [];
b_n(1) = 0;

f_val_nes_init = (1/(2*n))*norm(A*y(:,k) - b)^2;
f_val_nes_merged = f_val_nes_init;
Q= sigma/L;

%Nesterov main loop (scheme II and scheme III)

while(1)

    x_nes(:,k+1) = y(:,k) - (1/(L))*A'*(A*y(:,k)-b);

    %for scheme II
    c = [1 (a(k)^2 - Q) -a(k)^2];
    s = roots(c);

    a(k+1) = s(s > 0); 

    b_n(k+1) = (a(k)*(1 - a(k)))/(a(k)^2 + a(k+1));

    y(:,k+1) = x_nes(:,k+1) + b_n(k+1)*(x_nes(:,k+1) - x_nes(:,k));

    k = k + 1;
    
    f_val_nes(k) = (1/(2*n))*norm(A*x_nes(:,k) - b)^2;
    
    crit_nes = norm(x_nes(:,k) - x_nes(:,k-1))/norm(x_nes(:,k-1));
    if(crit_nes < 10^(-6) || k*n > MAX_OUTER_ITER)%abs(f_val_nes(k) - f_star) < ep)
        break;
    end
    
    f_val_nes_merged = [f_val_nes_merged; ones(n,1)*f_val_nes(k)];
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Nesterov');
time_nes = toc


%%Katyusha
%Initializations
m = 2*n;

t_2 = 0.5;
t_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
alpha = 1/(3*t_1*L);

x_kat = x_init_all;
y = x_kat;
z = x_kat;
x_hat = x_kat;

S = 300;

iter = 1;
sum1 = 0;
sum2 = 0;
sum_y = y;


f_val_kat = (1/(2*n))*norm(A*x_kat - b)^2;
f_val_merged_kat = [f_val_kat];
f_val_kat_par = [];

tic;
while(1)

    mu_s = (1/(n))*A'*(A*x_hat-b); %full gradient
    %augment vector of stochastic values by n
    for j = 1:m

        %k = s*m + j;

        x_next = t_1*z + t_2*x_hat + (1 - t_1 - t_2)*y;

        i = randi(n,1);

        stoch_grad = mu_s + (A(i,:)*x_next - b(i)).*A(i,:)' - (A(i,:)*x_hat - b(i)).*A(i,:)'; %calculate both

        z_next = z - alpha*stoch_grad;

        y_next = x_next - (1/(3*L))*stoch_grad;%t_1*(z_next - z);

        fval_x_par_s = (1/(2*n))*norm(A*y_next - b)^2; 
        f_val_kat_par = [f_val_kat_par ones(1,2)*fval_x_par_s];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %x = x_next;
        sum_y = sum_y + y_next;
        y = y_next;
        z = z_next;
%         sum1 = sum1 + (1 + alpha*sigma)^(j-1);
%         sum2 = sum2 + (1 + alpha*sigma)^(j-1)*y;

    end

    %here we have calculated an estimate of x
    %x_hat = inv(sum1)*sum2;
    crit_kat = norm(x_hat - y_next)/norm(y_next);
    x_hat = y_next;
    stoch_step_counter_kat = stoch_step_counter_kat + (n + 2*m); %in each iteration we have (n + m) stoch gradient computations
    
    f_val_kat_i = (1/(2*n))*norm(A*x_hat - b)^2;
    
    if( crit_kat < 10^(-6) || (m+n)*iter > MAX_OUTER_ITER) %norm(f_val_kat_i - f_star) < ep)
        break;
    end
    
    f_val_merged_kat = [f_val_merged_kat f_val_kat_par f_val_kat_i*ones(1,n)];
    f_val_kat_par = [];
    iter = iter + 1;

end

sum_of_iters_kat = iter*m;
figure();
semilogy(abs(f_val_kat - f_star));
hold on;
fprintf('Katyusha \t');
t_kat = toc;
t_kat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%SVRG

iter = 1;
n_par = 2*n; 

f_val_svrg_par  = [];
f_val_merged_svrg = [];
x_tilda_prev = x_init_all; 

f_val_svrg = [(1/(2*n))*norm(A*x_tilda_prev - b)^2];
eta = 1/(2*L);

while(1)

    x_tilda = x_tilda_prev;

    mu = (1/(n))*(A'*(A*x_tilda - b)); 
    %augment by n the vector
    x_prev = x_tilda;

    t = 1;

    while(t < n_par)
        %loop that changes x
        i_t = randi(n,1);

        x_svrg = x_prev - eta*((A(i_t,:)*x_prev - b(i_t)).*A(i_t,:)' - (A(i_t,:)*x_tilda - b(i_t)).*A(i_t,:)' + mu); %CHECK stochastic gradients

        x_prev = x_svrg;

        fval_x_par = (1/(2*n))*norm(A*x_prev - b)^2; 
        f_val_svrg_par = [f_val_svrg_par ones(1,2)*fval_x_par];
        t = t + 1;
    end
    %option I
    
    crit_svrg = norm(x_svrg - x_tilda_prev)/norm(x_tilda_prev);
    x_tilda_prev = x_svrg;
    stoch_step_counter_svrg = stoch_step_counter_svrg + (n + 2*t);
    
    fval_x = (1/(2*n))*norm(A*x_svrg - b)^2;  
 

    if( crit_svrg < 10^(-6) || (n_par+n)*iter > MAX_OUTER_ITER) %norm(fval_x - f_star) < ep)
        break;
    end
    f_val_merged_svrg = [f_val_merged_svrg f_val_svrg_par fval_x*ones(1,n)];
    f_val_svrg_par = [];
    iter = iter + 1;

end
sum_of_iters_svrg = iter*n_par;
fprintf('SVRG \t')
semilogy(abs(f_val_svrg - f_star));
hold on;
t_svrg = toc;
t_svrg

%     %accelarated stochastic gradient Prattek Jain
% 
%     tic;
%     x_pr = randn(d,n);
%     v_prev = x_pr(:,1);
%     sum_x =  x_pr(:,1);
% 
%     j = 2;
% 
%     t_burnin = n/2;
% 
%     alpha = (3*sqrt(5)*sqrt(L/sigma*(L/sigma - 0.7)))/(1+3*sqrt(5)*sqrt(L/sigma*(L/sigma - 0.7)));
%     beta = 1/(9*sqrt(L/sigma*(L/sigma - 0.7)));
%     gamma = 1/(3*sqrt(5)*sigma*sqrt(L/sigma*(L/sigma - 0.7)));
%     delta = 1/(5*sqrt(L));
% 
%     f_val_as = (1/(2*n))*norm(A*x_pr(:,1) - b)^2;
% 
%     while(1)
% 
%         if(norm(f_val_as(end) - f_star) < ep || j > n) 
%             break;
%         end
% 
%         y_prev = alpha*x_pr(:,j-1) + (1-alpha)*v_prev;
%         stoch_gradient = (A(j,:)*y_prev - b(j)).*A(j,:)';
%         x_pr(:,j) = y_prev - delta*stoch_gradient;
%         z_prev = beta*y_prev + (1 - beta)*v_prev;
%         v = z_prev - gamma*stoch_gradient;
% 
%         if j>t_burnin
%             sum_x = sum_x + x_pr(:,j-1);
%         end
%         v_prev = v;
% 
%         f_val_as = [f_val_as (1/(2*n))*norm(A*x_pr(:,j) - b)^2];
% 
%         j = j + 1;
% 
%     end
% 
%     x_asgd = (1/(n-t_burnin)*sum_x);
%     f_asgd_fin = (1/2)*norm(A*x_asgd - b)^2;
% 
%     fprintf('Accelarated SGD ~ Jain \t')
%     toc;
%     f_val_as = [f_val_as f_asgd_fin];
%     %semilogy(abs(f_val_as - f_star));

xlabel('iterations');
ylabel('|f_t - f^*|');
legend('Katyusha', 'SVRG');

hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Stochastic gradient descent type I
tic;
x_sd = [];
x_sd(:,1) = x_init_all;

f_val_sd_init = (1/(2*n))*norm(A*x_sd(:,1) - b)^2;

f_val_sd = [];
f_val_sd = [f_val_sd_init f_val_sd];
epsilon = 10^(-3);

iter = 1;


while(1) 

    rand = randi(n,1);

    if rand>n/2 
        r = randperm(rand,n/2); 
        break;  
    end
end

%r = randperm(randi(n,1),n/2,1);
eta_sd = 1/(2*max(svd(A(r,:)'*A(r,:))));

while(1)

    r = randi(n,1);

    stoch_grad = (A(r,:)*x_sd(:,iter) - b(r))*A(r,:)';

    %eta_sd = 1/norm(A(r,:))^2;
    %eta_sd = 1/sqrt(iter);
    eta_sd = 1/(iter*L);

    x_sd(:,iter+1) = x_sd(:,iter) - eta_sd*stoch_grad;
    stoch_step_counter_sg = stoch_step_counter_sg + 1;
    
    f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;
    
    crit_sg = norm(x_sd(:,iter+1)-x_sd(:,iter))/norm(x_sd(:,iter));
    if(crit_sg < 10^(-6)|| iter > MAX_OUTER_ITER) %norm(f_val_sd(iter+1) - f_star) < epsilon  )
        break;
    end

    iter = iter + 1;
end

sum_of_iters_sd = iter;

figure();
semilogy(abs(f_val_sd - f_star));
fprintf('SD Type I\t')
hold on;
t_sgd = toc;
t_sgd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Accelarated Stochastic Gradient Descent
%Initializations
tic;
k = 1;
x(:,k) = x_init_all;
y = x;

f_val_init = (1/(2*n))*norm(A*x(:,1) - b)^2;

f_val = [];
f_val = [f_val_init f_val];

Q = sigma/L;


a = [];
a(1) = 0.3;
b_par = [];
b_par(1) = 0;

%Nesterov main loop (scheme II and scheme III)
while(1)
    


    r = randi(n,1);
    eta_sd = 1/(iter*L);
    x(:,k+1) = y(:,k) - eta_sd*(A(r,:)*y(:,k) - b(r))*A(r,:)'; %1/L incorporate it to eta_Sd

    %for scheme II
    c = [1 (a(k)^2 - Q) -a(k)^2];
    s = roots(c);

    a(k+1) = s(s > 0); 

    b_par(k+1) = (a(k)*(1 - a(k)))/(a(k)^2 + a(k+1));

    y(:,k+1) = x(:,k+1) + b_par(k+1)*(x(:,k+1) - x(:,k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %for scheme III
    %y(:,k+1) = x(:,k+1) + ( (sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu)))*(x(:,k+1) - x(:,k));
    stoch_step_counter_asg = stoch_step_counter_asg + 1;
    f_val(k+1) = (1/(2*n))*norm(A*x(:,k+1) - b)^2;
    k = k + 1;
    crit_anes = norm(x(:,k) - x(:,k-1))/norm(x(:,k-1));
    if(crit_anes < 10^(-6) || k > MAX_OUTER_ITER)%f_val(end) - f_star < epsilon 
        break;
    end

end

sum_of_iters_asgd = k;
fprintf('ASGD \t');
t_asgd = toc;
t_asgd
xlabel('iterations');
ylabel('|f_t - f^*|');
semilogy(abs(f_val - f_star));
legend('SGD','ASGD');
hold off;


figure(5)
semilogy(abs(f_val_gd_merged -f_star));
hold on;
semilogy(abs(f_val_nes_merged - f_star));
hold on;
semilogy(abs(f_val_merged_kat - f_star));
hold on;
semilogy(abs(f_val_merged_svrg - f_star));
hold on;
semilogy(abs(f_val_sd - f_star));
hold on;
semilogy(abs(f_val - f_star));
grid on;
xlabel('no of stoch. gradients');
ylabel('|f_t - f^*|');
legend('Gradient','Accelerated Gradient','Katyusha','SVRG','SGD','ASGD');








