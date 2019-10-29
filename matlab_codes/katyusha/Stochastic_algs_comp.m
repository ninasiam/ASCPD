%katyusha
    %plot for five different experiments   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input data
    n = 2000;
    d = 500;

    A = randn(n,d);
    b = randn(n,1);
    
    condition = rcond(A'*A);
    sigma = min(svd(A'*A));
    L = max(svd(A'*A));
    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          NON STOCHASTIC METHODS
%%Gradient descent with constant step size
tic();
x_gd = [];
x_gd(:,1) = zeros(n,1);

f_val_gd = [];

eta_gd = 1/(2*max(svd(A'*A)));

iter = 1;
while(1)

    grad_f = A'*(A*x_gd(:,iter) - b);
    x_gd(:,iter+1) = x_gd(:,iter) - eta_gd*grad_f;
    
    f_val_gd(:,iter+1) = (1/(2*n))*norm(A*x_gd(:,iter+1) - b)^2;
    
    iter = iter + 1;
    
    if(abs(f_val_gd(:,iter) - f_star)< ep)
        break;
    end
    
    f_val_gd_merged = [f_val_gd_merged ones(n,1)*f_val_gd];
end

time_gd = toc 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Accelerated Gradient Descend
k = 1;

a = [];
a(1) = 0.3;
b = [];
b(1) = 0;

%Nesterov main loop (scheme II and scheme III)
while(1)
    
    x_nes(:,k+1) = y(:,k) - (1/L)*(*y(:,k) + q);
    
    %for scheme II
    c = [1 (a(k)^2 - Q) -a(k)^2];
    s = roots(c);
    
    a(k+1) = s(s > 0); 
    
    b(k+1) = (a(k)*(1 - a(k)))/(a(k)^2 + a(k+1));
    
    y(:,k+1) = x(:,k+1) + b(k+1)*(x(:,k+1) - x(:,k));
   
    k = k + 1;
    
    if(abs(f_val_nes(k) - f_star))
        break;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Katyusha
    %Initializations
    m = 2*n;

    t_2 = 0.5;
    t_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
    alpha = 1/(3*t_1*L);

    x_kat = randn(d,1);
    y = x_kat;
    z = x_kat;
    x_hat = x_kat;

    S = 300;

    iter = 1;
    sum1 = 0;
    sum2 = 0;
    sum_y = y;
    f_val_merged_kat = [];
    f_val_kat = (1/(2*n))*norm(A*x_kat - b)^2;
    tic;
    f_val_kat_par = [];
    while(1)


        mu_s = (1/(n))*A'*(A*x_hat-b); %full gradient

        for j = 1:m

            %k = s*m + j;

            x_next = t_1*z + t_2*x_hat + (1 - t_1 - t_2)*y;

            i = randi(n,1);

            stoch_grad = mu_s + (A(i,:)*x_next - b(i)).*A(i,:)' - (A(i,:)*x_hat - b(i)).*A(i,:)';

            z_next = z - alpha*stoch_grad;

            y_next = x_next - (1/(3*L))*stoch_grad;%t_1*(z_next - z);

            fval_x_par_s = (1/(2*n))*norm(A*y_next - b)^2; 
            f_val_kat_par = [f_val_kat_par fval_x_par_s];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %x = x_next;
            sum_y = sum_y + y_next;
            y = y_next;
            z = z_next;
            %sum1 = sum1 + (1 + alpha*sigma)^(j-1);
            %sum2 = sum2 + (1 + alpha*sigma)^(j-1)*y;

        end
        
        %here we have calculated an estimate of x
        %x_hat = inv(sum1)*sum2;
        x_hat = y_next;%(1/m)*sum_y;
        stoch_step_counter_kat = stoch_step_counter_kat + (n + 2*m); %in each iteration we have (n + m) stoch gradient computations
        f_val_kat_i = (1/(2*n))*norm(A*x_hat - b)^2;
        f_val_kat = [f_val_kat f_val_kat_i];

        if(norm(f_val_kat(iter) - f_star) < ep || m*iter > MAX_OUTER_ITER )
            break;
        end
        f_val_merged_kat = [f_val_merged_kat f_val_kat_par f_val_kat_i];
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
    x_tilda_prev = randn(d,1); 

    f_val_svrg = [(1/(2*n))*norm(A*x_tilda_prev - b)^2];
    eta = 1/(2*max(svd(A'*A)));
    tic;

    while(1)

        x_tilda = x_tilda_prev;

        mu = 1/(n)*(A'*(A*x_tilda - b)); 

        x_prev = x_tilda;

        t = 1;

        while(t < n_par)
            %loop that changes x
            i_t = randi(n,1);

            x_svrg = x_prev - eta*((A(i_t,:)*x_prev - b(i_t)).*A(i_t,:)' - (A(i_t,:)*x_tilda - b(i_t)).*A(i_t,:)' + mu); 

            x_prev = x_svrg;
            
            fval_x_par = (1/(2*n))*norm(A*x_prev - b)^2; 
            f_val_svrg_par = [f_val_svrg_par fval_x_par];
            t = t + 1;
        end
        %option I

        x_tilda_prev = x_svrg;
        stoch_step_counter_svrg = stoch_step_counter_svrg + (n + 2*t);
        fval_x = (1/(2*n))*norm(A*x_svrg - b)^2;  
        f_val_svrg = [f_val_svrg fval_x];

        if(norm(f_val_svrg(iter) - f_star) < ep || n_par*iter > MAX_OUTER_ITER)
            break;
        end
        f_val_merged_svrg = [f_val_merged_svrg f_val_svrg_par fval_x];
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
    x_sd(:,1) = randn(d,1);

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
        %eta_sd = 1/(iter*L);
        
        x_sd(:,iter+1) = x_sd(:,iter) - eta_sd*stoch_grad;
        stoch_step_counter_sg = stoch_step_counter_sg + 1;
        f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;

        if(norm(f_val_sd(iter+1) - f_star) < epsilon || iter > MAX_OUTER_ITER )
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
    x = randn(d,1);
    y = x;

    f_val_init = (1/(2*n))*norm(A*x(:,1) - b)^2;

    f_val = [];
    f_val = [f_val_init f_val];

    Q = inv(condition);
    k = 1;

    a = [];
    a(1) = 0.3;
    b_par = [];
    b_par(1) = 0;

    %Nesterov main loop (scheme II and scheme III)
    while(1)

        if(f_val(end) - f_star < epsilon || k > MAX_OUTER_ITER)
            break;
        end

        r = randi(n,1);

        x(:,k+1) = y(:,k) - eta_sd*(A(r,:)*x(:,k) - b(r))*A(r,:)';

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
    semilogy(abs(f_val_merged_kat - f_star))
    hold on;
    semilogy(abs(f_val_merged_svrg - f_star));
    hold on;
    semilogy(abs(f_val_sd - f_star));
    hold on;
    semilogy(abs(f_val - f_star));
    xlabel('no of stoch. gradients');
    ylabel('|f_t - f^*|');
    legend('Katyusha','SVRG','SGD','ASGD');








