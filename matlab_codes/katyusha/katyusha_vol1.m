%katyusha
    %plot for five different experiments   
iters_katp = zeros(5,1);
iters_svrgp = zeros(5,1);
iters_sgdp = zeros(5,1);
iters_asgdp = zeros(5,1);

time_katp = zeros(5,1);
time_svrgp = zeros(5,1);
time_sgdp = zeros(5,1);
time_asgdp = zeros(5,1);


n = 3000;
d = 1000;



for ii = 1:5
    ii

    A = randn(n,d);
    b = randn(n,1);

    MAX_OUTER_ITER = 3000;
    ep = 10^(-4);
    %closed form solution
    x_star = inv((A'*A))*A'*b;
    f_star = (1/(2*n))*norm(A*x_star - b)^2;

    condition = cond(A'*A);

    sigma = min(svd(A'*A));
    L = max(svd(A'*A));

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
    f_val_kat = (1/(2*n))*norm(A*x_kat - b)^2;
    tic;

    while(1)


        mu_s = (1/(2*n))*A'*(A*x_hat-b); %full gradient

        for j = 1:m

            %k = s*m + j;

            x_next = t_1*z + t_2*x_hat + (1 - t_1 - t_2)*y;

            i = randi(n,1);

            stoch_grad = mu_s + (A(i,:)*x_next - b(i)).*A(i,:)' - (A(i,:)*x_hat - b(i)).*A(i,:)';

            z_next = z - alpha*stoch_grad;

            y_next = x_next - (1/(3*L))*stoch_grad;%t_1*(z_next - z);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %x = x_next;
            sum_y = sum_y + y_next;
            y = y_next;
            z = z_next;
            %sum1 = sum1 + (1 + alpha*sigma)^(j-1);
            %sum2 = sum2 + (1 + alpha*sigma)^(j-1)*y;

        end

        %x_hat = inv(sum1)*sum2;
        x_hat = y_next;%(1/m)*sum_y;
        f_val_kat = [f_val_kat (1/(2*n))*norm(A*x_hat - b)^2];

        if(norm(f_val_kat(iter) - f_star) < ep)
            break;
        end

        iter = iter + 1;


    end

    sum_of_iters_kat = iter*m;
    figure();
    semilogy(abs(f_val_kat - f_star));
    hold on;
    fprintf('Katyusha \t');
    t_kat = toc;
    t_kat

    %SVRG

    iter = 1;
    n_par = 2*n; 


    x_tilda_prev = randn(d,1); 

    f_val_svrg = [(1/(2*n))*norm(A*x_tilda_prev - b)^2];
    eta = 1/(2*max(svd(A'*A)));
    tic;

    while(1)

        x_tilda = x_tilda_prev;

        mu = 1/(2*n)*(A'*(A*x_tilda - b)); 

        x_prev = x_tilda;

        t = 1;

        while(t < n_par)
            %loop that changes x
            i_t = randi(n,1);

            x_svrg = x_prev - eta*((A(i_t,:)*x_prev - b(i_t)).*A(i_t,:)' - (A(i_t,:)*x_tilda - b(i_t)).*A(i_t,:)' + mu); 

            x_prev = x_svrg;

            t = t + 1;
        end
        %option I

        x_tilda_prev = x_svrg;
        fval_x = (1/(2*n))*norm(A*x_svrg - b)^2;  
        f_val_svrg = [f_val_svrg fval_x];

        if(norm(f_val_svrg(iter) - f_star) < ep)
            break;
        end

        iter = iter + 1;

    end
    sum_of_iters_svrg = iter*n_par;
    fprintf('SVRG \t')
    semilogy(abs(f_val_svrg - f_star));
    hold on;
    t_svrg = toc;
    t_svrg
    %accelarated stochastic gradient Prattek Jain

    tic;
    x_pr = randn(d,n);
    v_prev = x_pr(:,1);
    sum_x =  x_pr(:,1);

    j = 2;

    t_burnin = n/2;

    alpha = (3*sqrt(5)*sqrt(L/sigma*(L/sigma - 0.7)))/(1+3*sqrt(5)*sqrt(L/sigma*(L/sigma - 0.7)));
    beta = 1/(9*sqrt(L/sigma*(L/sigma - 0.7)));
    gamma = 1/(3*sqrt(5)*sigma*sqrt(L/sigma*(L/sigma - 0.7)));
    delta = 1/(5*sqrt(L));

    f_val_as = (1/(2*n))*norm(A*x_pr(:,1) - b)^2;

    while(1)

        if(norm(f_val_as(end) - f_star) < ep || j > n) 
            break;
        end

        y_prev = alpha*x_pr(:,j-1) + (1-alpha)*v_prev;
        stoch_gradient = (A(j,:)*y_prev - b(j)).*A(j,:)';
        x_pr(:,j) = y_prev - delta*stoch_gradient;
        z_prev = beta*y_prev + (1 - beta)*v_prev;
        v = z_prev - gamma*stoch_gradient;

        if j>t_burnin
            sum_x = sum_x + x_pr(:,j-1);
        end
        v_prev = v;

        f_val_as = [f_val_as (1/(2*n))*norm(A*x_pr(:,j) - b)^2];

        j = j + 1;

    end

    x_asgd = (1/(n-t_burnin)*sum_x);
    f_asgd_fin = (1/2)*norm(A*x_asgd - b)^2;

    fprintf('Accelarated SGD ~ Jain \t')
    toc;
    f_val_as = [f_val_as f_asgd_fin];
    semilogy(abs(f_val_as - f_star));
    xlabel('iterations');
    ylabel('|f_t - f^*|');
    legend('Katyusha', 'SVRG','ASGD');

    hold off;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Stochastic gradient descent type I
    tic;
    x_sd = [];
    x_sd(:,1) = randn(d,1);

    f_val_sd_init = (1/(2*n))*norm(A*x_sd(:,1) - b)^2;

    f_val_sd = [];
    f_val_sd = [f_val_sd_init f_val_sd];
    epsilon = 10^(-2);

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

        f_val_sd(:,iter+1) = (1/(2*n))*norm(A*x_sd(:,iter+1) - b)^2;

        if(norm(f_val_sd(iter+1) - f_star) < epsilon || iter > 50000 )
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

        if(f_val(end) - f_star < epsilon || k > 50000)
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
    pause;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %saving variables
    iters_katyusha(ii) = sum_of_iters_kat;
    iters_svrg(ii) = sum_of_iters_svrg;

    iters_sgd(ii) = sum_of_iters_sd;
    iters_asgd(ii) = sum_of_iters_asgd;

    time_katyusha(ii) = t_kat;
    time_svrg(ii) = t_svrg;

    time_sgd(ii) = t_sgd;
    time_asgd(ii) = t_asgd;

end
%1 -> 2000,1000
%2 -> 3000,1000
    
iters_katp(2) = mean(iters_katyusha);
iters_svrgp(2) = mean(iters_svrg);

iters_sgdp(2) = mean(iters_sgd);
iters_asgdp(2) = mean(iters_asgd);

time_katp(2) = mean(time_katyusha);
time_svrgp(2) = mean(time_svrg);

time_sgdp(2) = mean(time_sgd);
time_asgdp(2) = mean(time_asgd);










