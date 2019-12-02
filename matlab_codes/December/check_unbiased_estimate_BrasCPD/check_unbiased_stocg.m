%Bras CPD accel chech unbiased estimate
clc, close all, clear all;
addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');
%addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');

%Initializations
I = 150;
J = 150;
K = 150;
dims = [I J K];
R = 10;
B = 15*[10 10 10]; %can be smaller than rank
order = 3;
I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};
MAX_OUTER_ITER = 10000;

%create true factors 
for ii = 1:order
    A_true{ii} = randn(dims(ii),R);
end

T = cpdgen(A_true);

%create initial point
for jj = 1:order
    A_init{jj} = randn(dims(jj),R);
end

lambda = [0 0 0]; %parameter of proximal term


%Calculate required quantities 
T_1 = tens2mat(T,1,[2 3])';
T_2 = tens2mat(T,2,[1 3])';
T_3 = tens2mat(T,3,[1 2])';

T_mat = {T_1, T_2, T_3};

A_est = A_init;
A_est_y = A_init;
A_est_Bras = A_init;

alpha0 = 0.1;
beta_Bras = 10^(-6);

iter = 1;
error = frob(T - cpdgen(A_est))/frob(T);
error_Bras = frob(T - cpdgen(A_est_Bras))/frob(T);

iter_per_epoch = I*J/B(1);
while(1)
    %for the specific problem of updating a factor
    n = randi(order,1); %choose factor to update
    
    kr_idx = find([1:order] - n);
    J_n = dims(kr_idx(1))*dims(kr_idx(2));
    idx = randperm(J_n,B(n)); %choose the sample size
    F_n = sort(idx);
  
    T_s = T_mat{n};
    T_s = T_s(F_n,:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %accelerated Bras
    Hessian = svd((A_est{kr_idx(2)}'*A_est{kr_idx(2)}).*(A_est{kr_idx(1)}'*A_est{kr_idx(1)}));
    L(n) = max(Hessian);
    sigma(n) = min(Hessian);
    
    H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});
    G_n = (1/B(n))*(A_est_y{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:)) - lambda(n)*(A_est_y{n} - A_est{n});
    Q(n) = (sigma(n) + lambda(n))/(L(n) + lambda(n));
    
    A_est_next = A_est_y{n} - ((J_n*B(n))/(L(n)*dims(n)))*G_n;
    
    beta = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
    A_est_y_next{n} = A_est_next + beta*(A_est_next - A_est{n});

    A_est{n} = A_est_next;
    A_est_y{n} = A_est_y_next{n};
    
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %simple Bras
    H_Bras = kr(A_est_Bras{kr_idx(2)},A_est_Bras{kr_idx(1)});
    G_n_bras = (1/B(n))*(A_est_Bras{n}*H_Bras(F_n,:)'*H_Bras(F_n,:) - T_s'*H_Bras(F_n,:));
    
    alpha_Bras = alpha0/(iter^beta_Bras);
    
    A_est_next_Bras = A_est_Bras{n} - alpha_Bras*G_n_bras; 
    
    A_est_Bras{n} = A_est_next_Bras;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if(iter > MAX_OUTER_ITER )%max(cpderr(A_true,A_est)) < 10^(-4) || )
        break;
    end
    
    if(mod(iter,iter_per_epoch) == 0 && iter > 1)
        i = iter/iter_per_epoch;
        error_Bras(i+1) = frob(T - cpdgen(A_est_Bras))/frob(T);
        error(i+1) = frob(T - cpdgen(A_est))/frob(T);
        semilogy(error)
        hold on;
        semilogy(error_Bras,'g')
        hold off;
        xlabel('epochs');
        ylabel('error');
        legend('Bras accel','Bras');
        grid on;
        pause(0.001)
    end
    
    iter = iter + 1;

end
err_Bras_accel = cpderr(A_true,A_est)
err_Bras = cpderr(A_true,A_est_Bras)
A_est_cpd = cpd(T,A_init);