%Bras CPD non-negative
clear all, close all;
%addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');
addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');
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
    A_true{ii} = rand(dims(ii),R);
end

T = cpdgen(A_true);

%create initial point
for jj = 1:order
    A_init{jj} = rand(dims(jj),R);
end

%Calculate required quantities 
T_1 = tens2mat(T,1,[2 3])';
T_2 = tens2mat(T,2,[1 3])';
T_3 = tens2mat(T,3,[1 2])';

T_mat = {T_1, T_2, T_3};

A_est = A_init;
A_est_wo_adapt = A_init;
A_est_accel = A_init;
A_est_y = A_init;

alpha0 = 1.5;
beta = 10^(-6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Adaptive step size

eta = 0.1;
beta = 10^(-4);
epsilon = 10^(-4); %it seems sensitive in changing epsilon

sum_G_n = [];
iter = 1;

while(1)
    %for the specific problem of updating a fa
    n = randi(order,1); %choose factor to update
    
    kr_idx = find([1:order] - n);
    J_n = dims(kr_idx(1))*dims(kr_idx(2));
    idx = randperm(J_n,B(n)); %choose the sample size
    F_n = sort(idx);

    H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});
    
    T_s = T_mat{n};
    T_s = T_s(F_n,:);
    
    G_n = (1/B(n))*(A_est{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %for adaptive step size 
    G_n_squared = G_n.^2;
    if iter == 1
       sum_G_n1 =  zeros(size(G_n_squared));
       sum_G_n2 =  zeros(size(G_n_squared));
       sum_G_n3 =  zeros(size(G_n_squared));
    end
    
    if n == 1
        sum_G_n1 = sum_G_n1 + G_n_squared;
        eta_r1 = eta./ ((beta + sum_G_n1).^(1/2 + epsilon));
        A_est_next = max(0,A_est{n} - eta_r1.*G_n);
    elseif n == 2
        sum_G_n2 = sum_G_n2 + G_n_squared;
        eta_r2 = eta./ ((beta + sum_G_n2).^(1/2 + epsilon));
        A_est_next = max(0,A_est{n} - eta_r2.*G_n);
    elseif n == 3
        sum_G_n3 = sum_G_n3 + G_n_squared;
        eta_r3 = eta./ ((beta + sum_G_n3).^(1/2 + epsilon));
        A_est_next = max(0,A_est{n} - eta_r3.*G_n);
    end
    
    A_est{n} = A_est_next;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Simple step size
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    G_n_Bras = (1/B(n))*(A_est_wo_adapt{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
    
    alpha = alpha0/(iter^beta);
    
    A_est_next_wo_adapt = max(0,A_est_wo_adapt{n} - alpha*G_n_Bras); 
     
    A_est_wo_adapt{n} = A_est_next_wo_adapt;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Hessian = svd((A_est_accel{kr_idx(2)}'*A_est_accel{kr_idx(2)}).*(A_est_accel{kr_idx(1)}'*A_est_accel{kr_idx(1)}));
    L(n) = max(Hessian);
    sigma(n) = min(Hessian);
    
    G_n_accel = (1/B(n))*(A_est_y{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
    Q(n) = (sigma(n))/(L(n));
    
    A_est_next_accel = max(0,A_est_y{n} - ((J_n*B(n))/(L(n)*dims(n)))*G_n_accel);
    
    beta = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
    A_est_y_next{n} = A_est_next_accel + beta*(A_est_next_accel - A_est_accel{n});

    A_est_accel{n} = A_est_next_accel;
    A_est_y{n} = A_est_y_next{n};
    
    if(iter > MAX_OUTER_ITER)
        break;
    end
    
    if(mod(iter,100) == 0 && iter > 0)
        i = iter/100;
        error(i) = frob(T - cpdgen(A_est))/frob(T); %Error for adaptive scheme
        error_wo_adapt(i) = frob(T - cpdgen(A_est_wo_adapt))/frob(T); %error for vanilla case
        error_accel(i) = frob(T - cpdgen(A_est_accel))/frob(T);
        semilogy(error,'m')
        hold on;
        semilogy(error_wo_adapt,'y')
        hold on;
        semilogy(error_accel,'g')
        legend('Adagrad Bras','Bras', 'Bras_accel')
        pause(0.001)
    end
    
    iter = iter + 1;
    
end

cpd(T,A_init);