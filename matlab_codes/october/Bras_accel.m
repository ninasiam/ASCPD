%Bras CPD accel

%Initializations
I = 100;
J = 100;
K = 100;
dims = [I J K];
R = 10;
B = 10*[100 100 100]; %can be smaller than rank
order = 3;
I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};
MAX_OUTER_ITER = 50000;

%create true factors 
for ii = 1:order
    A_true{ii} = randn(dims(ii),R);
end

T = cpdgen(A_true);

%create initial point
for jj = 1:order
    A_init{jj} = randn(dims(jj),R);
end

Hessian_1 = svd((A_true{3}'*A_true{3}).*(A_true{2}'*A_true{2}));
Hessian_2 = svd((A_true{3}'*A_true{3}).*(A_true{1}'*A_true{1}));
Hessian_3 = svd((A_true{2}'*A_true{2}).*(A_true{1}'*A_true{1}));

L(1) = max(Hessian_1);
L(2) = max(Hessian_2);
L(3) = max(Hessian_3);

sigma(1) = min(Hessian_1);
sigma(2) = min(Hessian_1);
sigma(3) = min(Hessian_1);

lambda = 1*sigma./100;%10*sigma;
Q = (L + lambda)./(sigma + lambda);

%Calculate required quantities 
T_1 = tens2mat(T,1,[2 3])';
T_2 = tens2mat(T,2,[1 3])';
T_3 = tens2mat(T,3,[1 2])';

T_mat = {T_1, T_2, T_3};

A_est = A_init;
A_est_y = A_init;

alpha0 = 0.01;
beta = 10^(-4)
% gamma = 0.001;
% beta = 1./(100.*sigma);

% alpha = []
% alpha(:,1) = ones(3,1);
thr = 800;
iter = 1;
iter2 = 1;
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
    
    G_n = (1/B(n))*(A_est_y{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:)) - lambda(n)*(A_est_y{n} - A_est{n});
    
    if iter < thr
        A_est_next = A_est_y{n} - (iter/(L(n)+lambda(n)))*G_n;

    else    
        A_est_next = A_est_y{n} - ((iter/3)/(iter + L(n)+lambda(n)))*G_n;
    end


    A_est_y_next{n} = A_est_next + ((sqrt(Q(n))-1)/(sqrt(Q(n))+1))*(A_est_next - A_est{n});

    A_est{n} = A_est_next;
    A_est_y{n} = A_est_y_next{n};
    
    if(max(cpderr(A_true,A_est)) < 10^(-3) || iter > MAX_OUTER_ITER)
        break;
    end
    
    error(iter) = frob(T - cpdgen(A_est))/frob(T);
    
    if(mod(iter,100) == 0)
        semilogy(error)
        pause(0.001)
    end
    
    iter = iter + 1;
    iter2 = iter;
    
end

cpd(T,A_init);