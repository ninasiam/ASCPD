%Bras CPD accel

%Initializations for the problem
I = 100;
J = 100;
K = 100;
dims = [I J K];
R = 10;
B = 10*[100 100 100]; %can be smaller than rank
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

%calculata L,sigma parametes For Nesterov
Hessian_1_eigs = svd((A_true{3}'*A_true{3}).*(A_true{2}'*A_true{2}));
Hessian_2_eigs = svd((A_true{3}'*A_true{3}).*(A_true{1}'*A_true{1}));
Hessian_3_eigs = svd((A_true{2}'*A_true{2}).*(A_true{1}'*A_true{1}));

L(1) = max(Hessian_1_eigs);
L(2) = max(Hessian_2_eigs);
L(3) = max(Hessian_3_eigs);

sigma(1) = min(Hessian_1_eigs);
sigma(2) = min(Hessian_2_eigs);
sigma(3) = min(Hessian_3_eigs);

%Proximal parameter
lambda = 1*sigma./1000;

%Condition number (proximal)
Q = (L + lambda)./(sigma + lambda);


%Calculate required quantities 
T_1 = tens2mat(T,1,[2 3])';
T_2 = tens2mat(T,2,[1 3])';
T_3 = tens2mat(T,3,[1 2])';

T_mat = {T_1, T_2, T_3};

%Variable initialize
A_est = A_init;
A_est_y = A_init;

A_est_woAdapt = A_init;
A_est_y_woAdapt = A_init;

A_est_Bras = A_init;

error_n1 = [];
error_n2 = [];
error_n3 = [];

%parameters Bras
alpha0 = 0.1;
beta_Bras = 10^(-6);
%parameter Nesterov (woAdapt)
thr = 800;
%parameter Nesterov (Adapt)
alpha = {[1], [1] ,[1]};
restart_crit = zeros(3,1);

iter = 1;

while(1)
    %for the specific problem of updating a facto at random
    n = randi(order,1); %choose factor to update
    
    kr_idx = find([1:order] - n);
    J_n = dims(kr_idx(1))*dims(kr_idx(2));
    idx = randperm(J_n,B(n)); %choose the sample size
    F_n = sort(idx);

    H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)}); %The right dimension of Khatri rao
    
    T_s = T_mat{n};
    T_s = T_s(F_n,:);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%With Adaptive Stepsize
    G_n = (1/(B(n)))*(A_est_y{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:)) - lambda(n)*(A_est_y{n} - A_est{n}); 
    
    %Upadate the alpha-beta parameters (momentum paremeter)
    step = (iter/(L(n)+lambda(n)));
    %Check the restart criterion
%     if(restart_crit(n) > 0)
%         alpha{n}(end) = 1;
%         step = (1/((iter) + L(n)+lambda(n))); %check
%     end
    if(restart_crit(n) > 0)
        alpha{1}(end) = 1;
        alpha{2}(end) = 1;
        alpha{3}(end) = 1;
        step = (1/((iter) + L(n)+lambda(n))); %check
    end   

    q = 0; %OKK this is weird
    a = 1;
    b = alpha{n}(end)^2 - q;
    c = -alpha{n}(end)^2;
    D = b^2 - 4 * a * c;

    alpha{n}(end+1) = (-b+sqrt(D))/2;                                              %correct
    beta = (alpha{n}(end-1)*(1-alpha{n}(end-1)))/(alpha{n}(end-1)^2 + alpha{n}(end));  %correct
        
    A_est_next = A_est_y{n} - step*G_n; 
    A_est_y_next{n} = A_est_next + beta*(A_est_next - A_est{n});
       
    A_est{n} = A_est_next;
    A_est_y{n} = A_est_y_next{n};
    
    error(iter) = frob(T - cpdgen(A_est))/frob(T);
    if iter > 1 && size(error,2) > 1
        restart_crit(n) = error(end) - error(end-1);
    end
    %check for the whole error
%     if n == 1
%         error_n1 = [error_n1 error(iter)];
%         if iter > 1 && size(error_n1,2) > 1
%             restart_crit(n) = error_n1(end) - error_n1(end-1);
%         end
%     elseif n == 2 
%         error_n2 = [error_n2 error(iter)];
%         if iter > 1 && size(error_n2,2) > 1
%             restart_crit(n) = error_n2(end) - error_n2(end-1);
%         end
%     elseif n == 3 
%         error_n3 = [error_n3 error(iter)];
%         if iter > 1 && size(error_n3,2) > 1
%             restart_crit(n) = error_n3(end) - error_n3(end-1);
%         end
%     end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Nesterov without adaptation
    G_n = (1/(B(n)))*(A_est_y_woAdapt{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:))- lambda(n)*(A_est_y_woAdapt{n} - A_est_woAdapt{n});

    if iter < thr
        A_est_next_woAdapt = A_est_y_woAdapt{n} - (iter/(L(n)+lambda(n)))*G_n;

    else    
        A_est_next_woAdapt = A_est_y_woAdapt{n} - ((iter/3)/(iter + L(n)+lambda(n)))*G_n;
    end
    
    A_est_y_next_woAdapt{n} = A_est_next_woAdapt + ((sqrt(Q(n))-1)/(sqrt(Q(n))+1))*(A_est_next_woAdapt - A_est_woAdapt{n});

    A_est_woAdapt{n} = A_est_next_woAdapt;
    A_est_y_woAdapt{n} = A_est_y_next_woAdapt{n};
    
    error_woAdapt(iter) = frob(T - cpdgen(A_est_woAdapt))/frob(T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %simple Bras
    G_n = (1/B(n))*(A_est_Bras{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
    
    alpha_Bras = alpha0/(iter^beta_Bras);
    
    A_est_next_Bras = A_est_Bras{n} - alpha_Bras*G_n; 
    
    A_est_Bras{n} = A_est_next_Bras;
    
    error_Bras(iter) = frob(T - cpdgen(A_est_Bras))/frob(T);
    if(iter > MAX_OUTER_ITER)
        break;
    end
    
    
    if(mod(iter,100) == 0)
        semilogy(error,'r')
        hold on;
        semilogy(error_woAdapt,'y')
        hold on;
        semilogy(error_Bras,'g')
        legend('With AdaptiveStep','Without AdaptiveStep','Bras');
        pause(0.001)
    end
    
    iter = iter + 1;
    
end

cpd(T,A_init);