%Bras CPD

%Initializations
I = 100;
J = 100;
K = 100;
dims = [I J K];
R = 10;
B = 1*[1 1 1]; %can be smaller than rank
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

%Calculate required quantities 
T_1 = tens2mat(T,1,[2 3])';
T_2 = tens2mat(T,2,[1 3])';
T_3 = tens2mat(T,3,[1 2])';

T_mat = {T_1, T_2, T_3};

A_est = A_init;

alpha0 = 0.1;
beta = 10^(-6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Adaptive step size

eta = 0.1;
beta = 10^(-4);
epsilon = 10^(-4);                     %it seems sensitive in changing epsilon


Hessian_1 = svd((A_true{3}'*A_true{3}).*(A_true{2}'*A_true{2}));
Hessian_2 = svd((A_true{3}'*A_true{3}).*(A_true{1}'*A_true{1}));
Hessian_3 = svd((A_true{2}'*A_true{2}).*(A_true{1}'*A_true{1}));

L(1) = max(Hessian_1);
L(2) = max(Hessian_2);
L(3) = max(Hessian_3);

sigma(1) = min(Hessian_1);
sigma(2) = min(Hessian_1);
sigma(3) = min(Hessian_1);



A_outer = A_init;                     %Outer estimate of Katyusha
A_inner_y = A_init;                   %Inner loop y sequence of Katyusha
A_inner_z = A_init;                   %Inner loop z sequence of Katyusha
A_inner = A_init;
A_inner_z_next = A_init;
A_inner_y_next = A_init;

stoch_grad = [];
full_grad = [];
iter = 1;
kat_outer = 1;

while(1)
    %for the specific problem of updating a factor 
    n = randi(order,1);              %choose factor to update 
                
    kr_idx = find([1:order] - n);    %Khatri-rao dimensions
    
    J_n = dims(kr_idx(1))*dims(kr_idx(2));
    m_par = J_n/B(n);                %katyusha inner iteration limit
    tau_1 = min(1/(2*B(n)),sqrt(2*sigma(n))/sqrt(3*L(n)));
    tau_2 = min(1/2,1/(2*B(n)));     %tuning tau_2 according to batch size
    alpha_kat = 1/(3*tau_1*L(n));
    
%     sum_y_next = zeros(size(A_outer{n}));
    %Katyusha algorithm to solve the factor updating problem
    while(kat_outer < 5)            %Initially for one outer of Katyusha

        %compute the full gradient for the problem in respect to factor n 
        
        H = kr(A_outer{kr_idx(2)},A_outer{kr_idx(1)});
        full_grad = A_outer{n}*(H'*H) - T_mat{n}'*H;
        
        for j = 1:5               %Katyusha Inner loop

            A_inner{n} = tau_1*A_inner_z{n} + tau_2*A_outer{n} + (1 - tau_1 - tau_2)*A_inner_y{n};
            
            %Sampling fibers
            idx = randperm(J_n,B(n)); %choose the sample size
            F_n = sort(idx);
            
            T_s = T_mat{n};
            T_s = T_s(F_n,:);
            stoch_grad = full_grad + (1/(B(n)))*(A_inner{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:)) - (1/B(n))*(A_outer{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
            
            A_inner_z_next{n} = A_inner_z{n} - alpha_kat*stoch_grad;
            A_inner_y_next{n} = A_inner{n} - (1/(3*L(n)))*stoch_grad;
            %A_inner_y_next{n} = A_inner{n} - tau_1*(A_inner_z_next{n} - A_inner_z{n});
            
            
            %Update for next inner iter
            A_inner_y{n} = A_inner_y_next{n};
            A_inner_z{n} = A_inner_z_next{n};
            
%             sum_y_next = sum_y_next + A_inner_y{n};
            
        end
        
%         theta = 1 + min(alpha_kat*sigma(n),1/(4*m_par));
%         theta_sum = inv(j*theta);
%         
%         A_outer{n} = theta_sum*j*theta*sum_y_next;
        A_outer{n} = A_inner_y{n};
        kat_outer = kat_outer + 1;
          
    end
    
%     A_outer_kat{n} = (tau_2*j*A_outer{n} + (1 - tau_1 - tau_2)*A_inner_y{n})/(tau_2*j+(1 - tau_1 - tau_2));
    kat_outer = 1;
    A_est{n} = A_outer{n};
    
    if(max(cpderr(A_true,A_est)) < 10^(-3) || iter > MAX_OUTER_ITER)
        break;
    end
    
    error(iter) = frob(T - cpdgen(A_est))/frob(T);
    
    if(mod(iter,10) == 0)
        semilogy(error)
        pause(0.001)
    end
    
    
    iter = iter + 1;
    
end

cpd(T,A_init);