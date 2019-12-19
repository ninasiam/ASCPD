%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Tensor Factorization using BrasCPD                        % 
%                    ~ Non negative constraints ~                         %
%               Tensorlab is required to run the script                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all, close all;

% Path to Tensorlab
addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');
%addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');

% Tests with 3-order Tensor
% For higher order, we have to adapt 'AdaGrad' scheme

%% Initializations
order = 3;
I = 100;
J = 100;
K = 100;

dims = [I J K];

R = randi([50 min(dims)],1,1)
scale = randi([10 min(dims)],1,1)                                          % parameter to control the blocksize
B = scale*[10 10 10];                                                      % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

MAX_OUTER_ITER = 10000;                                                    % Max number of iterations


%% create true factors 
for ii = 1:order
    A_true{ii} = rand(dims(ii),R);
end

%% Put bottlenecks

%First Factor
A_corr = A_true{1};
A_corr(:,2) = A_corr(:,1);
A_true{1} = A_corr;

%Second Factor
A_corr = A_true{2};
A_corr(:,2) = A_corr(:,1);
A_true{2} = A_corr;

%Third Factor
A_corr = A_true{3};
A_corr(:,2) = A_corr(:,1);
A_true{3} = A_corr;

T = cpdgen(A_true);

%% create initial point
for jj = 1:order
    A_init{jj} = rand(dims(jj),R);
end
error_init = frob(T - cpdgen(A_init))/frob(T);                             % initial error
%% Calculate required quantities (matricizations) 

T_mat{1} = tens2mat(T,1,[2 3])';
T_mat{2} = tens2mat(T,2,[1 3])';
T_mat{3} = tens2mat(T,3,[1 2])';

%% Algorithms Initializations
A_est = A_init;                                                            % |
A_est_y_NAG = A_init;                                                      % |
eta = 0.1;                                                                 % | For Algorithm AdaBrasCPD 
epsilon = 10^(-4);                                                         % | it seems sensitive in changing epsilon
error = error_init;                                                        % |

A_est_wo_adapt = A_init;                                                   % |
alpha0 = 0.1;                                                              % | For Algorithm BrasCPD
beta = 10^(-6);                                                            % |
error_wo_adapt = error_init;                                               % |

A_est_y = A_init;                                                          % | For Algorithm Adaccel_CPD 
A_est_Adaccel = A_init;                                                    % | (we also use the eta, epsilon)
error_accel = error_init;                                                  % |


iter_per_epoch = floor(I*J/B(1));                                          % epochs

% fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);

iter = 1;

while(1)
    
    n = randi(order,1);                                                    % choose factor to update
    kr_idx = find([1:order] - n);                                          % factors for Khatri-Rao 
    J_n = dims(kr_idx(1))*dims(kr_idx(2));                                 % khatri rao dimension
    idx = randperm(J_n,B(n));                                              % choose the sample 
    F_n = sort(idx);                                                       % sorted sample of row fibers

    T_s = T_mat{n};                                                        
    T_s = T_s(F_n,:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For each method, we use the same sample, and same factor            %
    % to optimize.                                                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Accel_Bras NAG
    H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});                             % khatri rao product
    
    Hessian = svd(H(F_n,:)'*H(F_n,:));                                     % |
    L(n) = max(Hessian);                                                   % | Hessian, smoothness and str. conv parameters
    sigma(n) = min(Hessian);                                               % |
    Q(n) = (sigma(n))/(L(n));                                              % inverse condition number
    G_n = (A_est_y_NAG{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));             % derivative matrix 
    

    step = 1/L(n);
    %step_alt = 0.1;                                                       % alternative step size
    
    A_est_next = max(0,A_est_y_NAG{n} - step*G_n);                         % S.Gradient step at A_est_y
    
    beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
    A_est_y_next_NAG{n} = A_est_next + beta_accel*(A_est_next - A_est{n}); % NAG momentum

    A_est{n} = A_est_next;                                                 % update factor list
    A_est_y_NAG{n} = A_est_y_next_NAG{n};                                  % update Y factor list
    
    
    %% Simple step size (BrasCPD)
    H_Bras = kr(A_est_wo_adapt{kr_idx(2)},A_est_wo_adapt{kr_idx(1)});
    G_n_Bras = (1/B(n))*(A_est_wo_adapt{n}*H_Bras(F_n,:)'*H_Bras(F_n,:) ...
             - T_s'*H_Bras(F_n,:));
    
    alpha = alpha0/(iter^beta);                                            % step size (variant) ~ it is relatively constant though (0.01, 0.1)
    
    A_est_next_wo_adapt = max(0,A_est_wo_adapt{n} - alpha*G_n_Bras); 
     
    A_est_wo_adapt{n} = A_est_next_wo_adapt;                               % update the factor list
    
    
    %% Adaptive step size using Nesterov momentum (Adaccel_CPD)
    H_accel = kr(A_est_Adaccel{kr_idx(2)},A_est_Adaccel{kr_idx(1)});
    
    Hessian = svd((A_est_Adaccel{kr_idx(2)}'*A_est_Adaccel{kr_idx(2)}) ... % |
            .*(A_est_Adaccel{kr_idx(1)}'*A_est_Adaccel{kr_idx(1)}));       % | Hessian, smoothness and str. conv parameters
    L(n) = max(Hessian);                                                   % |
    sigma(n) = min(Hessian);
    Q(n) = (sigma(n))/(L(n));                                              % inverse condition number 
    
    G_n_accel = (1/B(n))*(A_est_y{n}*H_accel(F_n,:)'*H_accel(F_n,:) ...
              - T_s'*H_accel(F_n,:));
          
    %for adaptive step size scheme
    G_n_squared_accel = G_n_accel.^2;
    
    if iter == 1
       sum_G_n1_accel =  zeros(size(G_n_squared_accel));
       sum_G_n2_accel =  zeros(size(G_n_squared_accel));
       sum_G_n3_accel =  zeros(size(G_n_squared_accel));
    end
    
    if n == 1
        sum_G_n1_accel = sum_G_n1_accel + G_n_squared_accel;
        eta_r1_accel = eta./ ((beta + sum_G_n1_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y{n} - eta_r1_accel.*G_n_accel);    % S.Gradient step at A_est_y
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n)))); 
        A_est_y_next = A_est_accel_next + beta_accel...
                     *(A_est_accel_next - A_est_Adaccel{n});               % NAG momentum
    elseif n == 2
        sum_G_n2_accel = sum_G_n2_accel + G_n_squared_accel;
        eta_r2_accel = eta./ ((beta + sum_G_n2_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y{n} - eta_r2_accel.*G_n_accel);    % S.Gradient step at A_est_y
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
        A_est_y_next = A_est_accel_next + beta_accel...
                     *(A_est_accel_next - A_est_Adaccel{n});               % NAG momentum
    elseif n == 3
        sum_G_n3_accel = sum_G_n3_accel + G_n_squared_accel;
        eta_r3_accel = eta./ ((beta + sum_G_n3_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y{n} - eta_r3_accel.*G_n_accel);
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
        A_est_y_next = A_est_accel_next + beta_accel...
                     *(A_est_accel_next - A_est_Adaccel{n});
        
    end
    
    A_est_Adaccel{n} = A_est_accel_next;                                   % update factor list
    A_est_y{n} = A_est_y_next;                                             % update factor list of Y
    
    if(iter > MAX_OUTER_ITER)
        break;
    end
    
    %% Plotting...
    if(mod(iter,iter_per_epoch) == 0 && iter > 0)
        i = iter/iter_per_epoch;
        error(i+1) = frob(T - cpdgen(A_est))/frob(T);                      % error for accel scheme
        error_wo_adapt(i+1) = frob(T - cpdgen(A_est_wo_adapt))/frob(T);    % error for vanilla case
        error_accel(i+1) = frob(T - cpdgen(A_est_Adaccel))/frob(T);        % error for adaptive scheme with acceleration
       
        semilogy(error,'m')
        hold on;
        semilogy(error_wo_adapt,'y')
        hold on;
        semilogy(error_accel,'g')
        legend('Bras Accel','Bras', 'Adaccel Bras')
        xlabel('epochs');
        ylabel('error');
        grid on;
        pause(0.001)
    end
    
    iter = iter + 1;
    
end
% file_name = ['i' '_' num2str(scale) '_' 'R' '_' num2str(R) '_''stepAd' 'NN'];
% saveas(fig1,[file_name '.pdf']);
cpd(T,A_init);