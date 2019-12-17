%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Tensor Factorization using BrasCPD                        % 
%                    ~ Unconstrained Problem ~                            %
%               Tensorlab is required to run the script                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, close all, clear all;

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

R = randi([10 min(dims)],1,1)
scale = randi([10 min(dims)],1,1)                                          % parameter to control the blocksize
B = scale*[10 10 10];                                                      %can be smaller than rank


%I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

MAX_OUTER_ITER = 10000;                                                    % Max number of iterations

%% create true factors 
for ii = 1:order
    A_true{ii} = randn(dims(ii),R);
end

T = cpdgen(A_true);

%% create initial point
for jj = 1:order
    A_init{jj} = randn(dims(jj),R);
end
error_init = frob(T - cpdgen(A_init))/frob(T);                             % initial error

%% Calculate required quantities 
T_mat{1} = tens2mat(T,1,[2 3])';
T_mat{2} = tens2mat(T,2,[1 3])';
T_mat{3} = tens2mat(T,3,[1 2])';

%% Algorithms Initializations

A_est = A_init;                                                            % |
A_est_y = A_init;                                                          % | For Accelerated Bras with Nesterov Momentum
error = error_init;                                                        % |

A_est_Bras = A_init;                                                       % |
alpha0 = 0.1;                                                              % | For BrasCPD
beta_Bras = 10^(-6);                                                       % |
error_Bras = error_init;                                                   % |

A_est_Adaccel = A_init;                                                    % |
A_est_y_adap = A_init;                                                     % |
eta = 0.1;                                                                 % | For Adaccel with Nesterov Momentum
epsilon = 10^(-4);                                                         % | %it seems sensitive in changing epsilon
error_Adaccel = error_init;                                                % |

%beta_Bras_accel = 10^(-6);

iter = 1;

iter_per_epoch = floor(I*J/B(1));                                          % epochs

% fig1 = figure('units','normalized','outerposition',[0 0 0.4 0.4]);

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
    
    Hessian = svd((A_est{kr_idx(2)}'*A_est{kr_idx(2)}) ...                 % |
            .*(A_est{kr_idx(1)}'*A_est{kr_idx(1)}));                       % |
    L(n) = max(Hessian);                                                   % | Hessian, smoothness and str. conv parameters
    sigma(n) = min(Hessian);                                               % |
    Q(n) = (sigma(n))/(L(n));                                              % inverse condition number
    
    H = kr(A_est{kr_idx(2)},A_est{kr_idx(1)});
    G_n = (1/B(n))*(A_est_y{n}*H(F_n,:)'*H(F_n,:) - T_s'*H(F_n,:));
        
    step = (R*size(F_n,2)/(L(n)));                                         % relative small block size
%     step = (R*J_n/(L(n)*dims(n))); % seems to work for all ranks
%     step = 1/(L(n)); % BAD
%     step_alt = (((alpha0))/(iter^beta_Bras_accel)); % to show that
%     acceleration works
%     step_alt = ((J_n*B(n))/(L(n)*dims(n)*sqrt(iter)));% test for big 
%ranks
    step_alt = 0.1;                                                        % alternative step size
    
    A_est_next = A_est_y{n} - min(step,step_alt)*G_n;                      % S.Gradient step at A_est_y
    
    beta = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
    A_est_y_next{n} = A_est_next + beta*(A_est_next - A_est{n});           % NAG momentum

    A_est{n} = A_est_next;                                                 % update factor list
    A_est_y{n} = A_est_y_next{n};                                          % update Y factor list
    
   
%% Simple step size (BrasCPD)
   
    H_Bras = kr(A_est_Bras{kr_idx(2)},A_est_Bras{kr_idx(1)});
    G_n_bras = (1/B(n))*(A_est_Bras{n}*H_Bras(F_n,:)'*H_Bras(F_n,:) ...
             - T_s'*H_Bras(F_n,:));
    
    alpha_Bras = alpha0/(iter^beta_Bras);                                  % step size (variant) ~ it is relatively constant though (0.01, 0.1)
    
    A_est_next_Bras = A_est_Bras{n} - alpha_Bras*G_n_bras; 
    
    A_est_Bras{n} = A_est_next_Bras;                                       % update the factor list
    
    %% Adaptive step size using Nesterov momentum (Adaccel_CPD)
    
    H_accel = kr(A_est_Adaccel{kr_idx(2)},A_est_Adaccel{kr_idx(1)});
    
    Hessian = svd((A_est_Adaccel{kr_idx(2)}'*A_est_Adaccel{kr_idx(2)})...  % |
            .*(A_est_Adaccel{kr_idx(1)}'*A_est_Adaccel{kr_idx(1)}));       % |
    L(n) = max(Hessian);                                                   % | Hessian, smoothness and str. conv parameters
    sigma(n) = min(Hessian);                                               % |
    Q(n) = (sigma(n))/(L(n));                                              % inverse condition number
    
    G_n_accel = (1/B(n))*(A_est_y_adap{n}*H_accel(F_n,:)'*H_accel(F_n,:)...
              - T_s'*H_accel(F_n,:));
    
    %for adaptive step size 
    G_n_squared_accel = G_n_accel.^2;
    
    if iter == 1
       sum_G_n1_accel =  zeros(size(G_n_squared_accel));
       sum_G_n2_accel =  zeros(size(G_n_squared_accel));
       sum_G_n3_accel =  zeros(size(G_n_squared_accel));
    end
    
    if n == 1
        sum_G_n1_accel = sum_G_n1_accel + G_n_squared_accel;
        eta_r1_accel = eta./ ((beta + sum_G_n1_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y_adap{n} - ...      
                               eta_r1_accel.*G_n_accel);                   % S.Gradient step at A_est_y
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n)))); 
        A_est_y_next_adap = A_est_accel_next + beta_accel...
                          *(A_est_accel_next - A_est_Adaccel{n});          % NAG momentum
    elseif n == 2
        sum_G_n2_accel = sum_G_n2_accel + G_n_squared_accel;
        eta_r2_accel = eta./ ((beta + sum_G_n2_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y_adap{n} - ...
                               eta_r2_accel.*G_n_accel);
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
        A_est_y_next_adap = A_est_accel_next + beta_accel...
                          *(A_est_accel_next - A_est_Adaccel{n});
    elseif n == 3
        sum_G_n3_accel = sum_G_n3_accel + G_n_squared_accel;
        eta_r3_accel = eta./ ((beta + sum_G_n3_accel).^(1/2 + epsilon));
        A_est_accel_next = max(0,A_est_y_adap{n} - ...
                               eta_r3_accel.*G_n_accel);
        beta_accel = ((1-sqrt(Q(n)))/(1 + sqrt(Q(n))));
        A_est_y_next_adap = A_est_accel_next + beta_accel...
                          *(A_est_accel_next - A_est_Adaccel{n});
        
    end
    
    A_est_Adaccel{n} = A_est_accel_next;                                   % update factor list
    A_est_y_adap{n} = A_est_y_next_adap;                                   % update Y factor list
    
    if( iter > MAX_OUTER_ITER )
        break;
    end
    
    
    if(mod(iter,iter_per_epoch) == 0 && iter > 1)
        i = iter/iter_per_epoch;
        error_Bras(i+1) = frob(T - cpdgen(A_est_Bras))/frob(T);            % error for vanilla case BrasCPD
        error(i+1) = frob(T - cpdgen(A_est))/frob(T);                      % error for accelerated scheme
        error_Adaccel(i+1) = frob(T - cpdgen(A_est_Adaccel))/frob(T);      % error for adaptive scheme with acceleration
        
        semilogy(error)
        hold on;
        semilogy(error_Bras,'g')
        hold on;
        semilogy(error_Adaccel,'r')
        hold off;
        xlabel('epochs');
        ylabel('error');
        legend('Bras accel','Bras', 'Adaccel Bras');
        grid on;
        pause(0.001)
    end
    
    iter = iter + 1;

end
% file_name = ['i' '_' num2str(scale) '_' 'R' '_' num2str(R) 'stepAd'];
% saveas(fig1,[file_name '.pdf']);
