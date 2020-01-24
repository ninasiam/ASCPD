clear;
clc;
close all

% add paths (home)
% addpath('/home/nina/Documents/uni/codes/matlab/master/demo_vol1/functions')
% addpath('/home/nina/Documents/uni/nina_s/matlab_codes/December/myBras_functions');
% addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');

% add paths (local-dell)
addpath('/home/telecom/Desktop/nina/matlab_codes/functions');
addpath('/home/telecom/Desktop/nina/nina_s/matlab_codes/December/myBras_functions');
addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');

%% Initializations
order = 3;
I = 100;
J = 100;
K = 100;

dims = [I J K];

R = randi([60 min(dims)-10],1,1)
scale = randi([10 15],1,1);                                                % parameter to control the blocksize
B = scale*[10 10 10]                                                       % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

MAX_OUTER_ITER = 50;                                                       % Max number of iterations (epochs)


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

%BrasCPD accel
options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'on';
options.cyclical = 'off';
options.proximal = 'off';

[A_est, MSE, error] = BrasCPD_vol2(T,options);

%BrasCPD 
options.A_true = A_true;
options.A_init = A_init;
options.constraint{1} = 'nonnegative';
options.constraint{2} = 'nonnegative';
options.constraint{3} = 'nonnegative';
options.max_iter = floor(dims(1)*dims(2)/B(1))*MAX_OUTER_ITER; 
options.bz = B(1);
options.tol = eps^2;
options.tol_rel = 10^(-5);
options.alpha0 = 0.1;
options.dims = dims;
options.acceleration = 'off';
options.cyclical = 'off';
options.proximal = 'off';

[A_est2, MSE2, error2] = BrasCPD_vol2(T,options);

%% plot
figure(1)
semilogy([0:(size(MSE,2)-1)],MSE,'-sb','linewidth',1.5);hold on
semilogy([0:(size(MSE2,2)-1)],MSE2,'->m','linewidth',1.5);
legend('BrasCP accel','BrasCP');
xlabel('no. of MTTKRP computed')
ylabel('MSE')
set(gca,'fontsize',14)
grid on
  
