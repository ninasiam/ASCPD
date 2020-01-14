clear;
clc;
close all

% add paths (home)
addpath('/home/nina/Documents/uni/codes/matlab/master/demo_vol1/functions')
addpath('/home/nina/Documents/uni/nina_s/matlab_codes/December/myBras_functions');
addpath('/home/nina/Documents/uni/Libraries/Tensor_lab');

% add paths (local-dell)
% addpath('/home/telecom/Desktop/nina/matlab_codes/functions');
% addpath('/home/telecom/Desktop/nina/nina_s/matlab_codes/December/myBras_functions');
% addpath('/home/telecom/Documents/Libraries/tensorlab_2016-03-28');

%% Initializations
order = 3;
I = 100;
J = 100;
K = 100;

dims = [I J K];

R = 50%randi([10 min(dims)-10],1,1)
scale = 2;%randi([10 15],1,1);                                             % parameter to control the blocksize
B = scale*[10 10 10]                                                       % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

MAX_OUTER_ITER = 10;                                                       % Max number of iterations (epochs)


%% create true factors 
for ii = 1:order
    A_true{ii} = rand(dims(ii),R);
end

T = cpdgen(A_true);

%% create initial point
for jj = 1:order
    A_init{jj} = rand(dims(jj),R);
end

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
options.ratio_var = 0;
[A_est, MSE, error] = BrasCPD_vol2(T,options);

%BrasCPD ratio 1 
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
options.proximal = 'true';
options.ratio_var = 1;

[A_est_r1, MSE_r1, error_r1] = BrasCPD_vol2(T,options);

%BrasCPD ratio 10 
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
options.proximal = 'true';
options.ratio_var = 10;

[A_est_r10, MSE_r10, error_r10] = BrasCPD_vol2(T,options);

%BrasCPD ratio 100
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
options.proximal = 'true';
options.ratio_var = 100;

[A_est_r100, MSE_r100, error_r100] = BrasCPD_vol2(T,options);


%BrasCPD ratio 1000
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
options.proximal = 'true';
options.ratio_var = 1000;

[A_est_r1000, MSE_r1000, error_r1000] = BrasCPD_vol2(T,options);


%BrasCPD ratio 10000
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
options.proximal = 'true';
options.ratio_var = 10000;

[A_est_r10000, MSE_r10000, error_r10000] = BrasCPD_vol2(T,options);


%% plot
figure(1)
semilogy([0:(size(MSE,2)-1)],MSE,'->y','linewidth',1.5);hold on
semilogy([0:(size(MSE2,2)-1)],MSE2,'->b','linewidth',1.5); hold on
semilogy([0:(size(MSE_r1,2)-1)],MSE_r1,'->k','linewidth',1.5);hold on
semilogy([0:(size(MSE_r10,2)-1)],MSE_r10,'->m','linewidth',1.5); hold on
semilogy([0:(size(MSE_r100,2)-1)],MSE_r100,'->c','linewidth',1.5);hold on
semilogy([0:(size(MSE_r1000,2)-1)],MSE_r1000,'->g','linewidth',1.5); hold on;
semilogy([0:(size(MSE_r10000,2)-1)],MSE_r10000,'->r','linewidth',1.5); 
legend('BrasCP accel','BrasCP', 'BrasCP ratio 1', 'BrasCP ratio 10', 'BrasCP ratio 100', 'BrasCP ratio 1000', 'BrasCP ratio 10000');
xlabel('no. of MTTKRP computed')
ylabel('MSE')
set(gca,'fontsize',14)
grid on

figure(2)
semilogy([0:(size(error,2)-1)],error,'->y','linewidth',1.5);hold on
semilogy([0:(size(error2,2)-1)],error2,'->b','linewidth',1.5); hold on
semilogy([0:(size(error_r1,2)-1)],error_r1,'->k','linewidth',1.5);hold on
semilogy([0:(size(error_r10,2)-1)],error_r10,'->m','linewidth',1.5); hold on
semilogy([0:(size(error_r100,2)-1)],error_r100,'->c','linewidth',1.5);hold on
semilogy([0:(size(error_r1000,2)-1)],error_r1000,'->g','linewidth',1.5); hold on;
semilogy([0:(size(error_r10000,2)-1)],error_r10000,'->r','linewidth',1.5); 
legend('BrasCP accel','BrasCP', 'BrasCP ratio 1', 'BrasCP ratio 10', 'BrasCP ratio 100', 'BrasCP ratio 1000', 'BrasCP ratio 10000');
xlabel('no. of MTTKRP computed')
ylabel('error')
set(gca,'fontsize',14)
grid on