%% Plot Logisitc Cost Function
%  Authors: Ioanna Siaminou, Giannis Papagiannakos, Chris Kolomvakis
%  date: 7/06/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
clear ; close all; clc

addpath('Libraries/cvx')

%% Create Data
m = 50;
n = 2;
lambda = 1;

% (non-separable)
%Ground truth 
x_star = zeros(n, 1);
A = 3*randn(m,n);
% Measurements in y
y = (rand(m,1) < exp(A*x_star)./(1+exp(A*x_star)));


% (fully separable)
% Ground truth 
% x_star = ones(n, 1);				
% c = 10*sign(randn);
% A = [c*ones(m/2,n); -c*ones(m/2,n)];
% A = A + 5*randn(m,n);
% threshold = 0;
% param = A*x_star > threshold;
% %Measurements in y
% y = param;
%% Initial values
x_init = rand(n, 1);

% log loss
param = A*x_init;
sigm_x = (1./(1 + exp(-param)));
log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));

% initial cost function value
J_init = sum(log_loss)/m + (lambda/(2*m))*norm(x_init)^2;

dt = 0.01;
x1 = -1:dt:1;
x2 = -1:dt:1;

[X1,X2] = meshgrid(x1,x2);

for i =1:length(x1)
    for j = 1:length(x2)
        x = [x1(i) x2(j)]';
        % log loss
        param = A*x;
        sigm_x = (1./(1 + exp(-param)));
        log_loss = -y.*(log(sigm_x)) - (1 - y).*(log(1 - sigm_x));
        J(i,j) = sum(log_loss)/m + (lambda/(2*m))*norm(x)^2;
    end
end

figure()
mesh(X1, X2, J)

