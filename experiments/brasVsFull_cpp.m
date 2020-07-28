close all

%% Test: Full CPD vs BrasCPD (CPP)
% "Compare the two methods in respect to the #mttkrps"
% -- 28 / 07 / 2020 --
% * Non-Negative
% * 500 x 500 x 500
% * rank = 50
% * #iterations = 10 
% * #MTTKRPs = 10 * (ORDER + 1)

% *** FULL ***
time_total_full  = mean([14.395,            14.5469,          14.1759]);
% mttkrp + accel (accel = order x updates + 1 x mttkrp + 1 x compute cost fun) 
time_mttkrp_full = mean([10.6371 + 2.91497, 10.758 + 2.92894, 10.4739 + 2.8768]);
f_val_full       = 0.0272275;

% *** Block size = 100 ***
time_total_bras_100  = mean([105.382, 101.963, 105.208]);
time_mttkrp_bras_100 = mean([12.271,  12.0064, 12.1532]);
f_val_bras_100       = mean([10^-15,  10^-15,  10^-16]);

% *** Block size = 200 ***
time_total_bras_200  = mean([81.7775, 84.5416, 86.8842]);
time_mttkrp_bras_200 = mean([11.2811, 11.5009, 11.8038]);
f_val_bras_200       = mean([10^-15,  10^-16,  10^-16]);

% *** Block size = 500 ***
time_total_bras_500  = mean([67.9591, 68.7636, 67.2671]);
time_mttkrp_bras_500 = mean([10.7913, 10.8716, 10.6599]);
f_val_bras_500       = mean([10^-15,  10^-16,  10^-16]);

x_axis = categorical({'all', '100', '200', '500'});

y_axis = [time_total_full time_total_bras_100 time_total_bras_200 time_total_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('total time (s)')
title('Tensor of dims=500x500x500, rank=50, non-negative')

y_axis = [time_mttkrp_full time_mttkrp_bras_100 time_mttkrp_bras_200 time_mttkrp_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('mttkrp time (s)')
title('Tensor of dims=500x500x500, rank=50, non-negative')

y_axis = time_total_full./[time_total_full time_total_bras_100 time_total_bras_200 time_total_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('total speedup')
title('Tensor of dims=500x500x500, rank=50, non-negative')

y_axis = time_mttkrp_full./[time_mttkrp_full time_mttkrp_bras_100 time_mttkrp_bras_200 time_mttkrp_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('mttkrp speedup')
title('Tensor of dims=500x500x500, rank=50, non-negative')

figure()
grid on;
semilogy(f_val_full,'x')
hold on;
semilogy(f_val_bras_100, 'o')
hold on;
semilogy(f_val_bras_200, '*')
hold on;
semilogy(f_val_bras_500, '+')
hold off;
legend('all', 'block size=100', 'block size=200', 'block size=500')
ylabel('accuracy')
title('Tensor of dims=500x500x500, rank=50, non-negative')

%%
% * Non-Negative
% * 500 x 500 x 500
% * rank = 10
% * #iterations = 10 
% * #MTTKRPs = 10 * (ORDER + 1)

% *** FULL ***
time_total_full  = mean([6.15867,           6.17131,          6.37187]);
% mttkrp + accel (accel = order x updates + 1 x mttkrp + 1 x compute cost fun) 
time_mttkrp_full = mean([4.79377 + 1.33114, 4.78838+ 1.35009, 4.95028 + 1.38769]);
f_val_full       = 0.0272275;

% *** Block size = 100 ***
time_total_bras_100  = mean([48.144, 48.4015, 49.1824]);
time_mttkrp_bras_100 = mean([3.77811, 3.82521, 4.09548]);
f_val_bras_100       = mean([10^-16,  10^-16,  10^-16]);

% *** Block size = 200 ***
time_total_bras_200  = mean([47.4112, 48.3467, 47.6731]);
time_mttkrp_bras_200 = mean([4.09609, 4.08636, 4.08436]);
f_val_bras_200       = mean([10^-16,  10^-16,  10^-15]);

% *** Block size = 500 ***
time_total_bras_500  = mean([48.6752, 47.4416, 47.6852]);
time_mttkrp_bras_500 = mean([4.19114, 4.06593, 4.10875]);
f_val_bras_500       = mean([10^-16,  10^-16,  10^-16]);

x_axis = categorical({'all', '100', '200', '500'});

y_axis = [time_total_full time_total_bras_100 time_total_bras_200 time_total_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('total time (s)')
title('Tensor of dims=500x500x500, rank=10, non-negative')

y_axis = [time_mttkrp_full time_mttkrp_bras_100 time_mttkrp_bras_200 time_mttkrp_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('mttkrp time (s)')
title('Tensor of dims=500x500x500, rank=10, non-negative')

y_axis = time_total_full./[time_total_full time_total_bras_100 time_total_bras_200 time_total_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('total speedup')
title('Tensor of dims=500x500x500, rank=10, non-negative')

y_axis = time_mttkrp_full./[time_mttkrp_full time_mttkrp_bras_100 time_mttkrp_bras_200 time_mttkrp_bras_500];
figure()
bar(x_axis, y_axis)
xlabel('block size')
ylabel('mttkrp speedup')
title('Tensor of dims=500x500x500, rank=10, non-negative')

figure()
grid on;
semilogy(f_val_full,'x')
hold on;
semilogy(f_val_bras_100, 'o')
hold on;
semilogy(f_val_bras_200, '*')
hold on;
semilogy(f_val_bras_500, '+')
hold off;
legend('all', 'block size=100', 'block size=200', 'block size=500')
ylabel('accuracy')
title('Tensor of dims=500x500x500, rank=10, non-negative')
