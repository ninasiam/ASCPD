
clear, clc

n = 2; % data dimension
N = 50;  % number of data points

% Hyperplane
w = 1 * randn(2,1);
x_axis = [-3:.1:3];
y_axis = w(1)/w(2) * x_axis;
figure(1), plot(x_axis, y_axis), hold on

% Data generation
mu1 = [-w(2); w(1)];
mu2 = -mu1;
for ii=1:N
     rnd=rand;
     if rnd>0.5
          x(:,ii) = mu1 + 2* randn(2,1);
          if (w'*x(:,ii) > 0), y(ii)=1; else y(ii)=0; end
     else
         x(:,ii) = mu2 + 2* randn(2,1);
         if (w'*x(:,ii) > 0), y(ii)=1; else y(ii)=0; end
     end
end
figure(1), plot(x(1,:), x(2,:), 'o'), hold off



% Computation of the cost function
w_range=10;
w1_axis=w(1)-w_range:.2:w(1)+w_range;
w2_axis=w(2)-w_range:.2:w(2)+w_range;
lambda = 1;  % regularization parameter
for i1=1:length(w1_axis)
    for i2=1:length(w2_axis)
       w_tmp=[w1_axis(i1); w2_axis(i2)];
       log_regr(i1,i2) = comp_log_regr(w_tmp, x, y, lambda);
    end
end
figure(2), mesh(w1_axis, w2_axis, log_regr)
figure(3), contour(w1_axis, w2_axis, log_regr)

function f = comp_log_regr(w, x, y, lambda)

f = 0;
N = length(y);
for ii=1:N
    f = f + -y(ii) * log(1/(1+exp(-w'*x(:,ii))))  - (1-y(ii)) * log(1 - 1/(1+exp(-w'*x(:,ii))))  - (1-y(ii)) ;
    f = f + lambda/2*norm(w)^2;
end
end