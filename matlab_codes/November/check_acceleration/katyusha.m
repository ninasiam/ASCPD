function [ x_KAT_HAT, f_KAT ] = katyusha(A, b, f_star, tol, MAX_OUTER_ITER)
%%Katyusha
%Initializations

n = size(A,1);
d = size(A,2);
m = 2*d;


x_KAT = zeros(d,1);
y = x_KAT;
z = x_KAT;
x_KAT_HAT = x_KAT;


iter = 1;

f_KAT = (1/(2*n))*norm(A*x_KAT - b)^2;

A_T_A = A'*A;
A_T_b = A'*b;

L = max(eig(A_T_A))/n;
sigma =  min(eig(A_T_A))/n;

t_2 = 0.5;
t_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
alpha = 0.000001%1/(3*t_1*L);

while(1)


    mu_s = (1/(n))*(A_T_A*x_KAT_HAT-A_T_b); %full gradient

    for j = 1:m

        x_next = t_1*z + t_2*x_KAT_HAT + (1 - t_1 - t_2)*y;

        i = randi(n,1);

        stoch_grad = mu_s + (A(i,:)*x_next - b(i)).*A(i,:)' - (A(i,:)*x_KAT_HAT - b(i)).*A(i,:)';

        z_next = z - alpha*stoch_grad;

        y_next = x_next - (1/(3*L))*stoch_grad;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        y = y_next;
        z = z_next;
    end
    
    x_KAT_HAT = y_next;


    if(f_KAT(end) - f_star < tol || iter > MAX_OUTER_ITER )
        break;
    end
    if((mod(iter, m) == 0))
        f_KAT = [f_KAT 1/(2*n)* norm(A * x_KAT_HAT - b)^2];
    end
    iter = iter + 1


end

end

