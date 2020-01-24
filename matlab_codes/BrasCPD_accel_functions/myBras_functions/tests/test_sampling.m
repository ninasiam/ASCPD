rng(0);
order = 3;
I = 15;
J = 15;
K = 15;

dims = [I J K];

display('Run with...');
R = 5%randi([10 min(dims)-10],1,1)
scale = 3;%randi([10 15],1,1);                                             % parameter to control the blocksize
B = scale*[1 1 1]                                                       % can be smaller than rank

% I_cal = {1:dims(1), 1:dims(2), 1:dims(3)};

                                                       % Max number of iterations (epochs
%% create true factors 
for ii = 1:order
    A_true{ii} = rand(dims(ii),R);
end

T = cpdgen(A_true)

for k=1:K
    X(:,:,k)=A_true{1}*diag(A_true{3}(k,:))*A_true{2}';
end
X_data = tensor(X);

idxs = zeros(B(1), order);

n = 2;

[ idxs, factor_idxs, T_s ] = sample_fbrs( T, n, dims, B(n) );

[tensor_idx, factor_idx]  = sample_fibers2(B(n), dims, n)


X_sample = reshape(X_data(tensor_idx), dims(n), [])'

kr_idxs = [1, 3];
A_kr = A_true(kr_idxs);
H_s = sampled_khatri_rao(A_kr, factor_idxs);
A = {A_true{1} A_true{3}};
H_sample = sampled_kr(A,factor_idx)
