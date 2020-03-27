% Code for training an RNN using reservoir computing and calculating the
% Lyapunov exponents of the RNN dynamics.
% See Chaos 27, 121102 (2017) Pathak et al. (referred to as REF 1 in comments in this code)
% author: Jaideep Pathak. See Chaos 27, 121102 (2017) Pathak et al
% for details. Email: jpathak@umd.edu

clear;

tic;

jobid = 1;
m = matfile('mult_asym_kursiv_delta10wl15'); % generate this file using "generate_data.m"
sigma = 0.5;
data = m.uu;
cdelta = 0.0;
        
LL = m.d;

[~, num_inputs] = size(data);

set_average_degree = 3;

%approx_reservoir_size = 9000; %ideally you need a reservoir size ~9000 to
% accurately replicate the lyapunov spectrum. 3000 will still give a quick
% mostly accurate result.
approx_reservoir_size = 3000;

sparsity = set_average_degree/approx_reservoir_size;

nodes_per_input = round(approx_reservoir_size/num_inputs);

N = num_inputs*nodes_per_input;

rho = 0.4;

W = adj_mat_uniform(N,rho,sparsity,jobid);

x=zeros(N,1);

dl = 10000; % discard transient
tl = 40000; % training length
pl = 10000; % prediction length (for autonomous prediction)

w_in = zeros(N, num_inputs); 

% input coupling mask

q = N/num_inputs;
ip = zeros(q,1);
for i=1:num_inputs
    rng(i)
    ip = sigma*(-1 + 2*rand(q,1));
    w_in((i-1)*q+1:i*q,i) = ip;
end


%transient

for i=1:dl
    x = tanh(W*x + w_in*transpose(data(i,:)));
end

xtrain = zeros(N,tl);
xtrain(:,1)=x;

for i=1:tl-1
    xtrain(:,i+1) = tanh(W*xtrain(:,i) + w_in*transpose(data(dl+i,:)));
end

beta = 0.0001;

idenmat = beta*speye(N);

xtrain(2:2:N,:) = xtrain(2:2:N,:).^2;
w_out = transpose(data(dl+1:dl+tl,:))*transpose(xtrain)*pinv(xtrain*transpose(xtrain)+idenmat); %train 
w_out_2 = zeros(size(w_out)); % see included pdf for details on w_out1 and w_out2
w_out_1 = zeros(size(w_out));
w_out_2(:,2:2:N) = w_out(:,2:2:N);
odd_ind = setdiff(1:1:N, 2:2:N);
w_out_1(:,odd_ind) = w_out(:, odd_ind);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = xtrain(:,tl);

num_lyaps = 40;

delta = orth(rand(N,num_lyaps));

prediction = zeros(pl,num_inputs);
norm_time = 10;
R_ii = zeros(num_lyaps,pl/norm_time);
K1 = W + w_in*w_out_1;
K2 = 2*w_in*w_out_2;
for i=1:pl
    x_augment = x;
    x_augment(2:2:N) = x_augment(2:2:N).^2;
    out = (w_out)*x_augment;
    x_ = x;  
    x = tanh(W*x + w_in*out); % evolve RNN
    delta = bsxfun(@times,(1-x.^2),(K1*delta + K2*(bsxfun(@times,x_,delta)))); % evolve tangent map of RNN (see included pdf)
    prediction(i,:) = out;
    if mod(i,norm_time)==0
        [QQ,RR] = qr(delta,0);
        delta = QQ(:,1:num_lyaps);
        R_ii(:,i/norm_time) = log(diag(RR(1:num_lyaps,1:num_lyaps)));  
    end
end

ll = real(sum(R_ii,2))./(pl*0.25);
%%
filename = strcat('masymLyapsN', num2str(N), '-L-', num2str(LL), '-tl-', num2str(tl/10000), 'delta', num2str(100*cdelta), 'sig', num2str(sigma), '.mat');

% save(filename,'ll','R_ii','rho','LL','jobid','num_lyaps','prediction','tl');


plot(ll, '+')

