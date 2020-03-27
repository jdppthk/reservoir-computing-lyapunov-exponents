function w = adj_mat_uniform(size, radius, sparsity, jobid)

%w = zeros(size,'gpuArray');
%w = zeros(size);
rng(jobid);

w = sprand(size,size,sparsity);

e = eigs(w);
e = abs(e);
w = (w./max(e))*radius;