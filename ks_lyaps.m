% Computes the Lyapunov Exponents of the Kuramoto-Sivashinsky equation with
% a multiplicative asymmetry term that breaks spatial homogeneity.
% See Chaos 27, 121102 (2017) Pathak et al. (referred to as REF 1 in comments in this code)
% Code for basic KS PDE integration adapted from Kassam, Trefethen
% SIAM J. Sci. Comput., 26(4), 1214–1233 
% https://epubs.siam.org/doi/10.1137/S1064827502410633
% Lyapunov spectrum calculation and modifications to the KS system
% implemented by Jaideep Pathak. See Chaos 27, 121102 (2017) Pathak et al
% for details. Email: jpathak@umd.edu


clear;
d = 60;  %domain size/periodicity length (denoted by L in Pathak et al)
N = 128; % discretization grid size (denoted by Q)
    
x = d*(-N/2+1:N/2)'/N;


rng('shuffle')
delta = 0.1;  % \mu in Eq. (7) REF 1. (set delta = 0 for the standard spatially homogeneous KS equation)
wavelength = d/4;  % \lambda in Eq. (7) REF 1. (sets the spatial inhomgeneity wavelength)
omega = 2*pi/wavelength;
p = delta.*cos(omega.*x);
px = -omega*delta.*sin(omega.*x);
pxx = -(omega^2).*p;


u = 0.6*(-1+2*rand(size(x)));
v = fft(u);

num_lyaps = 40;  

Y = orth(rand(N,num_lyaps));

h = 1/4;
k = [0:N/2-1 0 -N/2+1:-1]'*(2*pi/d);
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);


Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));
tt = 0;
tmax = 4000; nmax = round(tmax/h); nplt = 1;%floor((tmax/10000)/h);
g = -0.5i*k;
norm_steps = 10;

vv = zeros(N, nmax);
Rii = zeros(num_lyaps, nmax/norm_steps);


vv(:,1) = v;
transient = 1000;
for n = 1:transient
t = n*h;
rifftv = real(ifft(v));
Nv = g.*fft(rifftv.^2) + 2i*k.*fft(rifftv.*px) - fft(rifftv.*pxx) + k.^2.*fft(rifftv.*p);
a = E2.*v + Q.*Nv;
riffta = real(ifft(a));
Na = g.*fft(riffta.^2) + 2i*k.*fft(riffta.*px) - fft(riffta.*pxx) + k.^2.*fft(riffta.*p);
b = E2.*v + Q.*Na;
rifftb = real(ifft(b));
Nb = g.*fft(rifftb.^2) + 2i*k.*fft(rifftb.*px) - fft(rifftb.*pxx) + k.^2.*fft(rifftb.*p);
c = E2.*a + Q.*(2*Nb-Nv);
rifftc = real(ifft(c));
Nc =  g.*fft(rifftc.^2) + 2i.*k.*fft(rifftc.*px) - fft(rifftc.*pxx) + k.^2.*fft(rifftc.*p);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
end

count = 0;


for n=1:nmax

    %Evolve KS 
    
t = n*h;
rifftv = real(ifft(v));
Nv = g.*fft(rifftv.^2) + 2i*k.*fft(rifftv.*px) - fft(rifftv.*pxx) + k.^2.*fft(rifftv.*p);
a = E2.*v + Q.*Nv;
riffta = real(ifft(a));
Na = g.*fft(riffta.^2) + 2i*k.*fft(riffta.*px) - fft(riffta.*pxx) + k.^2.*fft(riffta.*p);
b = E2.*v + Q.*Na;
rifftb = real(ifft(b));
Nb = g.*fft(rifftb.^2) + 2i*k.*fft(rifftb.*px) - fft(rifftb.*pxx) + k.^2.*fft(rifftb.*p);
c = E2.*a + Q.*(2*Nb-Nv);
rifftc = real(ifft(c));
Nc =  g.*fft(rifftc.^2) + 2i.*k.*fft(rifftc.*px) - fft(rifftc.*pxx) + k.^2.*fft(rifftc.*p);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
vv(:,n) = v;


    %Evolve tangent map of KS


Hv = bsxfun(@times,-1i*k,fft(bsxfun(@times,real(ifft(v)),real(ifft(Y))))) + ...
     bsxfun(@times,+1i*k,fft(bsxfun(@times,px,real(ifft(Y))))) + ...
     -fft(bsxfun(@times,pxx,real(ifft(Y))));
a_ = bsxfun(@times,E2,Y) + bsxfun(@times,Q,Hv);
Ha = bsxfun(@times,-1i*k,fft(bsxfun(@times,real(ifft(v)),real(ifft(a_))))) + ...
     bsxfun(@times,+1i*k,fft(bsxfun(@times,px,real(ifft(Y))))) + ...
     -fft(bsxfun(@times,pxx,real(ifft(Y))));
b_ = bsxfun(@times,E2,Y) + bsxfun(@times,Q,Ha);
Hb = bsxfun(@times,-1i*k,fft(bsxfun(@times,real(ifft(v)),real(ifft(b_))))) + ...
     bsxfun(@times,+1i*k,fft(bsxfun(@times,px,real(ifft(Y))))) + ...
     -fft(bsxfun(@times,pxx,real(ifft(Y))));
c_ = bsxfun(@times,E2,a_) + bsxfun(@times,Q,(2*Hb-Hv));
Hc = bsxfun(@times,-1i*k,fft(bsxfun(@times,real(ifft(v)),real(ifft(c_))))) + ...
     bsxfun(@times,+1i*k,fft(bsxfun(@times,px,real(ifft(Y))))) + ...
     -fft(bsxfun(@times,pxx,real(ifft(Y))));
Y = bsxfun(@times,E,Y) + bsxfun(@times,Hv,f1) + bsxfun(@times,2*(Ha+Hb),f2) + bsxfun(@times,Hc,f3);

    % Normalize tangent vectors and record normalization

if mod(n,norm_steps) == 0
    count = count+1;
    [matQ, matR] = qr(real(ifft(Y)),0);
    Rii(:,count) = log(diag(matR(1:num_lyaps,1:num_lyaps)));
    Y = fft(matQ(:,1:num_lyaps));

end

end

complex_spectrum = sum(Rii, 2)./(nmax*h); 

spectrum = real(complex_spectrum);

spec_sum = zeros(num_lyaps,1);

for i = 1:num_lyaps
    spec_sum(i) = sum(spectrum(1:i));
end

abs_spec_sum = abs(spec_sum);
ky_point = min(abs_spec_sum);
kydim = find(spec_sum<0, 1,'first');


figure()
plot(spectrum, '+')
refline([0,0])
filename = strcat('L', num2str(d),'asym_m_delta', num2str(100*delta), 'wl', num2str(wavelength), 'numl', num2str(num_lyaps));
%save(filename, 'spectrum', 'spec_sum', 'ky_point', 'kydim', 'Rii', 'delta', 'wavelength');
