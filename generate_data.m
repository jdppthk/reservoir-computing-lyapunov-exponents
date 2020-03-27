% Integrates the Kuramoto-Sivashinsky equation with
% a multiplicative asymmetry term that breaks spatial homogeneity. Creates
% data for training a recurrent neural network to infer the Lyapunov
% exponents using data alone (without access to equations). (RNN training
% code is in a separate file.)
% See Chaos 27, 121102 (2017) Pathak et al. (referred to as REF 1 in comments in this code)
% Code for basic KS PDE integration adapted from Kassam, Trefethen
% SIAM J. Sci. Comput., 26(4), 1214–1233 
% https://epubs.siam.org/doi/10.1137/S1064827502410633
% Lyapunov spectrum calculation and modifications to the KS system
% implemented by Jaideep Pathak. See Chaos 27, 121102 (2017) Pathak et al
% for details. Email: jpathak@umd.edu

clear;
N = 128;
d = 60;
x = d*(-N/2+1:N/2)'/N;

delta = 0.0;
wavelength = d/4;
omega = 2*pi/wavelength;
p = delta.*cos(omega.*x);
px = -omega*delta.*sin(omega.*x);
pxx = -(omega^2).*p;


u = 0.6*(-1+2*rand(size(x)));
v = fft(u);
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
tmax =25000; nmax = round(tmax/h); nplt = 1;%floor((tmax/10000)/h);
g = -0.5i*k;

vv = zeros(N, nmax);

vv(:,1) = v;

for n = 1:nmax
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
end

uu = transpose(real(ifft(vv)));



fig2 = figure('pos',[5 270 600 200],'color','w');
imagesc(transpose(uu))
shading flat
colormap(jet);
colorbar;


save(['mult_asym_kursiv_delta' num2str(100*delta) 'wl' num2str(wavelength) '.mat'], 'uu', 'd', 'wavelength', '-v7.3');


