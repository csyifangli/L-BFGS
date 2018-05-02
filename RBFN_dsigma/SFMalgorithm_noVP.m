clear
tic

load('training_data_set1.mat');
load('validation_data_set2.mat');
load('testing_data_set3.mat');

data = size(xtr,1);
no_var = size(xtr,2);

a = -1; b = 1;

[xtrnorm, allmaxxtr, allminxtr] = normalizationx(xtr, size(xtr,2), a, b);
[ytrnorm, maxvarytr, minvarytr] = normalizationy(ytr, a, b);

[xvanorm] = normalization_val_testing(xva, size(xva,2), a, b, allmaxxtr, allminxtr);
[xtenorm] = normalization_val_testing(xte, size(xte,2), a, b, allmaxxtr, allminxtr);

k=1;
smin = 5;
smax = 15;
for s = smin:smax
    

    
   %[idx, c] = kmeans(xtrnorm,N);
   [c, N] = SFMfunction(no_var,data,xtrnorm,s);
   centersS(1,k) = N;
      rng('default')


rng(N)
   %[SIGMA] = Pnn(N,c);
   SIGMA = (1-0.5).*rand(1,N)+0.5;

   
    [phi] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,SIGMA);
    %phi(:,N+1) = 1;
      [Q1, Q2, R1] = QRfactorization(phi, data, N+1);    
    w = R1\(Q1'*ytrnorm); 
    
    ytruetrnorm=phi*w;
    [ytruetr] = unnormalization(a, b, minvarytr, maxvarytr, ytruetrnorm, data);
    
    RMSEtr(k)= sqrt(mean((ytr-ytruetr).^2));
    
    
    [phiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),ytrnorm,SIGMA);
    %phiva(:,N+1) = 1;
    
    ytruevanorm=phiva*w;
    [ytrueva] = unnormalization(a, b, minvarytr, maxvarytr, ytruevanorm, size(yva,1));
 
    RMSEva(k)= sqrt(mean((yva-ytrueva).^2));
       
    [phite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),ytrnorm,SIGMA);
    %phite(:,N+1) = 1;   
    
    ytruetenorm=phite*w;
    [ytruete] = unnormalization(a, b, minvarytr, maxvarytr, ytruetenorm, size(yte,1));

     RMSEte(k)= sqrt(mean((yte-ytruete).^2));
     k=k+1;
end

[minRMSEvaa,indexoptVA] = min(RMSEva);

RBFs = centersS(indexoptVA) 
s = smin + indexoptVA - 1
RMSEtr_va = RMSEtr(indexoptVA)
minRMSEva = RMSEva(indexoptVA)
minRMSEte = RMSEte(indexoptVA)

toc