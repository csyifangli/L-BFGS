clear
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


runs = 20;
itRMSEtr(runs,1) = 0;
itRMSEva(runs,1) = 0;
itRMSEte(runs,1) = 0;

Nin=30;
Nfin=80;
rng('default')

centers = 1;
for N = Nin:Nfin
    rng(N)

    for runs=1:20
        tic 
        %kmeans
        [idx, c] = kmeans(xtrnorm,N);    
        [SIGMA] = Pnn(N,c);  
        cinitial = c;
        %% PHI
        [phi] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);    
        [Q1, Q2, R1] = QRfactorization(phi, data, N);
        %weights
        w = R1\(Q1'*ytrnorm);   
        %% RMSE                 
        itytruetrnorm=phi*w; 
        [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));        
        itRMSEtr(runs,1)= sqrt(mean((ytr-itytruetr).^2));     
        %% VALIDATION data           
        [itphiva] = phi_fun(c,xvanorm,N, size(xva,1), size(xva,2),SIGMA);   
        itytruevanorm=itphiva*w;  
        [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));    
        itRMSEva(runs,1)= sqrt(mean((yva-itytrueva).^2));          
        %% TESTING data  
        [itphite] = phi_fun(c,xtenorm,N, size(xte,1), size(xte,2),SIGMA);  
        itytruetenorm=itphite*w;     
        [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));   
        itRMSEte(runs,1)= sqrt(mean((yte-itytruete).^2)); 
        time(runs,1) = toc; 
    end   
    %% best run
    [bva, indexbestva] = min(itRMSEva);
    %
    bvacenters(centers,1)=bva;
    bvameancenters(centers,1)=mean(itRMSEva);
    bvastdcenters(centers,1)=std(itRMSEva);
    %
    bte_vacenters(centers,1)=itRMSEte(indexbestva);
    btemeancenters(centers,1)=mean(itRMSEte);
    btestdcenters(centers,1)=std(itRMSEte);
    %
    btr_vacenters(centers,1)=itRMSEtr(indexbestva);
    btrmeancenters(centers,1)=mean(itRMSEtr);
    btrstdcenters(centers,1)=std(itRMSEtr);
    %
    timecenters(centers,1)=time(indexbestva);
    meancenters(centers,1)=mean(time);
    stdcenters(centers,1)=std(time);
    %
    centers = centers+1;
end
%% best N
[bestvamean, ind] = min(bvameancenters);
%
bestva = bvacenters(ind)
bestte = bte_vacenters(ind)
besttr = btr_vacenters(ind)
%
bestvamean
bvastd = bvastdcenters(ind)
%
btemean = btemeancenters(ind)
btestd = btestdcenters(ind)
%
btrmean = btrmeancenters(ind)
btrstd = btrstdcenters(ind)
%
RBFs = Nin + ind - 1
%
tim = timecenters(ind)
mea = meancenters(ind)
st = stdcenters(ind)