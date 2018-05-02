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

%KMEANS INIT
%rand('state',10)%outside the loop to reproduce the results
rng('default')
runs = 20;
itRMSEtr(runs,1) = 0;
itRMSEva(runs,1) = 0;
itRMSEte(runs,1) = 0;
Ninit = 149;
Nfinal = Ninit;


N = Ninit;
rng(N)
for runs=1:20

tic

         
        
         %RBF center selection (3)
        
         %3. kmeans
         [idx, c] = kmeans(xtrnorm,N);  
         %silhouette(xtrnorm,idx)
        % [SIGMA] = Pnn(N,c);
          SIGMA = (1-0.5).*rand(1,N)+0.5;

         cinitial = c;

        %% PHI matrix computation TRAINING OBJECTIVE
         [phi] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,SIGMA);
         
         [Q1, Q2, R1] = QRfactorization(phi, data, N+1);    

         %ekpaideysh me training
         w = R1\(Q1'*ytrnorm);
         
       %% RMSE                
                itytruetrnorm=phi*w;
                [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
             
                itRMSEtr(runs,1)= sqrt(mean((ytr-itytruetr).^2));   
              %% VALIDATION data
                [itphiva] = phi_fun(c,xvanorm,N, size(xva,1), size(xva,2),ytrnorm,SIGMA);
                
                itytruevanorm=itphiva*w;
                [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));

                itRMSEva(runs,1)= sqrt(mean((yva-itytrueva).^2));   
              %% TESTING data
                [itphite] = phi_fun(c,xtenorm,N, size(xte,1), size(xte,2),ytrnorm,SIGMA);
                
                itytruetenorm=itphite*w;
                [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));

                itRMSEte(runs,1)= sqrt(mean((yte-itytruete).^2)); 

time(runs,1) = toc;
end

% 
% 
% %kmeans initialization results
% %best va & te on va, tr on va
% %best centers & LM
[bva, indexbestva] = min(itRMSEva);

bva
bte_va=itRMSEte(indexbestva)
btr_va=itRMSEtr(indexbestva)

% 
% %mean std
% %validation
meanVA=mean(itRMSEva)
stdVA=std(itRMSEva)

% %testing
meanTE=mean(itRMSEte)
stdTE=std(itRMSEte)
% 
% %training
meanTR=mean(itRMSEtr)
stdTR=std(itRMSEtr)

%time
besttime = time(indexbestva)
meantime=mean(time)
stdtime=std(time)
% 
% plot(itRMSEtr)
% hold on
% plot(itRMSEva)
% plot(itRMSEte)
% legend('training','validation','testing')
% xlabel('run')
% ylabel('RMSE')
% %set(gca, 'YScale', 'log')
% 
% figure
% plot(w)
% xlabel('RBF')
% ylabel('weights')

