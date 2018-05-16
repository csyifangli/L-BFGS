%removing all variables from the current workspace
clear
%% datasets
%training dataset
load('training_data_set1.mat');
%validation dataset
load('validation_data_set2.mat');
%testing dataset
load('testing_data_set3.mat');

%number of examples
data = size(xtr,1);
%number of inputs
no_var = size(xtr,2);

%% normalization
%normalization limits [a,b]
a = -1; b = 1;

%training dataset
[xtrnorm, allmaxxtr, allminxtr] = normalizationx(xtr, size(xtr,2), a, b);
[ytrnorm, maxvarytr, minvarytr] = normalizationy(ytr, a, b);

%validation dataset
[xvanorm] = normalization_val_testing(xva, size(xva,2), a, b, allmaxxtr, allminxtr);

%testing dataset
[xtenorm] = normalization_val_testing(xte, size(xte,2), a, b, allmaxxtr, allminxtr);

%number of runs
runs = 1;
%initializing matrices, vectors
RMSEtr_va(1:runs,1) = 0;
minRMSEvaRUNS(1:runs,1) = 0;
RMSEte_va(1:runs,1) = 0;
LMitRUNS(1:runs,1) = 0;
time(1:runs,1) = 0;
it(1:runs,1) = 0;
indx(1:runs,1:2) = 0;


 %% RBF center selection
 %fuzzy means
 [c, N] = SFMfunction(no_var,data,xtrnorm,10);

 wruns(1:N+1,1:runs) = 0;
 wrunsinitial(1:N+1,1:runs) = 0;

 SIGMAruns(1:runs,1:N) = 0;
 SIGMArunsinitial(1:runs,1:N) = 0;
    
%controlling the seed
%outside the loop to reproduce the results
rng('default')
rng(N)

%counter, centers (all runs)
init = 1;
fin = no_var;
for runs=1:1
    %reset the vector of RMSE in the validation dataset
    itRMSEva = 0;
    
    %stopwatch timer
    tic
    
    %% RBF center selection
    %fuzzy means
    [c, N] = SFMfunction(no_var,data,xtrnorm,10);
    cinitial = c;
           
    %% sigma ( P-nearest neighbors )
    [SIGMA] = Pnn(N,c);
    SIGMAinitial = SIGMA;
    
    %% PHI matrix (m x N)
    % ------> (N <- N + 1, bias term) <------
    [phi] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);  
    [Q1, Q2, R1] = QRfactorization(phi, data, N);
    
    %% initial weights ( Linear Least Square Solution )
    w = R1\(Q1'*ytrnorm);                    
    winitial = w;

    %% initial cost
    hh(1,1) = 0.5*norm(Q2'*ytrnorm)^2;
    hha(1,1) = 0.5*norm(phi*w-ytrnorm)^2;
            
    %% RMSE
    %% training dataset
    itytruetrnorm = phi*w;
    
    [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
    RMSEtr(1,1) = sqrt(mean((ytr-itytruetr).^2));
    
    %% validation dataset
    [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA); 
    itytruevanorm = itphiva*w;
    
    [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1)); 
    RMSEva(1,1)= sqrt(mean((yva-itytrueva).^2));
    
    %% testing dataset
    [itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),SIGMA); 
    itytruetenorm = itphite*w;
    
    [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1)); 
    RMSEte(1,1)= sqrt(mean((yte-itytruete).^2));
                    
    %% save for each run
    itRMSEvaruns{:,runs}=itRMSEva(:,1);
    %training
    RMSEtr_va(runs,1)=RMSEtr;
    %validation
    minRMSEvaRUNS(runs,1)=RMSEva;
    %testing
    RMSEte_va(runs,1)=RMSEte;
    %weights
    wruns(:,runs) = w;
    wrunsinitial(:,runs) = winitial;
    %widths
    SIGMAruns(runs,:) = SIGMA;
    SIGMArunsinitial(runs,:) = SIGMAinitial;
    %centers
    cruns(:,init:fin) = c;
    crunsinitial(:,init:fin) = cinitial;
    %counters
    indx (runs,1)= init;
    indx (runs,2) = fin;
    init = fin + 1;
    fin = fin + no_var;
    %it = number of successful iterations (r>0) needed
    %to find the min RMSE in the validation dataset
    [miin,indexminn]=min(round(itRMSEva,4));
    %in each run
    it(runs,1) = indexminn-1;
    time(runs,1) = toc;
end%runs
%% final best values from the 20 runs
[bva, indexbestva] = min(minRMSEvaRUNS);
%% RMSE
%best RMSE in the validation
bva
%testing
bte_va=RMSEte_va(indexbestva)
%training
btr_va=RMSEtr_va(indexbestva)
%LM iterations
% bit=it(indexbestva)
%RMSE path in the validation
% bitRMSEva = itRMSEvaruns(:,indexbestva);
% bitRMSEmat = cell2mat(bitRMSEva);
% %% parameters
%weights
bwinitial = wrunsinitial(:,indexbestva);
bwfinal = wruns(:,indexbestva);
%widths
bSIGMArunsinitial = SIGMArunsinitial(indexbestva,:);
bSIGMAfinal = SIGMAruns(indexbestva,:);
%centers
bindx = indx(indexbestva,:);
bcfinal = cruns(:,bindx(1,1):bindx(1,2));
bcinitial = crunsinitial(:,bindx(1,1):bindx(1,2));

% % % mean & std
% % LM iterations
% % meanitRUNS = round(mean(it))
% % stditRUNS = round(std(it))
% % 
% % RMSE validation
% % meanVA = mean(minRMSEvaRUNS)
% % stdVA = std(minRMSEvaRUNS)
% % 
% % RMSE testing
% % meanTE = mean(RMSEte_va)
% % stdTE = std(RMSEte_va)
% % 
% % RMSE training
% % meanTR = mean(RMSEtr_va)
% % stdTR = std(RMSEtr_va)

%time
besttime = time(indexbestva)
% meantime = mean(time)
% stdtime = std(time)
%% FINAL PARAMETERS BY THE BEST RUN
SIGMA = bSIGMAfinal;
SIGMAinitial = bSIGMArunsinitial;
c = bcfinal;
cinitial = bcinitial;
w = bwfinal;
winitial = bwinitial;
%% check
%% TRAINING DATASET

[itphitr] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);
itytruetrnorm=itphitr*w;

[itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));

RMSEtrfinal= sqrt(mean((ytr-itytruetr).^2));
%% VALIDATION DATASET

[itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA);
itytruevanorm=itphiva*w;

[itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));
RMSEvafinal= sqrt(mean((yva-itytrueva).^2));
%% TESTING DATASET

[itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),SIGMA);
itytruetenorm=itphite*w;

[itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));
RMSEtefinal = sqrt(mean((yte-itytruete).^2));