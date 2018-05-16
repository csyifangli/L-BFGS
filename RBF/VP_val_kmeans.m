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
runs = 20;
%initializing matrices, vectors
RMSEtr_va(1:runs,1) = 0;
minRMSEvaRUNS(1:runs,1) = 0;
RMSEte_va(1:runs,1) = 0;
LMitRUNS(1:runs,1) = 0;
time(1:runs,1) = 0;
it(1:runs,1) = 0;
indx(1:runs,1:2) = 0;

%RBFs
Ninit = 480; Nfinal = Ninit;
N = Ninit;

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
%% runs
for runs=1:20
    %RMSE vector in the validation dataset
    itRMSEva = 0;
    
    %stopwatch timer
    tic
         
    %% RBF center selection
    %kmeans
    [idx, c] = kmeans(xtrnorm,N);
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
    itRMSEtr(1,1) = sqrt(mean((ytr-itytruetr).^2));
    
    %% validation dataset
    [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA); 
    itytruevanorm = itphiva*w;
    
    [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1)); 
    itRMSEva(1,1)= sqrt(mean((yva-itytrueva).^2));
                    
    %save this value as the min value of RMSE in the validation dataset
    minva(1,1) = itRMSEva(1,1);
    % & the current parameters
    cmin=c;
    wmin=w;
    SIGMAmin = SIGMA;
    
    %% testing dataset
    [itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),SIGMA);
    itytruetenorm = itphite*w;
    
    [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));
    itRMSEte(1,1)= sqrt(mean((yte-itytruete).^2));
    
    %% 1/2 || A d - b ||^2 + lambda/2 ||d||^2
    %Jacobian of PHI matrix
    %DPHIwc (m x N no_var)
    DPHIwc = grad_fun(c,w,xtrnorm,N, data, no_var,SIGMA);
    %A (m - N) x N no_var
    A = Q2'*DPHIwc;
    %b (m - N) x 1
    B = -Q2'*ytrnorm;
    %I (N no_var x N no_var)
    I = eye(size(A'*A,1),size(A'*A,2));
    
    %% lambda
    %madsen(2004)
    %The algorithm is not very sensitive to the choice of tau
    %but as a rule of thumb, one should
    %use a small value, eg tau =10^-6 if x0 is believed to be a good approximation to x*.
    %otherwise, use tau =10^-3 or even tau =1.
    tau = 1;
    lambda = tau*max(max(A'*A));
    
    %% successful LM iterations (r>0) counter
    kva =1;
    %% k iterations
    k = 1;
    flagg = false;
    flagit = 0;
    while flagg == false
        %% computing the LM direction
        %Cholesky factorization
        %lower triangular matrix
        %L (Nno_var) x (Nno_var)
        % L*L'  ==  A'*A + lambda*I,'lower';
        L = chol(A'*A + lambda*I,'lower');  % inv(A'*A + lambda*I)*A'*B, slower
        %d vector (N no_var x 1)
        dvec = L'\(L\(A'*B));
        %d matrix (N x no_var)
        d = vec2mat(dvec',no_var);
        %trial centers
        trialc = c+d;  
        
        %% QR factorization of Phi at trialc = c+d
        [phicd] = phi_fun(trialc,xtrnorm,N, data, no_var,SIGMA); 
        [Q1_, Q2_, R1_] = QRfactorization(phicd, data, N);
                         
        %% gain ratio, r
        num = 0.5*norm(Q2'*ytrnorm)^2 - 0.5*norm(Q2_'*ytrnorm)^2;
        denom = 0.5*norm(Q2'*ytrnorm)^2 - 0.5*norm(Q2'*(ytrnorm+DPHIwc*dvec))^2;
       
        %cost at current c
        hh(k+1,1)=0.5*norm(Q2'*ytrnorm)^2;
        %cost at c + d
        hh(k+1,2)=0.5*norm(Q2_'*ytrnorm)^2;
        %linear model cost
        hh(k+1,3)=0.5*norm(Q2'*(ytrnorm + DPHIwc*dvec))^2;
         
        r = num/denom;
       
        %% update lambda
        if r<0.25
            %small value of r, increase lambda
            %go as the steepest descent
            %reduce step length
            lambda = 4*lambda;
        elseif r>0.75
            %large value of rho, decrease lambda
            %go as the gauss-newton
            %increase step length
            lambda = lambda/2;
        end
        %%
        if r>0
            %update centers
            c = c + d;
            %weights ( Linear Least Square Solution)
            w = R1_\(Q1_'*ytrnorm);
            
            %PHI matrix
            [phi] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);
            
            %calculate RMSE in the validation dataset
            [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA);
            itytruevanorm=itphiva*w;
            
            [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));    
            itRMSEva(kva+1,1)= sqrt(mean((yva-itytrueva).^2));
            
            %is this value lower than the minimum one?
            minva(1,1)=round(minva(1,1),4);
            itRMSEva(kva+1)=round(itRMSEva(kva+1),4); 
            if (minva(1,1)>itRMSEva(kva+1))
                %update the min value
                minva(1,1)=itRMSEva(kva+1);
                %and the parameters
                cmin = c;
                wmin = w;
                SIGMAmin = SIGMA;
            else
                flagit = flagit + 1;
            end
            if flagit >= 5
                flagg = true;
            end
            %next iteration LM
            kva = kva+1;
            %update Q2
            Q2 = Q2_;
            %Jacobian, A, b...
            DPHIwc = grad_fun(c,w,xtrnorm,N, data, no_var,SIGMA);  
            A = Q2'*DPHIwc;
            B = -Q2'*ytrnorm;    
        end
        k = k+1;
    end%max iterations
    %% final best values of the parameters in the current run
    c = cmin;
    w = wmin;
    SIGMA = SIGMAmin; 
    %% final results RMSE
    %% training dataset
    
    [itphitr] = phi_fun(c,xtrnorm,N, data, no_var,SIGMA);   
    itytruetrnorm=itphitr*w;
    
    [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
    RMSEtr= sqrt(mean((ytr-itytruetr).^2));
    
    %% validation dataset
    [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),SIGMA);
    itytruevanorm=itphiva*w;
    
    [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));
    RMSEva= sqrt(mean((yva-itytrueva).^2));
      
    %% testing dataset
    [itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),SIGMA);
    itytruetenorm=itphite*w;
    
    [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));  
    RMSEte = sqrt(mean((yte-itytruete).^2));
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
bit=it(indexbestva)
%RMSE path in the validation
bitRMSEva = itRMSEvaruns(:,indexbestva);
bitRMSEmat = cell2mat(bitRMSEva);
%% parameters
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

%% mean & std
%LM iterations
meanitRUNS = round(mean(it))
stditRUNS = round(std(it))

%RMSE validation
meanVA = mean(minRMSEvaRUNS)
stdVA = std(minRMSEvaRUNS)

%RMSE testing
meanTE = mean(RMSEte_va)
stdTE = std(RMSEte_va)

%RMSE training
meanTR = mean(RMSEtr_va)
stdTR = std(RMSEtr_va)

%time
besttime = time(indexbestva)
meantime = mean(time)
stdtime = std(time)
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