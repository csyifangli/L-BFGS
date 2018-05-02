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

runs = 1;
%deiktes gia thn apothikeysh
%twn kentrwn se kathe run
init = 1;
fin = no_var;
        

         
tic

%RBF center selection
%fuzzy means (runs=1)

[c, N] = SFMfunction(no_var,data,xtrnorm,7);%(n,m,x,s)

cinitial = c;

%P-nearest
[sigma] = Pnn(N,c);


sigmainitial = sigma;

         %% PHI matrix computation TRAINING OBJECTIVE
         [phi] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,sigma);
         [Q1, Q2, R1] = QRfactorization(phi, data, N+1);    

         %weights
         w = R1\(Q1'*ytrnorm);

                        
         winitial = w;

          
           %initial cost
           hh(1,1) = 0.5*norm(phi*w-ytrnorm)^2;
                    
                %% RMSE
                %% TRAINING data

                
                itytruetrnorm = phi*w;
                [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));
             
                itRMSEtr(1,1) = sqrt(mean((ytr-itytruetr).^2));   
                %% VALIDATION data
                [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),ytrnorm,sigma);
                
                itytruevanorm = itphiva*w;
                [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));

                itRMSEva(1,1)= sqrt(mean((yva-itytrueva).^2));  
                
                                %estw to prwto, min
                                minva(1,1)= itRMSEva(1,1);
                                %apothhkeysh twn parametrwn c, w, sigma
                                cmin=c;
                                wmin=w;
                                sigmamin = sigma;
                                
                %% TESTING data
                [itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),ytrnorm,sigma);
                
                itytruetenorm = itphite*w;
                [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));

                itRMSEte(1,1)= sqrt(mean((yte-itytruete).^2));   
               %% Jacobian...
               DPHIwc = grad_fun(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma);
               DPHIwsigma = grad_fun_SIGMA2(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma);
               % A = Q2'*DPHIwc;     
               % B = -Q2'*ytrnorm;     
               A = -[DPHIwc DPHIwsigma];    
               B = - (ytrnorm - phi*w);    
               I = eye(size(A'*A,1),size(A'*A,2));
               lambda = 1;
    
              %% k iterations   
              %LM succesive iterations     
              kva=1;
for k = 1:20
                        L = chol(A'*A + lambda*I,'lower');  %or inv(A'*A + lambda*I)*A'*B

                        % dvec = L'\(L\(A'*B)); 

                        %d = N x no_var
                        % d =vec2mat(dvec',no_var);
                        % trialc=c+d;

                        %directions
                        dvec = L'\(L\(A'*B)); 

                        dcvec = dvec(1:N*no_var);
                        dc =vec2mat(dcvec,no_var);                      
                        dsigma=dvec(N*no_var+1:N*no_var+N);
                        dsigma = dsigma';
                  
                        trialc = c+dc;
                        trialsigma = sigma+dsigma;


                        [phicd] = phi_fun(trialc,xtrnorm,N, data, no_var,ytrnorm,trialsigma);
                        [Q1_, Q2_, R1_] = QRfactorization(phicd, data, N+1);


                       
                        num = 0.5*norm(ytrnorm - phi*w)^2 - 0.5*norm(ytrnorm - phicd*w)^2;
                        denom = 0.5*norm(ytrnorm - phi*w)^2 - 0.5*norm(ytrnorm - phi*w + A*dvec)^2;

                        hh(k+1,1)=0.5*norm(ytrnorm - phi*w)^2;
                        hh(k+1,2)=0.5*norm(ytrnorm - phicd*w)^2;
                        hh(k+1,3)=0.5*norm(ytrnorm - phi*w + A*dvec)^2;

                        r = num/denom;
                        rr(k,1) = r;

                        if r<0.25 
                            lambda = 4*lambda;
                        elseif r>0.75
                            lambda = lambda/2;
                        end

                       if r>0 
                            c = c + dc;
                            sigma = sigma + dsigma;
                            w= R1_\(Q1_'*ytrnorm);   
                            [phi] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,sigma);
                             
                                                                                      
                               [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),ytrnorm,sigma);

                               itytruevanorm=itphiva*w;
                               [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));

                               %to RMSEva ypologizetai otan
                               %ginetai allagh sta kentra
                               itRMSEva(kva+1,1)= sqrt(mean((yva-itytrueva).^2));  
                             
                               %an vrethei mikrotero RMSEva
                               %enhmerwnontai oi times twn c, w, sigma
                               %kai apothikeyontai
                               minva(1,1)=round(minva(1,1),4);
                               itRMSEva(kva+1)=round(itRMSEva(kva+1),4);
                               if (minva(1,1)>itRMSEva(kva+1))   
                                    %update values @min RMSEva
                                    minva(1,1)=itRMSEva(kva+1);
                                    cmin=c;
                                    wmin=w;
                                    sigmamin = sigma;
                               end

                              kva = kva+1;
                           
                              Q2 = Q2_;
                            
                           
                              DPHIwc = grad_fun(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma);
                              DPHIwsigma2 = grad_fun_SIGMA2(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma);
                              % A = Q2'*DPHIwc;
                              % B = -Q2'*ytrnorm;
                              A = -[DPHIwc DPHIwsigma2];   
                              B = - (ytrnorm - phi*w); 
                       end
              end  
%telikes best times c, w, meta tis max k
%gia kathe run
c=cmin;
w=wmin;
sigma = sigmamin;

               %TELIKA APOTELESMATA
               %% TRAINING
               [itphitr] = phi_fun(c,xtrnorm,N, data, no_var,ytrnorm,sigma);

               itytruetrnorm=itphitr*w;
               [itytruetr] = unnormalization(a, b, minvarytr, maxvarytr, itytruetrnorm, size(ytr,1));

               RMSEtr= sqrt(mean((ytr-itytruetr).^2));         
               %% VALIDATION
               [itphiva] = phi_fun(c,xvanorm, N, size(xva,1), size(xva,2),ytrnorm,sigma);
                
               itytruevanorm=itphiva*w;
               [itytrueva] = unnormalization(a, b, minvarytr, maxvarytr, itytruevanorm, size(yva,1));
             
               RMSEva= sqrt(mean((yva-itytrueva).^2));   
                
               %% TESTING
               [itphite] = phi_fun(c,xtenorm, N, size(xte,1), size(xte,2),ytrnorm,sigma);
                
               itytruetenorm=itphite*w;
               [itytruete] = unnormalization(a, b, minvarytr, maxvarytr, itytruetenorm, size(yte,1));
             
               RMSEte = sqrt(mean((yte-itytruete).^2));  
            
%apothikeysh gia ola ta runs
itRMSEvaruns{:,runs}=itRMSEva(:,1);
RMSEtr_va(runs,1)=RMSEtr;
minRMSEvaRUNS(runs,1)=RMSEva;
RMSEte_va(runs,1)=RMSEte;
%w, sigma, c pou dinoun to elaxisto RMSEva
%se kathe run
%
wruns(:,runs) = w;
wrunsinitial(:,runs) = winitial;
%
sigmaruns(runs,:) = sigma;
sigmarunsinitial(runs,:) = sigmainitial;
%
cruns(:,init:fin) = c;
crunsinitial(:,init:fin) = cinitial;
%
indx (runs,1)= init;
indx (runs,2) = fin;
init = fin + 1;
fin = fin + no_var;


% it = meta apo poses epityxhmenes epanalhpseis (r>0)
%      brethike to elaxisto RMSEva
[miin,indexminn]=min(round(itRMSEva,4));
%se kathe run
it(runs,1) = indexminn-1;
time(runs,1) = toc;
 

%best values
[bva, indexbestva] = min(minRMSEvaRUNS);
bva
bte_va=RMSEte_va(indexbestva)
btr_va=RMSEtr_va(indexbestva)
bit=it(indexbestva)


bitRMSEva = itRMSEvaruns(:,indexbestva);
bitRMSEmat = cell2mat(bitRMSEva);

%times w
bwinitial=wrunsinitial(:,indexbestva);
bwfinal=wruns(:,indexbestva);

%times sigma
bsigmarunsinitial=sigmarunsinitial(indexbestva,:);
bsigmafinal=sigmaruns(indexbestva,:);

%kentra, c
bindx = indx(indexbestva,:);
bcfinal = cruns(:,bindx(1,1):bindx(1,2));
bcinitial = crunsinitial(:,bindx(1,1):bindx(1,2));


%time
besttime = time(indexbestva)


%telika sigma, c, w
sigma = bsigmafinal;
c = bcfinal;
w = bwfinal;