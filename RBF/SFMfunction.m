function [u,L] = SFMfunction(N,K,xtrnorm,s)
% %% prerequisites
% %input
% load('training_data_set1.mat'); 
% %N no. of variables
% N = size(xtr,2);
% %K. no. of training data
% K = size(xtr,1);
% %normalized input
% [xtrnorm, allmaxxtr, allminxtr] = normalizationx(xtr, N, -1, 1);
%training data set Utrain
U = xtrnorm';
%# of fuzzy sets for partitioning each input dimension
%s = 10;
%partitioning the domain of
%each input variable xi (i = 1, 2, ..., N) into s triangular fuzzy sets
amax = 1;
amin = -amax;
deltaa = (amax-amin)/(s-1);
a = [ amin (amin+deltaa):deltaa:amax ];

%% calculations
%take the first data point: k<--1
k = 1;
%begin calculations for the first RBF center: L<--1
L = 1;
for i = 1:N
    d = abs(a-repmat(U(i,k),1,s))/(deltaa);
    for j = 1:s
          if U(i,k) >= (a(1,j) - deltaa) && U(i,k) <= a(1,j) + deltaa
              membership(1,j) = 1 - d(1,j);
          else
              membership(1,j) = 0;
          end
    end
    %calculate the fuzzy set with max membership in dimension i
    [maxmember, index] = max(membership);
    A(1,i) = a(1,index);
end
    
%generate the first RBF center
u(L,:) = A;
  
%continue with the remaining input examples
for k = 2:K
    %is k covered by the hyperspheres?
    for l = 1:L
        rd(l) = norm(u(l,:)-U(:,k)')/(sqrt(N)*deltaa);
    end
    if min(rd) > 1
        %if not, add a new RBF center
        L = L +1;
        %begin calculations for the current RBF center:
        for i = 1:N
            d = abs(a-repmat(U(i,k),1,s))/(deltaa);
            for j = 1:s
                    if U(i,k)>=a(1,j) - deltaa && U(i,k)<=a(1,j) + deltaa
                      membership(1,j) = 1 - d(1,j);
                  else
                      membership(1,j) = 0;
                  end
            end
            %calculate the fuzzy set with max membership in dimension i
            [maxmember, index] = max(membership);
            A(1,i) = a(index);
        end
        %generate the current RBF center
        u(L,:) = A;
    end
end
end