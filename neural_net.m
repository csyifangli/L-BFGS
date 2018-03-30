function obj = neural_net(xtrnorm,ytrnorm,N,data,no_var)
   obj.makef = @() @(x) call_neural_net_f(x, xtrnorm,ytrnorm,N,data,no_var);
end

function [val, grad] = call_neural_net_f(x, xtrnorm,ytrnorm,N,data,no_var)
   %value and gradient of f(x)
  
   %centers, N x no_var
   x = vec2mat(x,no_var);
   [sigma] = Pnn(N,x);
   
   %activation function, weights
   [phi,w] = call_phi_fun(x,xtrnorm,N, data, no_var,ytrnorm,sigma);
  
   %value of f(x)
   val = 0.5*norm(phi*w-ytrnorm)^2;

   %grad of f(x)
   [DPHIw] = call_grad_fun(x,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma);
   grad = DPHIw;
end

%gaussian
function [phi,w] = call_phi_fun(cnorm,xnorm,N, data, no_var,ytrnorm,sigma)
    phi(1:data,1:N+1) = 0;
    x=repmat(xnorm,1,N);
    cnormnew(1,1:no_var)=cnorm(1,:);

    k = no_var + 1;
    for u = 2:N
        cnormnew(1, k:k+no_var-1)=cnorm(u,:);
        k = k+ no_var;
    end

    c=repmat(cnormnew,data,1);

    dist=x-c;

    d=sqrt(sum(dist(:,1:no_var).^2, 2));

    phi(:,1)=exp(-(d.^2)/(2*sigma(1,1)^2));

    k = no_var +1;
    for u=2:N
        d=sqrt(sum(dist(:, k:k+no_var-1).^2, 2));
        phi(:,u)=exp(-(d.^2)/(2*sigma(1,u)^2));   
        k = k+ no_var;
    end
       phi(:,N+1) = 1; 
       w = phi\ytrnorm;
end

function [DPHIw] = call_grad_fun(x,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma)
 DPHIw(1:N,1:no_var) = 0;
 for cent = 1:N
                      %  d = abs(xtrnorm-repmat(x(cent,:),data,1));
                        diff = repmat(x(cent,:),data,1) - xtrnorm;
                        dphidc_ =  (2*(-1/(2*sigma(1,cent)^2))*exp((-diff.^2)/(2*sigma(1,cent)^2)).*diff);
                        DPHIw(cent,:) = -w(cent,1)*(ytrnorm-phi*w)'*dphidc_;        
 end  
 DPHIw = vec(DPHIw');
end
  
%p-nearest neighbor
function [sigma] = Pnn(N,c)
sigma(1,N) = 0;
 for i = 1:N
    cne=repmat(c(i,:),N,1);   
    dist(1,1:N) = 0;
    for j = 1:N
            dist(j) = norm(cne(j,:)-c(j,:));
    end

    distsort=sort(dist);

    %% P=2
    min1 = distsort(1,2); % distsort(1,1) = 0
    min2 = distsort(1,3);

    sigma(i) = sqrt((1/2)*(min1^2+min2^2));

    %% P = N-1
     % sigma(1,i) = sqrt((1/(N-1))*(sum(distsort(1,1:N-1).^2))); 
 end

end