function [phi] = phi_fun(cnorm,xnorm,N, data, no_var,ytrnorm,sigma)
%tic
    xx=repmat(xnorm,1,N);

    cnormnew(1,1:no_var)=cnorm(1,:);

    k = no_var + 1;
    for u = 2:N
        cnormnew(1, k:k+no_var-1)=cnorm(u,:);
        k = k+ no_var;
    end

    c=repmat(cnormnew,data,1);

    dist=xx-c;

    d=sqrt(sum(dist(:,1:no_var).^2, 2));
    %Gauss
    phi(:,1)=exp(-(d.^2)/(2*sigma(1,1)^2));
   %TPS
    %phi(:,1)=(d.^2).*log10(d+1);
  

    k = no_var +1;
    for u=2:N
        d=sqrt(sum(dist(:, k:k+no_var-1).^2, 2));
        %Gauss
       phi(:,u)=exp(-(d.^2)/(2*sigma(1,u)^2));   
       
       %TPS
%        phi(:,u)=(d.^2).*log10(d+1);
        k = k+ no_var;
    end
    phi(:,N+1) = 1;
   % new = toc
end