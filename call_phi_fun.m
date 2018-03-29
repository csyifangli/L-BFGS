% function [phi,w] = call_phi_fun(cnorm,xnorm,N, data, no_var,ytrnorm,sigma)
%     phi(1:data,1:N+1) = 0;
%     x=repmat(xnorm,1,N);
%     cnormnew(1,1:no_var)=cnorm(1,:);
% 
%     k = no_var + 1;
%     for u = 2:N
%         cnormnew(1, k:k+no_var-1)=cnorm(u,:);
%         k = k+ no_var;
%     end
% 
%     c=repmat(cnormnew,data,1);
% 
%     dist=x-c;
% 
%     d=sqrt(sum(dist(:,1:no_var).^2, 2));
% 
%     phi(:,1)=exp(-(d.^2)/(2*sigma(1,1)^2));
% 
%     k = no_var +1;
%     for u=2:N
%         d=sqrt(sum(dist(:, k:k+no_var-1).^2, 2));
%         phi(:,u)=exp(-(d.^2)/(2*sigma(1,u)^2));   
%         k = k+ no_var;
%     end
%        phi(:,N+1) = 1; 
%        w = phi\ytrnorm;
% end


% function [phi,w] = call_phi_fun(cnorm,xnorm,N, data, no_var,ytrnorm,sigma)
%  phi(1:data,1:N+1) = 0;
%     for i = 1:data
%         for j = 1:N
%             phi(i,j) = exp(-norm(c(j,:)-xtrnorm(i,:))^2/(2*sigma(1,j)^2));
%         end
%     end
%     phi(:,N+1) = 1; 
%     w = phi\ytrnorm;
% end
%     