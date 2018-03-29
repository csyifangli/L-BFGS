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