function [phi,w] = call_phi_fun(c,xtrnorm,N, data, no_var,ytrnorm)
    for i = 1:data       
       for j = 1:N 
        phi(i,j) = norm(c(j,:)-xtrnorm(i,:))^2*log10(norm(c(j,:)-xtrnorm(i,:))+1);
       end
    end  
      phi(:,N+1) = 1; 
      w = phi\ytrnorm;
end