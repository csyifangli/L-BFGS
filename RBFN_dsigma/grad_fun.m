function [DPHIw] = grad_fun(c,w,phi,xtrnorm,N, data, no_var,ytrnorm,sigma)
%tic
DPHIw(1:N,1:N*no_var) = 0;
            n_arx=1;
            n_tel=no_var;
  
            for cent = 1:N
                        %norm squared
                        %d = abs(xtrnorm-repmat(c(cent,:),data,1));
                        diff = repmat(c(cent,:),data,1) - xtrnorm;
    
                        %Gaussian
                        dphidc_ =  (2*(-1/(2*sigma(1,cent)^2))*exp((-sum(diff.^2,2))/(2*sigma(1,cent)^2)).*diff);
                       
                        DPHIw(1:data,n_arx:n_tel) = w(cent,1)*dphidc_;
                        n_arx=n_tel+1;
                        n_tel=n_tel+no_var;
            end
            
%gradnew = toc
end
        %TPS
         %dphidc(1:data,n_arx:n_tel)=dphidc_;
                     %   dphidc_ =(2*log10(d+1) + (d)./((d+1)*log(10))).*diff;