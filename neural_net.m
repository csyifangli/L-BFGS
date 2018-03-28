function obj = neural_net(xtrnorm,ytrnorm,N,data,no_var)
   obj.makef = @() @(x) call_neural_net_f(x, xtrnorm,ytrnorm,N,data,no_var);
%     obj.L = mu; % Lipschitz constant of the gradient of f
%     obj.hasHessian = 1;
%     obj.isConvex = 1;
%     obj.isQuadratic = 0;
end

function [val, grad] = call_neural_net_f(x, xtrnorm,ytrnorm,N,data,no_var)
   %value and gradient of f(x)
  
   %centers, N x no_var
   x = vec2mat(x,no_var);
  
   %activation function, weights
   [phi,w] = call_phi_fun(x,xtrnorm,N, data, no_var,ytrnorm);
  
   %value of f(x)
   val = 0.5*norm(phi*w-ytrnorm)^2;

   %grad of f(x)
   [DPHIw] = call_grad_fun(x,w,phi,xtrnorm,N, data, no_var,ytrnorm);
   grad = DPHIw;
end

function [phi,w] = call_phi_fun(x,xtrnorm,N, data, no_var,ytrnorm)
   phi(data,N+1) = 0;
   x = vec2mat(x,no_var);
   xtrnorm = vec2mat(xtrnorm,no_var);
       for i = 1:data       
           for j = 1:N 
            phi(i,j) = norm(x(j,:)-xtrnorm(i,:))^2*log10(norm(x(j,:)-xtrnorm(i,:))+1);
           end
       end  
   phi(:,N+1) = 1; 
   w = phi\ytrnorm;
end

function [DPHIw] = call_grad_fun(x,w,phi,xtrnorm,N, data, no_var,ytrnorm)
   DPHIw(N,1:no_var) = 0;
   for cent = 1:N
                        d = abs(xtrnorm-repmat(x(cent,:),data,1));
                        diff = repmat(x(cent,:),data,1) - xtrnorm;
                        dphidc_ = (2*log10(d+1) + (d)./((d+1)*log(10))).*diff;
                        DPHIw(cent,:) = -w(cent,1)*(ytrnorm-phi*w)'*dphidc_;
   end
   DPHIw = vec(DPHIw');
end