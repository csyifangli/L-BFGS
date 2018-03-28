clear
%dataset: 3_autoMPG, Fuzzy Means (s = 10)
load 'centers.mat' 'c'
%training dataset
load 'xtrnorm.mat' 'xtrnorm'
load 'ytrnorm.mat' 'ytrnorm'

%N: no. of centers %no_var: no. of variables %data: no. of training examples
N = size(c,1); no_var = size(c,2); data = size(xtrnorm,1);

%initial value
x0 = c;
x0 = vec(x0');

%objective
f=neural_net(xtrnorm,ytrnorm,N,data,no_var);
call_f = f.makef();
%initial cost, grad
[f_x, grad_f_x] = call_f(x0);

g = zeroFunction();%l1Norm(0);

%options
opt.maxit = 10;
opt.method = 'lbfgs';
opt.linesearch = 'lemarechal';

fprintf('\nL-BFGS\n');

%results
out_lbfgs = forbes(f, g, x0,  [], [], opt);
fprintf('\n');
fprintf('message    : %s\n', out_lbfgs.message);
fprintf('iterations : %d\n', out_lbfgs.solver.iterations);
%fprintf('matvecs    : %d\n', out_lbfgs.solver.operations.C1);
%fprintf('prox       : %d\n', out_lbfgs.solver.operations.proxg);
fprintf('time       : %7.4e\n', out_lbfgs.solver.ts(end));
%fprintf('residual   : %7.4e\n', out_lbfgs.solver.residual(end));

%final value of x
finalc = out_lbfgs.x;

finalc = vec2mat(finalc,no_var);
[phi,w] = call_phi_fun(finalc,xtrnorm,N, data, no_var,ytrnorm);

h = 0.5*norm(phi*w-ytrnorm)^2