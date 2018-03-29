clear
%% input
%dataset: 3_autoMPG, Fuzzy Means (s = 10)
%centers
load 'centers.mat' 'c'
%training dataset
load 'xtrnorm.mat' 'xtrnorm'
load 'ytrnorm.mat' 'ytrnorm'

%N: no. of centers %no_var: no. of variables %data: no. of training examples
N = size(c,1); no_var = size(c,2); data = size(xtrnorm,1);

%% initial value of x
%N x no_var
x0 = c;
%N*no_var x 1
x0 = vec(x0');

%% objective
f=neural_net(xtrnorm,ytrnorm,N,data,no_var);
call_f = f.makef();
%initial cost, grad
[f_x0, grad_f_x0] = call_f(x0);

g = zeroFunction();

%% options
opt.maxit = 10;
%opt.method = 'lbfgs';
opt.linesearch = 'lemarechal';

%% results
fprintf('\nL-BFGS\n');
out_lbfgs = forbes(f, g, x0,  [], [], opt);
fprintf('\n');
fprintf('message    : %s\n', out_lbfgs.message);
fprintf('iterations : %d\n', out_lbfgs.solver.iterations);
fprintf('time       : %7.4e\n', out_lbfgs.solver.ts(end));

%% final value of x
finalc = out_lbfgs.x;

%N x no_var
finalc = vec2mat(finalc,no_var);

% [phi,w] = call_phi_fun(finalc,xtrnorm,N, data, no_var,ytrnorm);
% 
% h = 0.5*norm(phi*w-ytrnorm)^2