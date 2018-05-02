function [ynorm, maxvarytr, minvarytr] = normalizationy(y, a, b);

%ytr
ytrT = y';
maxvarytr = max(ytrT);
minvarytr = min(ytrT);

%for j = 1:data
    Ly = (ytrT(1,:)- minvarytr)/(maxvarytr-minvarytr);
    ytrn(1,:) = a+Ly*(b-a);
%end
ynorm = ytrn';

end