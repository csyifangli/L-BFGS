function [ytrue] = unnormalization(a, b, minvary, maxvary, ytruenorm, data);

for i = 1:data
    ytrue(i,1) =  minvary+((ytruenorm(i,1)-a)*(maxvary-minvary))/(b-a);
end

end
        