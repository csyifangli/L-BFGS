%p-nearest neighbor
function [SIGMA] = Pnn(N,c)
SIGMA(1,N) = 0;
 for i = 1:N
    cne=repmat(c(i,:),N,1);   
    dist(1,1:N) = 0;
    for j = 1:N
            dist(j) = norm(cne(j,:)-c(j,:));
    end

    distsort=sort(dist);

    %% P=2
    min1 = distsort(1,2); % distsort(1,1) = 0
    min2 = distsort(1,3);

    SIGMA(i) = sqrt((1/2)*(min1^2+min2^2));

    %% P = N-1
%    SIGMA(i) = sqrt((1/(N))*(sum(distsort(1,1:N).^2))); 
 end

end