function output = SMOTE(samples, N, k)
%SMOTE Summary of this function goes here
%   Detailed explanation goes here
    T = size(samples, 1);
    
    if N < 100
        p = randperm(T);
        samples = samples(p,:);
        T = floor(T*N/100);
        N = 100;
    end
    
    N = floor(N/100);
    index = 1;
    syntheticData = zeros(T*N, size(samples, 2));
    
    for i = 1:T
        nnarray = kNN(samples, i, k);
        aux = N;
        
        while aux > 0
           nn = randi(k);
           dif = samples(nnarray(nn),1:end-2) - samples(i,1:end-2);
           gap = rand();
           syntheticData(index,1:end-2) = samples(i,1:end-2) + gap*dif;
           syntheticData(index,end-1:end) = samples(i,end-1:end);
           index = index + 1;
           aux = aux - 1;
        end
    end
    
    output = shuffle([syntheticData; samples]);
end

