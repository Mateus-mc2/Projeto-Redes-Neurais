function output = adaptedSMOTE(samplesMajority, samplesMinority, N, k)
%ADAPTEDSMOTE Summary of this function goes here
%   Detailed explanation goes here
    S = size(samplesMajority, 1);
    T = size(samplesMinority, 1);
    
    if N < 100
        p = randperm(T);
        samplesMinority = samplesMinority(p,:);
        T = floor(T*N/100);
        N = 100;
    end
    
    N = floor(N/100);
    index = 1;    
    samples = [samplesMajority; samplesMinority];
    syntheticData = zeros(T*N, size(samples, 2));
    
    for i = S+1:S+T
        nnarray = kNN(samples, i, k);
        aux = N;
        
        while aux > 0
            nn = randi(k);            
            gap = rand();
            
            if nnarray(nn) <= S  % Majority class
                gap = 0.5*gap;
            end
            
            dif = samples(nnarray(nn),1:end-2) - samples(i,1:end-2);
            syntheticData(index,1:end-2) = samples(i,1:end-2) + gap*dif;
            syntheticData(index,end-1:end) = samples(i,end-1:end);
            index = index + 1;
            aux = aux - 1;
        end
    end
    
    output = shuffle([syntheticData; samplesMinority]);
end

