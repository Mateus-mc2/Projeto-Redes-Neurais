function set = KMeansUndersample(majorityClass, minorityClass, k, m)
%KMEANSUNDERSAMPLING Summary of this function goes here
%   Detailed explanation goes here

%% Cluster data using k-means and get majority and minority classes.
    data = [majorityClass; minorityClass];
    perm = randperm(size(data, 1));
    data = data(perm,:);
    
    idxs = kmeans(data, k);

    MA = cell(k,1);
    sizesMA = zeros(k,1);
    MI = cell(k,1);
    sizesMI = zeros(k,1);
    
    for i = 1:k
       cluster = data(idxs == i,:);
       MA{i} = cluster(find(cluster(:,end)), :);
       sizesMA(i) = size(MA{i}, 1);
       MI{i} = cluster(find(cluster(:,end) == 0), :);
       
       if size(MI{i}, 1) == 0
           sizesMI(i) = 1;
       else
           sizesMI(i) = size(MI{i}, 1);
       end
    end
    
    sizeMI = size(minorityClass, 1);
    totalRatio = sum(sizesMA ./ sizesMI);
    sizesSelectedMA = zeros(k, 1);
    
    for i = 1:k
       sizesSelectedMA(i) = m*sizeMI*sizesMA(i) / (totalRatio*sizesMI(i));
    end
    
    sizeSelectedMA = sum(sizesSelectedMA);
    set = zeros(sizeSelectedMA, size(data, 2));
    start = 0;
    
    for i = 1:k
        clusterMA = cell2mat(MA(i));
        sizeSelected = sizesSelectedMA(i);
        selectedSamples = randi(size(clusterMA, 1), sizeSelected, 1);
        set(start+1:start+sizeSelected, :) = clusterMA(selectedSamples, :);
        start = start + sizeSelected;
    end
end

