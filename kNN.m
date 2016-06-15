function nnarray = kNN(samples, i, k)
%KNN Summary of this function goes here
%   Detailed explanation goes here
    n = size(samples, 1);
    sample = samples(i, :);
    distances = zeros(n, 1);
    
    for j = 1:n
        % Euclidian distance.
        distances(i) = norm(sample - samples(j,:));
    end
    
    [~, sortedIdxs] = sort(distances);
    sortedSamples = samples(sortedIdxs, :);
    nnarray = sortedSamples(1:k, :);
end

