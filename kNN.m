function nnarray = kNN(samples, i, k)
%KNN Summary of this function goes here
%   Detailed explanation goes here
    n = size(samples, 1);
    sample = samples(i, :);
    distances = zeros(n, 1);
    
    for j = 1:n
        % Euclidian distance.
        distances(j) = norm(sample - samples(j,:));
    end
    
    [~, nnarray] = sort(distances);
    % Since the i-th sample will always appear on the first index,
    % it is necessary to start from the "second" neighbor.
    nnarray = nnarray(2:k+1);
end

