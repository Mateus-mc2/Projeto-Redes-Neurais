function set = oversample(minorSet, majorSize, minorSize)
%OVERSAMPLE Summary of this function goes here
%   Detailed explanation goes here
    m = majorSize;
    n = size(minorSet, 2);
    set = zeros(m, n);
    
    for i = 1:m
        if i <= minorSize
            set(i,:) = minorSet(i,:);
        else
            sample = randi(minorSize);
            set(i,:) = minorSet(sample,:);
        end
    end
    
    set = shuffle(set);
    
%     quotient = floor(majorSize / minorSize);
%     remainder = majorSize - quotient*minorSize;
%     set = [repmat(minorSet, quotient, 1) ; minorSet(1:remainder,:)];
end

