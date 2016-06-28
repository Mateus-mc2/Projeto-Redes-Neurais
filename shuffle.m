function output = shuffle(input)
%SHUFFLE Shuffles the input matrix rows.
%   Detailed explanation goes here
    permutation = randperm(size(input, 1));
    output = input(permutation,:);
end

