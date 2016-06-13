function [set] = Oversample(minorSet, majorSize, minorSize)
%OVERSAMPLE Summary of this function goes here
%   Detailed explanation goes here
    quotient = floor(majorSize / minorSize);
    remainder = majorSize - quotient*minorSize;
    set = [repmat(minorSet, quotient, 1) ; minorSet(1:remainder,:)];
end

