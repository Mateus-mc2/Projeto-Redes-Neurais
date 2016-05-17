function [labels, data] = getRawData(filename)
%GETDATA Summary of this function goes here
%   Detailed explanation goes here
    file = fopen(filename, 'rt');
    assert(file ~= -1, 'Could not read the specified file: ', filename);
%     cleanupObj = onCleanup(@() fclose(file));
    header = fgetl(file);
    labels = strsplit(header, char(9));
    cols = sum(header == char(9)) + 1;
    dataFormat = repmat('%f', 1, cols);
    data = cell2mat(textscan(file, dataFormat, 'Delimiter', char(9)));
    fclose(file);
end

