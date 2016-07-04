clear;
clc;
close all;

%% Parameters.
balancingMethod = 0;

%% Getting data from both classes.
filename = 'PAKDD-PAKDD_GERMANO.cod';
[labels, data] = getRawData(filename);
majorityClassIdxs = find(data(:,end));
minorityClassIdxs = find(data(:,end) == 0);

majorityClass = shuffle(data(majorityClassIdxs, 2:end));
minorityClass = shuffle(data(minorityClassIdxs, 2:end));

%% Partitioning into training, validation and test sets.
majorTrSetSize = floor(0.5*size(majorityClass, 1));
majorValSetSize = floor(0.25*size(majorityClass, 1));
majorTestSetSize = size(majorityClass, 1) - (majorTrSetSize + majorValSetSize);

majorTrSet = majorityClass(1:majorTrSetSize,:);
majorValSet = majorityClass(majorTrSetSize+1:majorTrSetSize+majorValSetSize,:);
majorTestSet = majorityClass(majorTrSetSize+majorValSetSize+1:end,:);

minorTrSetSize = floor(0.5*size(minorityClass, 1));
minorValSetSize = floor(0.25*size(minorityClass, 1));
minorTestSetSize = size(minorityClass, 1) - (minorTrSetSize + minorValSetSize);

minorTrSet = minorityClass(1:minorTrSetSize,:);
minorValSet = minorityClass(minorTrSetSize+1:minorTrSetSize+minorValSetSize,:);
minorTestSet = minorityClass(minorTrSetSize+minorValSetSize+1:end,:);

%% Balancing classes.
if balancingMethod == 0
    minorTrSet = oversample(minorTrSet, majorTrSetSize, minorTrSetSize);
    minorValSet = oversample(minorValSet, majorValSetSize, minorValSetSize);
%     minorTestSet = Oversample(minorTestSet, majorTestSetSize, minorTestSetSize);
elseif balancingMethod == 1
%     % Parameters for this method.
%     k = 1;
%     m = 1;
%     
%     majorTrSet = kMeansUndersample(majorTrSet, minorTrSet, minorTrSetSize, m);
%     majorValSet = kMeansUndersample(majorValSet, minorValSet, minorValSetSize, m);
% %     majorTestSet = kMeansUndersample(majorTestSet, minorTestSet, k, m);
    [~, majorTrSetAttrs] = kmeans(majorTrSet(:,1:end-2), minorTrSetSize);
    [~, majorValSetAttrs] = kmeans(majorValSet(:,1:end-2), minorValSetSize);
    
    majorTrSet = [majorTrSetAttrs, majorTrSet(1:size(majorTrSetAttrs, 1),end-1:end)];
    majorValSet = [majorValSetAttrs, majorValSet(1:size(majorValSetAttrs, 1),end-1:end)];
elseif balancingMethod == 2
    k = 3;
    
    minorTrSet = SMOTE(minorTrSet, 100*(floor(majorTrSetSize / minorTrSetSize)-1), k);
    minorValSet = SMOTE(minorValSet, 100*(floor(majorValSetSize / minorValSetSize)-1), k);
elseif balancingMethod == 3
    k = 3;
    
    minorTrSet = adaptedSMOTE(majorTrSet, minorTrSet, 100*(floor(majorTrSetSize / minorTrSetSize)-1), k);
    minorValSet = adaptedSMOTE(majorValSet, minorValSet, 100*(floor(majorValSetSize / minorValSetSize)-1), k);
end

% Building training, validation and data sets with balanced sets.
trainingSet = shuffle([majorTrSet; minorTrSet]);
validationSet = shuffle([majorValSet; minorValSet]);
testSet = shuffle([majorTestSet; minorTestSet]);

save('training.mat', 'trainingSet');
save('validation.mat', 'validationSet');
save('test.mat', 'testSet');
