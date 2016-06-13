clear;
clc;
close all;

%% Parameters.
balancingMethod = 1;

%% Getting data from both classes.
filename = 'PAKDD-PAKDD_GERMANO.cod';
[labels, data] = getRawData(filename);
majorityClassIdxs = find(data(:,end));
minorityClassIdxs = find(data(:,end) == 0);

majorityClass = data(majorityClassIdxs, 2:end);
minorityClass = data(minorityClassIdxs, 2:end);
permMajorityClass = randperm(size(majorityClassIdxs, 1))';
permMinorityClass = randperm(size(minorityClassIdxs, 1))';

majorityClass = majorityClass(permMajorityClass,:);
minorityClass = minorityClass(permMinorityClass,:);

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

if balancingMethod == 0
    %% Balancing minority class by oversampling.
    minorTrSet = Oversample(minorTrSet, majorTrSetSize, minorTrSetSize);
    minorValSet = Oversample(minorValSet, majorValSetSize, minorValSetSize);
    minorTestSet = Oversample(minorTestSet, majorTestSetSize, minorTestSetSize);
elseif balancingMethod == 1
    % Parameters for this method.
    k = 1;
    m = 1;
    
    majorTrSet = KMeansUndersample(majorTrSet, minorTrSet, k, m);
    majorValSet = KMeansUndersample(majorValSet, minorValSet, k, m);
    majorTestSet = KMeansUndersample(majorTestSet, minorTestSet, k, m);
end

% Building training, validation and data sets with balanced sets.
trainingSet = [majorTrSet; minorTrSet];
validationSet = [majorValSet; minorValSet];
testSet = [majorTestSet; minorTestSet];

save('training.mat', 'trainingSet');
save('validation.mat', 'validationSet');
save('test.mat', 'testSet');
