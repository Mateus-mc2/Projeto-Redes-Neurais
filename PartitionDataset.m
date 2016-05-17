clear;
clc;
close all;

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
majorTrSet = majorityClass(1:majorTrSetSize,:);
majorValSet = majorityClass(majorTrSetSize+1:majorTrSetSize+majorValSetSize,:);
majorTestSet = majorityClass(majorTrSetSize+majorValSetSize+1:end,:);

% Balancing minority class training data by oversampling.
minorTrSetSize = floor(0.5*size(minorityClass, 1));
quotientTr = floor(majorTrSetSize / minorTrSetSize);
remainderTr = majorTrSetSize - quotientTr*minorTrSetSize;
minorTrSet = minorityClass(1:minorTrSetSize,:);
minorTrSet = [repmat(minorTrSet, quotientTr, 1) ; minorTrSet(1:remainderTr,:)];

% Balancing minority class validation data by oversampling.
minorValSetSize = floor(0.25*size(minorityClass, 1));
quotientVal = floor(majorValSetSize / minorValSetSize);
remainderVal = majorValSetSize - quotientVal*minorValSetSize;
minorValSet = minorityClass(minorTrSetSize+1:minorTrSetSize+minorValSetSize,:);
minorValSet = [repmat(minorValSet, quotientVal, 1) ; minorValSet(1:remainderVal,:)];

% Balancing minority class test data by oversampling.
majorTestSetSize = size(majorityClass, 1) - (majorTrSetSize + majorValSetSize);
minorTestSetSize = size(minorityClass, 1) - (minorTrSetSize + minorValSetSize);
quotientTest = floor(majorTestSetSize / minorTestSetSize);
remainderTest = majorTestSetSize - quotientTest*minorTestSetSize;
minorTestSet = minorityClass(minorTrSetSize+minorValSetSize+1:end,:);
minorTestSet = [repmat(minorTestSet, quotientTest, 1) ; minorValSet(1:remainderTest,:)];

% Building training, validation and data sets with balanced sets.
trainingSet = [majorTrSet; minorTrSet];
validationSet = [majorValSet; minorValSet];
testSet = [majorTestSet; minorTestSet];

save('training.mat', 'trainingSet');
save('validation.mat', 'validationSet');
save('test.mat', 'testSet');
