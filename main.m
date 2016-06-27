echo on
clear
clc
close all

%    Informacoes sobre a rede e os dados
numInputs = 46;     % Numero de nodos de entrada
numHidden = 5;      % Numero de nodos escondidos
numOutputs = 2;     % Numero de nodos de saida
numTr = 39950;      % Numero de padroes de treinamento
numVal = 19975;     % Numero de padroes de validacao
numTest = 10175;    % Numero de padroes de teste

echo off

% %    Abrindo arquivos 
load('training.mat');
load('validation.mat');
load('test.mat');

trainingInputs = trainingSet(1:numTr, 1:numInputs)';
trainingOutputs = trainingSet(1:numTr, (numInputs + 1):(numInputs + numOutputs))';

validationInputs = validationSet(1:numVal, 1:numInputs)';
validationOutputs = validationSet(1:numVal, (numInputs + 1):(numInputs + numOutputs))';

testInputs = testSet(1:numTest, 1:numInputs)';
testOutputs = testSet(1:numTest, (numInputs + 1):(numInputs + numOutputs))';

%   Criando a rede (para ajuda, digite 'help newff')
intervalMatrix = zeros(numInputs, 2);

for entrada = 1 : numInputs;  % Cria 'matrizFaixa', que possui 'numEntradas' linhas, cada uma sendo igual a [0 1].
     intervalMatrix(entrada,:) = [0 1];  
end

net = newff(intervalMatrix,[numHidden numOutputs],{'tansig','tansig'},'traingdm','learngdm','mse');
% matrizFaixa                    : indica que todas as entradas possuem valores na faixa entre 0 e 1
% [numEscondidos numSaidas]      : indica a quantidade de nodos escondidos e de saida da rede
% {'logsig','logsig'}            : indica que os nodos das camadas escondida e de saida terao funcao de ativacao sigmoide logistica
% 'traingdm','learngdm'          : indica que o treinamento vai ser feito com gradiente descendente (backpropagation)
% 'sse'                          : indica que o erro a ser utilizado vai ser SSE (soma dos erros quadraticos)

% Inicializa os pesos da rede criada (para ajuda, digite 'help init')
net = init(net);
% net = feedforwardnet(numHidden, 'traingdm');

echo on
%   Parametros do treinamento (para ajuda, digite 'help traingd')
net.trainParam.epochs   = 10000;    % Maximo numero de iteracoes
net.trainParam.lr       = 0.2;  % Taxa de aprendizado
net.trainParam.goal     = 0;      % Criterio de minimo erro de treinamento
net.trainParam.max_fail = 10;      % Criterio de quantidade maxima de falhas na validacao
net.trainParam.min_grad = 0;      % Criterio de gradiente minimo
net.trainParam.show     = 10;     % Iteracoes entre exibicoes na tela (preenchendo com 'NaN', nao exibe na tela)
net.trainParam.time     = inf;    % Tempo maximo (em segundos) para o treinamento
echo off
fprintf('\nTreinando ...\n');

validationSetStruct.P = validationInputs; % Entradas da validacao
validationSetStruct.T = validationOutputs;   % Saidas desejadas da validacao

%   Treinando a rede
[newNet,performance,netOutputs,errors] = train(net,trainingInputs,trainingOutputs,[],[],validationSetStruct);
% redeNova   : rede apos treinamento
% desempenho : apresenta os seguintes resultados
%              desempenho.perf  - vetor com os erros de treinamento de todas as iteracoes (neste exemplo, escolheu-se erro SSE)
%              desempenho.vperf - vetor com os erros de validacao de todas as iteracoes (idem)
%              desempenho.epoch - vetor com as iteracoes efetuadas
% saidasRede : matriz contendo as saidas da rede para cada padrao de treinamento
% erros      : matriz contendo os erros para cada padrao de treinamento
%             (para cada padrao: erro = saida desejada - saida da rede)
% Obs.       : Os dois argumentos de 'train' preenchidos com [] apenas sao utilizados quando se usam delays
%             (para ajuda, digitar 'help train')

fprintf('\nTestando ...\n');

%    Testando a rede
[netTestOutputs,Pf,Af,testErrors,testPerformance] = privatesim(newNet,testInputs,[],[],testOutputs);
% saidasRedeTeste : matriz contendo as saidas da rede para cada padrao de teste
% Pf,Af           : matrizes nao usadas neste exemplo (apenas quando se usam delays)
% errosTeste      : matriz contendo os erros para cada padrao de teste
%                  (para cada padrao: erro = saida desejada - saida da rede)
% desempenhoTeste : erro de teste (neste exemplo, escolheu-se erro SSE)

fprintf('MSE para o conjunto de treinamento: %6.5f \n',performance.perf(length(performance.perf)));
fprintf('MSE para o conjunto de validacao: %6.5f \n',performance.vperf(length(performance.vperf)));
fprintf('MSE para o conjunto de teste: %6.5f \n',testPerformance);

%     Calculando a matriz de confusão e a curva ROC referente aos resultados gerados pela
%     rede acima.
[~, C, ~, per] = confusion(testOutputs, netTestOutputs);
disp(C);

%[tpr, fpr, thresholds] = roc(testOutputs, netTestOutputs);

%     Calculando o erro de classificacao para o conjunto de teste
%     (A regra de classificacao e' winner-takes-all, ou seja, o nodo de saida que gerar o maior valor de saida
%      corresponde a classe do padrao).
%     Obs.: Esse erro so' faz sentido se o problema for de classificacao. Para problemas que nao sao de classificacao,
%           esse trecho do script deve ser eliminado.

[netMaxOutputs, netWinnerNode] = max (netTestOutputs);
[targetMaxOutputs, targetWinnerNode] = max (testOutputs);

%      Obs.: O comando 'max' aplicado a uma matriz gera dois vetores: um contendo os maiores elementos de cada coluna
%            e outro contendo as linhas onde ocorreram os maiores elementos de cada coluna.

numMisses = 0;

for padrao = 1 : numTest;
    if netWinnerNode(padrao) ~= targetWinnerNode(padrao),
        numMisses = numMisses + 1;
    end
end

testClassificationError = 100 * (numMisses/numTest);

fprintf('Erro de classificacao para o conjunto de teste: %6.5f\n',testClassificationError);