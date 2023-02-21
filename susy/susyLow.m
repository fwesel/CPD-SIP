addpath('../src/ALS')
addpath('../src/kernels')
addpath('../src/utils')
%% SusyLow Dataset
M = 20;
NIte = 5;
NSweeps = 10;
RSet = [5,10,15,20];
trainAUC = zeros(NIte,numel(RSet));
testAUC = zeros(NIte,numel(RSet));
trainAccuracy = zeros(NIte,numel(RSet));
testAccuracy = zeros(NIte,numel(RSet));
discretizationError = zeros(NIte,numel(RSet));
time = zeros(NIte,numel(RSet));
%% Train/Test
warning('off','all');
RIdx = 0;
for R = RSet
    RIdx = RIdx+1;
    for ite = 1:NIte
        rng(ite+R);
        X = readmatrix('susy.csv');
        N = size(X,1);
        X = X(1:floor(N*0.9),:);
        Y = X(:,1);
        X = X(:,2:9);
        
        Y = (Y==1)-(Y==0);
        XMin = min(X);  XMax = max(X);
        X = (X-XMin)./(XMax-XMin);

        lambda = 2e-5;
        lengthscale = mean(std(X));

        disp("N: "+string(N)+" R: "+string(R)+" ite: "+string(ite));
        kernel = @(X,Z) SE(X,Z,lengthscale);
        tic;tic;
        [W, U] = CPLS(X,Y,M,R,lambda,kernel,NSweeps);
        time(ite,RIdx) = toc;toc;
        [~,~,~,trainAUC(ite,RIdx)] = perfcurve(Y,CPPredict(X, W, U, kernel),1);
        trainAccuracy(ite,RIdx) = mean(Y==sign(CPPredict(X, W, U, kernel)));
        discretizationError(ite,RIdx) = approximationError(X,U,kernel);

        % Test
        clear X Y
        X = readmatrix('SUSY.csv');
        X = X(floor(N*0.9)+1:end,:);
        Y = X(:,1);
        X = X(:,2:9);
        Y = (Y==1)-(Y==0);
        X = (X-XMin)./(XMax-XMin);
        [~,~,~,testAUC(ite,RIdx)] = perfcurve(Y,CPPredict(X, W, U, kernel),1);
        testAccuracy(ite,RIdx) = mean(Y==sign(CPPredict(X, W, U, kernel)));
        disp('Test error: '+string(testAUC(ite,RIdx)));
        save('susyLow.mat','trainAUC','testAUC','trainAccuracy','testAccuracy','discretizationError','time');
    end
end