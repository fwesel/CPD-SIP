addpath('../src/ALS')
addpath('../src/kernels')
addpath('../src/utils')
%% Hyperparameters
M = 10;
R = 20;
sweeps = 10;

uciDirectory = "C:\Users\LocalAdmin\Downloads\uci_data\experiments\";
allfolders = dir(uciDirectory);
allfolders(1:2) = [];
rng('default');
for i = 1:length(allfolders)
    file = dir(fullfile(allfolders(i).name));
    filename = string([allfolders(i).name, file.name]);
    disp(filename)
    filefolder = uciDirectory+filename+'\';

    % Load
    X = load(filefolder+filename+'.mat');
    X = X.data;
    [N,D] = size(X);

    % 3-fold CV
    k = 3;
    rng(0);
    c = cvpartition(N,'KFold',k);
    
    trainError = zeros(k,1);
    testError = zeros(k,1);
    approxError = zeros(k,1);
    trainTime = zeros(k,1);
    for fold = 1:k
        rng(fold)
        testIdx = test(c,fold);
        trainIdx = testIdx == 0;
        % Select features and target
        X = load(filefolder+filename+'.mat');
        X = X.data;
        X = X(trainIdx,:);
        Y = X(:,end);
        X = X(:,1:end-1);

        % Scale features in [0,1]^D and standardize responses
        XMin = min(X);
        XMax = max(X);
        YMean = mean(Y);
        YStd = std(Y);
        X = (X-XMin)./(XMax-XMin);
        Y = (Y-YMean)./YStd;
        nonNanCols = ~all(isnan(X));
        X = X(:,nonNanCols);
        % Estimate lengthscale and lambda on a small subset with GPR
        meanfunc = [];                    
        covfunc = @covSEiso;         
        likfunc = @likGauss;
        hyp = struct('mean', [], 'cov', [log(nanmean(std(X))) -0.5], 'lik', log(std(Y))); %#ok<NANMEAN> 
        hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc,covfunc,likfunc,X(1:2000,:),Y(1:2000));
        lengthscale = exp(hyp2.cov(1));
        lambda = exp(hyp2.lik-hyp2.cov(2))^2;
        disp('lengthscale: '+string(lengthscale));
        disp('lambda: '+string(lambda));

        % Train
        kernel = @(X,Z) SE(X,Z,lengthscale);
        tic; tic;
        [W,U] = CPLS(X,Y,M,R,lambda,kernel,sweeps);
        trainTime(fold) = toc;
        trainError(fold) = sqrt(mean((CPPredict(X, W, U, kernel)-Y).^2))/std(Y); 
        
        % Implicit approximation error
        approxError(fold) = approximationError(X,U,kernel);

        % Test
        X = load(filefolder+filename+'.mat');
        X = X.data;
        X = X(testIdx,:);
        Y = X(:,end);
        X = X(:,1:end-1);
        X = (X-XMin)./(XMax-XMin);
        Y = (Y-YMean)./YStd;
        X = X(:,nonNanCols);
        testError(fold) = sqrt(mean((CPPredict(X, W, U, kernel)-Y).^2))/std(Y); 
        
        % Save results
        save(filename+'.mat','trainError','testError','approxError','trainTime');
    end
    disp(mean(trainError));
    disp(mean(testError));
end