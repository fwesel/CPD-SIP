addpath('../src/ALS')
addpath('../src/kernels')
addpath('../src/utils')
%% Banana Dataset Plots
M = 50;
RSet = [2,3,6,M];
lambda = 1e-6;
kernel = @(X,Z) polynomialKernel(X,Z,5);
NIte = 10;
%% Generate Plots
close all
rng('default');
% warning off
X = readmatrix('banana.csv');
N = size(X,1);
X = X(randperm(size(X,1)),:);
Y = (X(:,end)==1)-(X(:,end)==2);
X = X(:,1:2);
XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

NPlot = 100;
X1Plot = linspace(min(X(:,1)),max(X(:,2)),NPlot);
[X1Plot,X2Plot] = meshgrid(X1Plot,X1Plot);
XPlot = [X1Plot(:),X2Plot(:)];



% Low-rank exact
Z = ones(size(X,1),1);
ZPlot = ones(size(XPlot,1),1);
w = linspace(min(X(:,1)),max(X(:,1)),M)';

[X1II, X2II] = meshgrid(w',w');
K = kernel(X,X);
wKRR = (K+lambda*eye(N))\Y;
scorePlotKRR = sign(kernel(XPlot,X)*wKRR);


%% Plots
plotIdx = 0;
for R = RSet
    rng(plotIdx);
    plotIdx = plotIdx+1;tic;
    [W,U] = CPLS(X,Y,M,R,lambda,kernel,NIte);toc;
    scorePlotCP = sign(CPPredict(XPlot,W,U,kernel));
    
    figure(plotIdx);
    fig = gcf;
    hold on
    s3 = scatter(X1II(:),X2II(:),36,'black','x');
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[216, 27, 96]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[30, 136, 229]/255,'filled');
    s1.MarkerFaceAlpha = 0.22;
    s2.MarkerFaceAlpha = 0.22;
%     c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'color',[255, 193, 7]/255,'LineWidth',1.5,'LineStyle','-.');
    c3 = contour(X1Plot,X2Plot,reshape(scorePlotKRR,size(X1Plot)),[0 0],'black','LineWidth',2,'LineStyle','--');
    c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(X1Plot)),[0 0],'black','LineWidth',2.5);
    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    axis equal
    axis off
    hold off
    filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';
%     exportgraphics(fig,filename,'BackgroundColor','none','ContentType','vector');
end