function [W, U] = CPLS(X, y, M, R,lambda, kernel, numberSweeps)
[~, D] = size(X);
W = cell(1,D);
Matd = 1;
reg = 1;
w = linspace(0,1,M)';
U = kernel(w,w);
[~,S,U] = svd(U);
U = sqrt(S)*U';
    for d = D:-1:1
        W{d} = randn(M,R);
        W{d} = W{d}./vecnorm(W{d},2,1);
        reg = reg.*(W{d}'*W{d});
        Mati = features(X(:,d),U,kernel);
        Matd = (Mati*W{d}).*Matd;
    end
    for ite = 1:2*numberSweeps % consistent with Wesel and Batselier 2021
        for d = 1:D
        Mati = features(X(:,d),U,kernel);
        reg = reg./(W{d}'*W{d});
        Matd = Matd./(Mati*W{d});
        [CC,Cy] = dotkronLargeScale(Mati,Matd,y);
        x = (CC+lambda*kron(reg,eye(M)))\Cy;
        clear CC Cy
        W{d} = reshape(x,M,R);
        scaling = vecnorm(W{d},2,1);
        W{d} = W{d}./scaling;
        reg = reg.*(W{d}'*W{d});
        Matd = Matd.*(Mati*W{d});
        end
    end
    W{d} = W{d}.*scaling;
end