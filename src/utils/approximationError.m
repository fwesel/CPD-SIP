function error = approximationError(X,U,kernel)
    perm = randperm(min(1000,size(X,1)));
    XPerm = X(perm,:);
%     K = exp(-0.5*pdist2(XPerm,XPerm).^2/kernel^2);
    K = kernel(XPerm,XPerm);
    KApprox = ones(size(K));
    for d = 1:size(XPerm,2)
        phi = features(XPerm(:,d),U,kernel);
        KApprox = KApprox.*(phi*phi');
    end
    error = norm(K-KApprox,'fro')/norm(K,'fro');
end
