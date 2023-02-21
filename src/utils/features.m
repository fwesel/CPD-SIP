function Mati = features(X,U,kernel)
%     Mati = kernel(X,linspace(0,1,size(U,1))')/U;
Mati = kernel(X,linspace(0,1,size(U,1))')/U;
end