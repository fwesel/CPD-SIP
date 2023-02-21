function K = polynomialKernel(X,Z,degree)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[~,D] = size(X);
K = 1;
for d = 1:D
    K = K.* (1+X(:,d)*Z(:,d)').^degree;
end
end