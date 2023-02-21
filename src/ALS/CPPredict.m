function score = CPPredict(X, W, U, kernel)
    [N,D] = size(X);
    M = size(W{1},1);
    score = ones(N,1);
    for d = 1:D
        score = score.*(features(X(:,d),U,kernel)*W{d});
    end
    score = real(sum(score,2));
end