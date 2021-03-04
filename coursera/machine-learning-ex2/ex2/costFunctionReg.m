function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hy = sigmoid(X*theta);
J = sum((-y).*log(hy).-((1.-y).*log(1.-hy)), 1)/m + lambda/(2*m)*sum(theta(2:end).^2);

for c = 1:size(X, 2),
    grad(c) = sum((hy-y).*X(:, c), 1)/m + (lambda/m)*theta(c)*(c!=1);
end
grad = X'*(hy-y)./m;
disp(grad)
grad( 2:end, :) = grad(2:end, :) .+ (lambda/m).*theta(2:end, :);
disp(grad)

% =============================================================

end
