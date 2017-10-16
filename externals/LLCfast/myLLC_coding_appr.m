% ========================================================================
% USAGE: [Coeff]=LLC_coding_appr(B,X,knn,lambda)
% Approximated Locality-constraint Linear Coding
%
% Inputs
%       B       -M x d codebook, M entries in a d-dim space
%       X       -N x d matrix, N data points in a d-dim space
%       knn     -number of nearest neighboring
%       lambda  -regulerization to improve condition
%
% Outputs
%       Coeff   -N x M matrix, each row is a code for corresponding X
%
% Jinjun Wang, march 19, 2010
% ========================================================================

function [Coeff] = myLLC_coding_appr(B, X, knn, beta)

if ~exist('knn', 'var') || isempty(knn),
    knn = 5;
end

if ~exist('beta', 'var') || isempty(beta),
    beta = 1e-4;
end


% find k nearest neighbors

D = slmetric_pw(X',B','eucdist');
%IDX = zeros(nframe, knn);

% for i = 1:nframe,
% 	d = D(i,:);
% 	[dummy, idx] = sort(d, 'ascend');
% 	IDX(i, :) = idx(1:knn);
% end
[dummy, IDX] = mink(D,knn,2);


%Coeff = zeros(10,10);
% llc approximation coding

Coeff = llcsub(IDX-1,B,X,beta);



