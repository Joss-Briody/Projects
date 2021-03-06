function [yhat, Vhat, Vjoint, like] = ssm_kalman(xx, y0, Q0, A, Q, C, R, pass)
% kalman-smoother estimates of SSM state posterior
%  y0 Kx1 - initial latent state
%  Q0 KxK - initial variance
%  A  KxK - latent dynamics matrix
%  Q  KxK - innovariations covariance matrix
%  C  DxK - output loading matrix
%  R  DxD - output noise matrix
% Returns:
%  Y  KxT - posterior mean estimates
%  V  1xT cell array of KxK matrices - posterior variances on y_t
%  Vj 1xT-1 cell array of KxK matrices - posterior covariances between y_{t+1}, y_t
%  L  1xT - conditional log-likelihoods log(p(x_t|x_{1:t-1}))

% default pass is smooth
if nargin < 8
  pass = 'smooth';
end
% check dimensions
[dd,kk] = size(C);
[tt] = size(xx, 2);

if any([size(y0) ~= [kk,1], ...
        size(Q0) ~= [kk,kk], ...
        size(A) ~=  [kk,kk], ...
        size(Q) ~=  [kk,kk], ...
        size(R) ~=  [dd,dd]])
  error ('inconsisent parameter dimensions');
end

%%%% allocate arrays
yfilt = zeros(kk,tt);                  % filtering estimate: \hat(y)_t^t
Vfilt = cell(1,tt);                    % filtering variance: \hat(V)_t^t
yhat  = zeros(kk,tt);                  % smoothing estimate: \hat(y)_t^T
Vhat  = cell(1,tt);                    % smoothing variance: \hat(V)_t^T
K = cell(1, tt);                       % Kalman gain
J = cell(1, tt);                       % smoothing gain
like = zeros(1, tt);                   % conditional log-likelihood: p(x_t|x_{1:t-1})
Ik = eye(kk);

%%%% forward pass
Vpred = Q0;
ypred = y0;

for t = 1:tt
  xprederr = xx(:,t) - C*ypred;
  Vxpred = C*Vpred*C'+R;
  like(t) = -0.5*logdet(2*pi*(Vxpred)) - 0.5*xprederr'/Vxpred*xprederr;
  K{t} = (Vpred*C')/(C*Vpred*C' + R);
  yfilt(:,t) = ypred + K{t}*xprederr;
  Vfilt{t} = Vpred - K{t}*C*Vpred;
  %% symmetrise the variance to avoid numerical drift
  Vfilt{t} = (Vfilt{t} + Vfilt{t}')/2;
  ypred = A*yfilt(:,t);
  Vpred = A*Vfilt{t}*A' + Q;
end

%%%% backward pass
if (strncmp(lower(pass), 'filt', 4) || strncmp(lower(pass), 'forw', 4))
  % skip if filtering/forward pass only
  yhat = yfilt;
  Vhat = Vfilt;
  Vjoint = {};
else
  yhat(:,tt) = yfilt(:,tt);
  Vhat{tt}   = Vfilt{tt};

  for t = tt-1:-1:1
    J{t} = (Vfilt{t}*A')/(A*Vfilt{t}*A' + Q);
    yhat(:,t) = yfilt(:,t) + J{t}*(yhat(:,t+1) - A*yfilt(:,t));
    Vhat{t}   = Vfilt{t} + J{t}*(Vhat{t+1} - A*Vfilt{t}*A' - Q)* J{t}';
  end

  Vjoint{tt-1} = (Ik - K{tt}*C)*A*Vfilt{tt-1};
  for t = tt-2:-1:1
    Vjoint{t} = Vfilt{t+1}*J{t}' + J{t+1}*(Vjoint{t+1} - A*Vfilt{t+1})*J{t}';
  end
end


function lgdt = logdet(A)
% Compute the logarithm of the determinant of A efficiently.  
% A must be positive definite or chol will raise an error.

R = chol(A);
lgdt = 2*sum(log(diag(R)));