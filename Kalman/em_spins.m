function likelihood = spins_EM(M)
    
    load ssm_spins.txt -ascii;
    xx  = ssm_spins;
    
    K = 4;
    D = 5;
    it_num = 50;
    
    Y0 = randn(K,1);  % Known a priori that y1 ~ N(0,I)
    Q0 = eye(K);
    %==========================================================================
    % Generating parameters
    A = 0.99*[cos(2*pi/180) , -sin(2*pi/180), 0             , 0 ;
        sin(2*pi/180) ,  cos(2*pi/180), 0             , 0 ;
        0             ,  0            , cos(2*pi/90) ,  -sin(2*pi/90);
        0             ,  0            , sin(2*pi/90)  , cos(2*pi/90)];
    
    Q = eye(K) - A*A';
    R = eye(D);
    
    C = [1,   0,  1,   0;
        0,   1,  0,   1;
        1,   0,  0,   1;
        0,   0,  1,   1;
        0.5, 0.5, 0.5, 0.5];
    %==========================================================================
    % Initialise A,Q,C,R
    
    % Covariance matrices need to be symmetric, positive definite.
    Q = M*rand(K);
    Q = Q*Q';
    R=rand(D);
    R=M*R*R';
    
    % Transimssion and emission matrices can be anything.
    C = M*rand(D,K);
    A = rand(K,K);
    %==========================================================================
    
    lik = zeros(it_num,1);
    
    for i=1:it_num
        
        %E step
        [yhat,Vhat,Vjoint,like] = ssm_kalman(xx',Y0,Q0,A,Q,C,R, 'smooth');
        
        %M step
        [A,Q,C,R] = M_step(xx,yhat,Vhat,Vjoint);
        
        lik(i)=sum(like);
        
    end
    
    likelihood = lik;
    
end

%==========================================================================
%==========================================================================
%Use analytic expressions for matrix updates

function [A,Q,C,R] = M_step(xx,yhat,Vhat,Vjoint)
    
    cellsum = @(C)(sum(cat(3,C{:}),3));
    
    sum_y = yhat*yhat';
    
    % Calculate C_new
    C_1 = (yhat*xx)';              % \sum(x_t*<y_t>')
    Vy = cellsum(Vhat)+ sum_y;     % \sum(<y_t*y_t'>)
    C = (Vy\C_1')';                % \sum(<y_t>*x_t)*Inv[sum(<y_t*y_t'>)]
    
    % Calculate R_new
    R1 = xx'*xx;                   % \sum(x_t*x_t')
    R2 = C_1*C';                   % \sum(x_t*<y_t>')*C'
    R = (R1-R2)/size(xx,1);
    R = 1/2*(R+R');                % symmetrise to avoid numerical errors
    
    %Calculate A_new
    y_t_y = zeros(size(cellsum(Vhat))); % <y_t*y_t+1'> = Cov(y_t,y_t+1) - <y_t><y_t'>
    
    for j=1:length(Vjoint)
        y_t_y = y_t_y + yhat(:,j+1)*yhat(:,j)';
    end
    
    sumVj = cellsum(Vjoint)+y_t_y;  % \sum(<y_t+1*y_t'>)
    A = (Vy'\sumVj')';              % \sum(<y_t+1*y_t'>)*Inv[sum(<y_t*y_t'>)]
    
    % Calculate Q_new
    sum_y2 = sum_y-yhat(:,1)*yhat(:,1)';         %sum starts from t=2
    sumVy2 = cellsum(Vhat(2:end))+sum_y2;        %\sum_{t=2}<y_t*y_t'>
    Q = (sumVy2 - sumVj*A') / (length(Vjoint));
    Q = 1/2*(Q+Q');                              % symmetrise to avoid numerical errors
    
end
