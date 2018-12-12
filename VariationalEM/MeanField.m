function [lambda,F] = MeanField(X,mu,sigma,pie,lambda0,maxsteps,tol,performVE)
    
    %performVE is boolean - whether or not to update.
    
    [N,D] = size(X);
    K = size(mu,2);
    lambda = lambda0;
    
    mu_ = mu'*mu;
    pie_ = repmat(pie./(1-pie),N,1);
    logpie_ = log(pie_);
    X_mu = X*mu./(sigma^2);
    mu_diag = diag(mu_);
    mu_diag_mat = repmat(mu_diag',N,1);
    
    onesN = ones(N,1);
    onesNK = ones(N,K);
    onesK = ones(K,1);
    
    nstep=0;
    diff=tol+1;
    F_old = 100000000;
    
    F_const = -D*N/2*log(2*pi*sigma^2) - 1/(2*sigma^2) * (sum(sum(X.*X)));
    
    if(performVE)
        
        while ( (nstep < maxsteps) && (diff > tol) )
            
            for i=1:K
                
                % A is an n*k matrix
                A = logpie_ + X_mu - mu_diag_mat/(2*(sigma^2))       ...
                    + (-lambda*mu_ + mu_diag_mat.*lambda)/(sigma^2);
                
                % Update lambda(n,i) for all n in one go
                lambda(:,i) = onesN ./ ( onesN + exp(-A(:,i)) );
                
            end
            
            %Ensure no numerical issues with lim_{x->0} xlogx
            lambda1=lambda;
            lambda2=lambda;
            lambda1(lambda1==1)=0;
            lambda2(lambda2==0)=1;
            
            %Calculate F efficiently
            F = F_const + 1/(2*sigma^2) * ( 2*sum(sum(lambda.*(X*mu)))                                 ...
                - sum(sum((lambda*mu_).*lambda)) - sum((lambda-lambda.^2)*mu_diag))                    ...
                + sum(lambda*log(pie')) + sum((onesNK-lambda)*log(onesK-pie'))                         ...
                - sum(sum(lambda2.*log(lambda2))) - sum(sum((onesNK-lambda1).*log((onesNK-lambda1))));
            
            diff = abs(F-F_old);
            F_old = F;
            
            % One step is defined as updating lambda_in for all i and n once.
            nstep = nstep+1;
            
        end
        
    else
        %To calculate F without performing VE first.
        
        %Ensure no numerical issues with lim_{x->0} xlogx
        lambda1=lambda;
        lambda2=lambda;
        lambda1(lambda1==1)=0;
        lambda2(lambda2==0)=1;
        
        F = F_const + 1/(2*sigma^2) * ( 2*sum(sum(lambda.*(X*mu)))                                 ...
            - sum(sum((lambda*mu_).*lambda)) - sum((lambda-lambda.^2)*mu_diag))                    ...
            + sum(lambda*log(pie')) + sum((onesNK-lambda)*log(onesK-pie'))                         ...
            - sum(sum(lambda2.*log(lambda2))) - sum(sum((onesNK-lambda1).*log((onesNK-lambda1))));
        
    end
    
end






















function F_n = free_energy(x_n,lambda_n,mu,pie,sigma)
    [D,K]=size(mu);
    ES = lambda_n';
    ESS = lambda_n*lambda_n' + eye(K).*(lambda_n-lambda_n.*lambda_n);
    
    F1 = -(D/2)*log(2*pi*sigma*sigma)-(1/(2*sigma^2))*(trace(ESS*(mu'*mu)) - 2*x_n*mu*ES -x_n*x_n');
    F2 = log(pie)*ES + log(1-pie)*(1-ES);
    
    if(any(lambda_n==0))
        lambda_n(lambda_n==0)=1;
         F3 = log(lambda_n)*ES;
    else
        F3 = log(lambda_n)*ES;
    end
    
    if(any(lambda_n==1))
        lambda_n(lambda_n==1)=0;
        F4 = log(1-lambda_n)*(1-ES);
    else
        F4 = log(1-lambda_n)*(1-ES);
    end
    
    
        
    F5 = F3 + F4;
    
    if(isnan(F1+F2+F3))
        disp('ooooo')
    end
    
    F_n = F1+F2-F5;
end


