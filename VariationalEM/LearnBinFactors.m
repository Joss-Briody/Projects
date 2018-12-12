function [mu, sigma, pie, Fvec] = LearnBinFactors(X,K,iterations)
    
    [N,D]=size(X);
    maxsteps = 100; % ie the maximum # updates for each lambda_in
    
    Fvec = zeros(iterations+1,1);
    tol = 10^-20;
    eps = 10^-8;
    
    [sigma, mu, pie, lambda] = init_variables(D,K,N);
    
    %Calculate the free energy before any updates
    [~,Fvec(1)] = MeanField4(X,mu,sigma,pie,lambda,maxsteps,tol,false);
    F_old = Fvec(1);
    
    for EM_step = 1:iterations
        
        %Factored VE step
        [lambda,F] = MeanField(X,mu,sigma,pie,lambda,maxsteps,tol,true);
        Fvec(EM_step + 1) = F;
        
        if(F + eps < F_old)
            msg = 'The Free energy has decreased!';
            error(msg)
        end
        
        F_old = F;
        
        %M-step only requires expected sufficient statistics under current
        %posterior
        ES = lambda;
        ESS = lambda'*lambda + sum(lambda-lambda.^2).*eye(K);
        ESS = 0.5*(ESS+ESS');
        
        [mu, sigma, pie] = MStep(X,ES,ESS);
        
    end
    
    plotpi(mu,Fvec);
    
end

function [sigma,mu,pie,lambda] = init_variables(D,K,N)
    % Initialise variables from a Dirichlet distribution
    
    sigma = 2;
    
    pie0 = gamrnd(1,1,1,K);
    pie0 = pie0 ./ sum(pie0,2);
    pie=pie0;
    
    mu0 = gamrnd(1,1,D,K);
    for i=1:D
        mu0(i,:) = mu0(i,:) / sum(mu0(i,:));
    end
    mu = mu0;
    
    lambda0 = gamrnd(1,1,N,K);
    for i=1:N
        lambda0(i,:) = lambda0(i,:) / sum(lambda0(i,:));
    end
    lambda = lambda0;
    
end