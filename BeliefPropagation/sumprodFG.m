function [marg mess normconstpot]=sumprodFG(pot,A,varargin)
%SUMPRODFG Sum-Product algorithm on a Factor Graph represented by A

variables=potvariables(pot);
V=length(variables); N=size(A,1);
fnodes=zeros(1,N); fnodes(V+1:end)=1:N-V; % factor nodes
vnodes=zeros(1,N); vnodes(1:V)=1:V; % variable nodes
nmess=full(max(max(A))); % number of messages
if nargin==2; initmess=[]; else initmess=varargin{1};end
if ~isempty(initmess); mess=initmess; end
if isempty(initmess) % message initialisation
    for count=1:nmess
 
        mess{count}=const(1);
        [FGnodeA FGnodeB]=find(A==count);
        if fnodes(FGnodeA)>0 % factor to variable message:
            
            if length(find(A(FGnodeA,:)))==1
                mess(count)=pot(fnodes(FGnodeA));    
        end
    end
end
 
% Do the message passing:
for count=1:length(mess)
    [FGnodeA FGnodeB]=find(A==count);
    FGparents=setdiff(find(A(FGnodeA,:)),FGnodeB); % FG parent nodes of FGnodeA
    
    if ~isempty(FGparents)
        tmpmess = multpots(mess(A(FGparents,FGnodeA))); % product of incoming messages
        
        factor=fnodes(FGnodeA);
        if ~factor % variable to factor message:
            mess{count}=tmpmess;
        else % factor to variable message:
            tmpmess = multpots([{tmpmess} pot(factor)]);
            mess{count} = sumpot(tmpmess,FGnodeB,0);
        end   

    end
end
% Get all the marginals: variable nodes are first in the ordering, so
for i=1:V
    [dum1 dum2 incoming]=find(A(:,i));
    tmpmess = multpots(mess(incoming));
    marg{i}=tmpmess;
end
normconstpot=sumpot(multpots(mess(mess2var(1,A))));


function newpot = sumpot(pot,varargin)
% sum potential over variables
sumover=1; 
pot=brml.str2cell(pot);
if nargin==1; invariables=[]; sumover=0;
else
    invariables=varargin{1};
end
if length(varargin)==2
    sumover=varargin{2};
end
if length(pot)>1
for p=1:length(pot)
    if sumover
        variables=invariables; 
    else
        variables=setdiff(pot{p}.variables,invariables); % variables that will be summed over
    end
    newpot{p}=sum(pot{p},variables);
end
else
    if sumover
        variables=invariables;  % variables that will be summed over
    else
        variables=setdiff(pot{1}.variables,invariables); % variables that will be summed over
    end
    newpot=sum(pot{1},variables);
end