function newpot = condpot(pot,varargin)
%Return a potential conditioned on another variable
pot=str2cell(pot);
if length(varargin)>0
    x=varargin{1};
    if length(varargin)==1
        y=[];
    else
        y=varargin{2};
    end
    for p=1:length(pot)
        if isempty(y)
            newpot{p}=sum(pot{p},setdiff(pot{p}.variables,x));
            newpot{p}=divpots(newpot{p},brml.sumpot(newpot{p}));
        else
            pxy=sum(pot{p},setdiff(pot{p}.variables,[x(:)' y(:)']));
            py =sum(pxy,x);
            newpot{p}=brml.divpots(pxy,py); % p(x|y) = p(x,y)/p(y)           
        end
    end
else
    newpot=pot;
    for p=1:length(pot)
        newpot{p}=divpots(newpot{p},sumpot(newpot{p}));
    end
end
if length(pot)==1
    newpot=newpot{1};
end


function newpot=divpots(pota,potb)
newpot=pota/potb;