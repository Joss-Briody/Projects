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