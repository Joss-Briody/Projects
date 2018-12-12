function A=FactorGraph(pot)
%Returns a Factor Graph adjacency matrix based on a cell of potentials.
% A = FactorGraph(pot)

pot=str2cell(pot);
F =length(pot); % number of factors (potentials in distribution)
variables=potvariables(pot); % number of variables

V=length(variables);
N=V+F; % all nodes in factor graph

vnodes=zeros(1,N); vnodes(1:V)=1:V; % variable nodes
fnodes=zeros(1,N); fnodes(V+1:end)=1:F; % factor nodes
A = sparse(N,N);
for f=1:length(pot)
    FGnodeA=find(fnodes==f); FGnodeB=pot{f}.variables;
    A(FGnodeA,FGnodeB)=1; A(FGnodeB,FGnodeA)=1;
end


% get a message passing sequence and initialise the messages
[tree elimseq forwardschedule]=istree(A);
reverseschedule=flipud(fliplr(forwardschedule));
schedule=vertcat(forwardschedule,reverseschedule);

if tree
    for count=1:length(schedule)
        % setup the structure for a message from FGnodeA -> FGnodeB
        [FGnodeA FGnodeB]=assign(schedule(count,:));
        A(FGnodeA,FGnodeB)=count; % store the FG adjacency matrix with mess number on edge
    end
else
    A = replace(A,1,-1);
end


function c=str2cell(x)
if iscell(x); c=x; return;
else
    for i=1:length(x)
        c{i}=x(i);
    end
end


function [variables nstates con convec]=potvariables(inpot)

pot=str2cell(inpot);

if isempty(pot);return;end

v=cell2mat(cellfun(@cellvariables,pot,'UniformOutput',false));

variables=[];nstates=[]; con=1;convec=[];

if ~isempty(v)
    [a b]=sort(v); i=[b(diff(a)>0) b(end)];variables=v(i);
    N=max(variables);
    convec=ones(1,N);nstates=-ones(1,N);
    for p=1:length(pot)
        if ~isempty(pot{p}.variables)
            nstates(1,pot{p}.variables) = size(pot{p});
        end
        if p>1
            convec(nstates(oldnstates>-1)~=oldnstates(oldnstates>-1))=0;
        end
        oldnstates=nstates;
    end
    con = all(convec); 
else
    variables=[]; nstates=[];
end
nstates=nstates(nstates>0);