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