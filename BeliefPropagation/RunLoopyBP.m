function demoLoopyBP
%Belief Propagation in a multiply-connected graph
W=3; X=2; Y=4; Z=2; [w x y z]=assign(1:4);
pot{1}.variables=[w x]; pot{1}.table=(rand([W X]).^2);
pot{2}.variables=[x y]; pot{2}.table=(rand([X Y]).^2);
pot{3}.variables=[y z]; pot{3}.table=(rand([Y Z]).^2);
pot{4}.variables=[z w]; pot{4}.table=(rand([Z W]).^2);
pot=setpotclass(pot,'array');
set
A = FactorGraph(pot);
opt.tol=10e-5; opt.maxit=10; [marg mess A2]=LoopyBP(pot,opt);
jpot=multpots(pot); fprintf('\nExact and Loopy BP marginals:\n\n')
for i=1:length(potvariables(pot))
    fprintf('%d   Exact     Loopy BP\n',i); disp([table(condpot(jpot,i)) table(condpot(marg{i}))])
end


function newpots=setpotclass(pots,potclass)

if length(pots)==1
  
    newpots=feval(['.' potclass]);
    
    if isstruct(pots)
        if ~isempty(pots)
            newpots.variables=pots.variables;
            newpots.table=pots.table;
        end
    end
    
    if iscell(pots)
        if ~isempty(pots)
            newpots.variables=pots.variables;
            newpots.table=pots.table;
        end
    end
    
else
    if iscell(pots)
        for i=1:length(pots)
            newpots{i}=feval(['.' potclass]);
            if ~isempty(pots{i})
                newpots{i}.variables=pots{i}.variables;
                newpots{i}.table=pots{i}.table;
            end
        end
    else
        for i=1:length(pots)
            newpots(i)=feval(['.' potclass]);
            if ~isempty(pots(i))
                newpots(i).variables=pots(i).variables;
                newpots(i).table=pots(i).table;
            end
        end
        
    end
end