function c=str2cell(x)
if iscell(x); c=x; return;
else
    for i=1:length(x)
        c{i}=x(i);
    end
end