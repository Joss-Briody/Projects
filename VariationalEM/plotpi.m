function plotpi(a,Fvec)
    
    plot(Fvec);
        figure
    
    for i=1:size(a,2)
        subplot(4,2,i);
        imagesc(reshape(a(:,i),4,4),[0 2]);
        axis off;
        axis equal;
    end;
    
end