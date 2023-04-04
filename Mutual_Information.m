function [score] = Mutual_Information(featuretrain,Labelstrain)
% featuretrain: timesamples*features
% Labelstrain: labels*1 (column vector)
featuretrain = featuretrain';
Labelstrain = Labelstrain';
yi= Labelstrain;
nbins=35;
n= size(featuretrain,2);
for i=1:size(featuretrain,1)
    xi= featuretrain(i,:);
    %%  step 1: calculate histogram of x & y, hx,hy
    hx= hist(xi,nbins);
    hy= hist(yi,nbins);
    %% step 2: calculate pdf of x & y, px,py
    px=hx/n ;
    py=hy/n ;
    %% step 3: calculate bivariate histogram of x & y, hxy
    X= [xi',yi'];
    hxy= hist3(X,[nbins,nbins]);
    %% step 4: calculate jonit pdf of x & y, pxy
    pxy_joint= hxy/n;
    pxy_indep= px'*py;
    tp= pxy_joint.* log2(pxy_joint./pxy_indep );

    MI(i) = nansum(tp(:));
%     index= find(isnan(tp)==0);
%     MI(i)=sum(tp(index));
end
score= MI;
end 

