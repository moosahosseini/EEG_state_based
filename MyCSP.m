function [W] = MyCSP(Xtr1,Xtr2,m)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
C1=0;
C2=0;
Ntr1=length(Xtr1);
Ntr2=length(Xtr2);
for itr=1:max(Ntr1,Ntr2)
    if itr<=Ntr1
        x=Xtr1{itr};
        for ich=1:size(x,2)
            x(:,ich)=x(:,ich)-mean(x(:,ich));
        end
%         c=x'*x/size(x,1);
        c=x'*x/trace(x'*x);
        C1=C1+c;
    end
    if itr<=Ntr2
        x=Xtr2{itr};
        for ich=1:size(x,2)
            x(:,ich)=x(:,ich)-mean(x(:,ich));
        end
%         c=x'*x/size(x,1);
        c=x'*x/trace(x'*x);
        C2=C2+c;
    end
end
Ndim=size(C1,1);
C1=C1/Ntr1;
C2=C2/Ntr2;
Cc=C1+C2;
[Uc,Lc]=eig(Cc);
[Lc,indx]=sort(diag(Lc),'descend');
Lc=diag(Lc);
Uc=Uc(:,indx);
G=(Lc^-0.5)*Uc';
S1=G*C1*G';
S2=G*C2*G';
[U1,L1]=eig(S1);
[L1,indx]=sort(diag(L1),'descend');
L1=diag(L1);
U1=U1(:,indx);
Wcsp=G'*U1;
W=Wcsp(:,[1:m,Ndim-m+1:Ndim]);
end

