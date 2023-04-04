
function [PERF1,WCSP_BOX,PL_BOX] = csp_cv_svm_single(class1, class2,Fs,K, m,rep,svmkernel)
% function containing CSP vanilla + k-fold + SVM
% single band
% class1 cell(1*Ntrials) containing class1 trials
% class2 cell(1*Ntrials) containing class1 trials
% K= 10
% m = 3;
% rep number of k-fold repetition


Ntrials = length(class1);
trial_length = size(class1{1, 1},1); % length of each trial



%% %%%%%%%%%%  iir filtering

% Slow Cortical Potentials(SCP) extraction
[b, a] = butter(3,0.5/(0.5*Fs),'low'); 

cont1 = cell2mat(class1');
cont2 = cell2mat(class2');

temp1 = filtfilt(b,a,cont1);
class1 = mat2cell(temp1,trial_length*ones(1,Ntrials))';


temp2 = filtfilt(b,a,cont2);
class2= mat2cell(temp2,trial_length*ones(1,Ntrials))';
    
shuffled_idx = randperm(length(class1));
class1 = class1(:,shuffled_idx); % 
class2 = class2(:,shuffled_idx);% 

PERF1 = zeros(rep,K);
WCSP_BOX = cell(rep,K);
PL_BOX = cell(rep,K);

for irep = 1:rep

   
    indx1=1:length(class1);
    indx2=1:length(class2);
    ind1=round(linspace(1,length(class1),K+1));
    ind2=round(linspace(1,length(class2),K+1));
    Ftrain1=[];Ftrain2=[];Ftest1=[];Ftest2=[];
    
    PERF=zeros(1,K);
    
    
    for k=1:K
        %fprintf("CV k = %d \n",k)
        % Class 1
        a=indx1;
        b=indx1(ind1(k)):indx1(ind1(k+1)-1);
        a(b)=[];
        Xtr1=class1(a);
        Xte1=class1(b);
        
        %%
        % * 
        % Class 2
        a=indx2;
        b=indx2(ind2(k)):indx2(ind2(k+1)-1);
        a(b)=[];
        Xtr2=class2(a);
        Xte2=class2(b);
        
        
        %%%%%************
        
        [W]=MyCSP(Xtr1,Xtr2,m);

        Ftrain1=[];Ftrain2=[];
        Ftest1=[];Ftest2=[];
        
        %% Training 1
        % 
        % 
        for i=1:length(Xtr1)
            x=Xtr1{i};
            y=x*W;
            f=log10(var(y));
%             [mobility,complexity] = HjorthParameters(y);
%             f = [f,mobility,complexity];
            Ftrain1=[Ftrain1;f];
        end
        %% Training 2
        for i=1:length(Xtr2)
            x=Xtr2{i};
            y=x*W;
            f=log10(var(y));
%             [mobility,complexity] = HjorthParameters(y);
%             f = [f,mobility,complexity];
            Ftrain2=[Ftrain2;f];
        end
        %% Test 1
        for i=1:length(Xte1)
            x=Xte1{i};
            y=x*W;
            f=log10(var(y));
%             [mobility,complexity] = HjorthParameters(y);
%             f = [f,mobility,complexity];
            Ftest1=[Ftest1;f];
        end
        %% Test 2
        for i=1:length(Xte2)
            x=Xte2{i};
            y=x*W;
            f=log10(var(y));
%             [mobility,complexity] = HjorthParameters(y);
%             f = [f,mobility,complexity];
            Ftest2=[Ftest2;f];
        end
        
        
        Ftrain=[Ftrain1;Ftrain2];
        Ftest=[Ftest1;Ftest2];
        GroupTR=[zeros(1,size(Ftrain1,1)),ones(1,size(Ftrain2,1))]';
        GroupTE=[zeros(1,size(Ftest1,1)),ones(1,size(Ftest2,1))]';
        
        [Ftrain,mu,sigma] = zscore(Ftrain);
        Ftest = (Ftest-mu./sigma);
        SVMStruct = fitcsvm(Ftrain,GroupTR,'Standardize',false,'KernelFunction',svmkernel,...
    'KernelScale','auto');
        
%         SVMStruct = fitcsvm(Ftrain,GroupTR);
        pred= predict(SVMStruct,Ftest);
        perf=sum(pred==GroupTE)/length(GroupTE);
        PERF(k)=perf;
        PL = [pred, GroupTE]; % [predicted(output), Label(True target)]
        PL_BOX{irep,k} = PL;
        WCSP_BOX{irep,k} = W;
    end
  PERF1(irep,:) = PERF;
  
  shuffled_idx = randperm(length(class1));
  class1 = class1(1,shuffled_idx);
  class2 = class2(1,shuffled_idx);
  fprintf("irep: %d \n",irep)
  fprintf("number of features: %d \n",size(Ftrain,2))
end
    
end