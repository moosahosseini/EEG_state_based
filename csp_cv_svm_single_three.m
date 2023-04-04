
function [PERF3,CONF_BOX,PERFbin,PredLabelBox,WCSP_BOX] = csp_cv_svm_single_three(class1, class2, class_off,Fs,K, m,rep,svmkernel)
% three calss discrimination based on binary classification with Wcsp
% k fold CV with random  targets (THREE-class)

Ntrials = length(class1);
trial_length = size(class1{1, 1},1); % length of each trial

%rep=1

%% %%%%%%%%%%  iir filtering

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

% off-tagets

N_off_trials = length(class_off);
cont3 = cell2mat(class_off');
temp3 = filtfilt(b,a,cont3);
class3= mat2cell(temp3,trial_length*ones(1,N_off_trials))';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PERF3 = zeros(rep,K);
PERFbin = zeros(rep,K); % performance binary
PredLabelBox = cell(rep,K);
CONF_BOX = [];
WCSP_BOX = cell(rep,K);


for irep = 1:rep

   
    indx1=1:length(class1);
    indx2=1:length(class2);
    ind1=round(linspace(1,length(class1),K+1));
    ind2=round(linspace(1,length(class2),K+1));
    Ftrain1=[];Ftrain2=[];Ftest1=[];Ftest2=[];
    
    PERF=zeros(1,K);
    perf2=zeros(1,K);
    CONF = zeros(3,3,K);
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

            Ftrain1=[Ftrain1;f];
        end
        %% Training 2
        for i=1:length(Xtr2)
            x=Xtr2{i};
            y=x*W;
            f=log10(var(y));

            Ftrain2=[Ftrain2;f];
        end
        %% Test 1
        for i=1:length(Xte1)
            x=Xte1{i};
            y=x*W;
            f=log10(var(y));

            Ftest1=[Ftest1;f];
        end
        %% Test 2
        for i=1:length(Xte2)
            x=Xte2{i};
            y=x*W;
            f=log10(var(y));

            Ftest2=[Ftest2;f];
        end
        
        
        Ftrain=[Ftrain1;Ftrain2];
        Ftest=[Ftest1;Ftest2];
        GroupTR=[zeros(1,size(Ftrain1,1)),ones(1,size(Ftrain2,1))]';% labels[0 1]
        GroupTE=[zeros(1,size(Ftest1,1)),ones(1,size(Ftest2,1))]';% labels[0 1]
        
        % extracting features from off-targets
        temp1 = cellfun(@(x)x*W,class3,'UniformOutput',false);
        temp1 = cellfun(@(x)log10(var(x)),temp1,'UniformOutput',false);
        F_off = cell2mat(temp1');
        
        
        
%         [Ftrain,mu,sigma] = zscore(Ftrain);
%         Ftest = (Ftest-mu./sigma);
        SVMStruct = fitcsvm(Ftrain,GroupTR,'Standardize',true,'KernelFunction',svmkernel,...
    'KernelScale','auto'); %'polynomial' 'PolynomialOrder' defeault 3, positiv integer ,'PolynomialOrder',2
        
%         SVMStruct = fitcsvm(Ftrain,GroupTR);
        [pred, score]= predict(SVMStruct,Ftrain);
        
        score1 = score(1:size(Ftrain1,1),:);
        score2 = score(size(Ftrain1,1)+1:end,:);
        

        
        % calcualting score for test and off-targets
        [pred_test, score_test]= predict(SVMStruct,Ftest);
        [pred_off, score_off]= predict(SVMStruct,F_off);
        



       
        
        perf_test_original=sum(pred_test==GroupTE)/length(GroupTE);
        perf2(1,k) = perf_test_original;
        
        score_three_class = [score_test;score_off];
        label_three_class = [GroupTE;3*ones(N_off_trials,1)];
        

        pred_three_class = NaN(length(label_three_class),1);
        %%%%%%%%%%%%% finding the borders in score space
        %%%%%%%%%%%%% for separating THREE classes
        
        slp = mean(diff(score1(:,2))./diff(score1(:,1))); % slope of the score line
        
       [~,firstIdx] = sort(score1(:,2),'descend');
       first_border = score1(firstIdx(1),:);
       
       [~,secondIdx] = sort(score2(:,2),'descend');
       second_border = score2(secondIdx(end),:);
       
       % THREE-class discrimination law!
       
       for itest=1:length(pred_three_class)
           score_temp = score_three_class(itest,:);
           
           if (first_border(1)>score_temp(1))&&(second_border(1)<score_temp(1))
               pred_three_class(itest) = 3;
           elseif first_border(1)<=score_temp(1)
               pred_three_class(itest) = 0;
           elseif second_border(1)>=score_temp(1)
               pred_three_class(itest) = 1;
           end
           
       end
       
       
       perf_three=sum(pred_three_class==label_three_class)/length(label_three_class);
       
        C = confusionmat(label_three_class,pred_three_class);
        CONF(:,:,k)=C;
        %%%%%% figure them all
        %score1 score2 scoreoff scoretest1 scoretest2
        score_test1 = score_test(1:size(Ftest1,1),:);
        score_test2 = score_test(size(Ftest1,1)+1:end,:);
%         
%         figure,
%         scatter(score1(:,1),score1(:,2),'b')
%         hold on
%         scatter(score2(:,1),score2(:,2),'r')
%         scatter(score_off(:,1),score_off(:,2),'*g')
%         scatter(score_test1(:,1),score_test1(:,2),'*m')
%         scatter(score_test2(:,1),score_test2(:,2),'*y')
%         grid on;grid minor;box on;
%         legend("1&3", '2&4', 'off targets','test1','test2')
        
        
%         pred= predict(SVMStruct,Ftest);
%         perf=sum(pred==GroupTE)/length(GroupTE);
        PERF(k)=perf_three;
        pl = [pred_three_class,label_three_class];
        PredLabelBox{irep,k} = pl;
       
        WCSP_BOX{irep,k} = W;
    end
  PERF3(irep,:) = PERF;
  PERFbin(irep,:)=perf2;
  CONF_BOX = cat(3,CONF_BOX,CONF);
  
  shuffled_idx = randperm(length(class1));
  class1 = class1(1,shuffled_idx);
  class2 = class2(1,shuffled_idx);
  fprintf("irep: %d \n",irep)
  fprintf("number of features: %d \n",size(Ftrain,2))
end
   % figure,stem(sort(PERF3(:),'descend'))
end