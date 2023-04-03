% this script generates classification results for each subject (three-class scenario)
% class 1 (movement along x-axis), class 2 (movement along y-axis),
% class 3 (random targets)
% Author: Seyyed Moosa Hosseini
% paper: State-Based Decoding of Continuous Hand Movements using EEG
% Signals (IEEE access)
% train model on nine training folds >>> finding Wcsp
% finding borders in score space for THREE classes (three-class discrimination rule)
% projecting test fold and non-principals with Wcsp
% classifing test targets based on score space rule!

%% 
tic
clc
clear
close all



remove_IDX = [1 2 3 4 5 33 34 61 62 63];
selected_ch = 1:63;
selected_ch(remove_IDX)=[];

%ldakernel = 'quadratic';% 'pseudoquadratic' 'diagquadratic' 'quadratic' 'linear'
svmkernel = 'rbf';
sub_no = '6';
Fs = 512;
string_str = '\clean_data\COT_str_run_';


savedir = 'D:\Thesis\NBML_DATA\cot_analysis2\final_results';
save_name = strcat(savedir,'\sub_','14','_NEW_cls_results_QDA.mat');

alignment = 'go_cue'; %= 'go_cue'
run_numbers = [1 2 3 4];
sp_filter = 'off'; % lap car off
remove_channels ='on'; % 'off' 'on'


T_before = floor(0.5*Fs); % time before alignement point
T_after = floor(0*Fs);%  time after alignement point

currentdir = 'D:\Spring402\NBML_DATA';


% continous data extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
EEG_Trials_Complex = []; % contains eeg data T_before and T_after movement onset!
off_target_EEG_Trials_Complex = []; % contains eeg data from selected runs!
VEL_Trials = []; % contains velocity data  from selected runs!
POS_Trials = []; % contains positions data  from selected runs!
off_target_POS_Trials = []; 
Target_ind_Trials = [];
EEG_base_Trials = [];
EEG_FB = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for ir=run_numbers
    runstr = num2str(ir);
    dirCOT_str = strcat(currentdir,'\sub_',sub_no,string_str,runstr,'.mat');
    fprintf("dir: %s \n",dirCOT_str)
    load(dirCOT_str)
    target_labels = COT_str.Mov.target_labels;
    principal_target_ind = (target_labels<5);
    
    target_indices = target_labels(principal_target_ind);
    
    temp_pos = COT_str.Mov.pos(principal_target_ind);
    temp_vel = COT_str.Mov.vel(principal_target_ind);
    
    eeg_fb_temp = COT_str.EEG.EEG_FB(principal_target_ind);
    eeg_pre_temp = COT_str.EEG.EEG_Pre(principal_target_ind);
    eeg_base_temp = COT_str.EEG.EEG_baseline(principal_target_ind);
    
    touch_ind = COT_str.Mov.touch_ind_FB(principal_target_ind);
    
    % off-target
    off_target_ind = (target_labels>=5);
    off_target_temp_pos = COT_str.Mov.pos(off_target_ind);
    off_target_temp_vel = COT_str.Mov.vel(off_target_ind);
    
    off_target_eeg_fb_temp = COT_str.EEG.EEG_FB(off_target_ind);
    off_target_eeg_pre_temp = COT_str.EEG.EEG_Pre(off_target_ind);
    off_target_eeg_base_temp = COT_str.EEG.EEG_baseline(off_target_ind);
    
    
    %%% building eeg complx with fb and pre
    if (T_after~=0)&&(T_before~=0)
        
        % targets
        eegafter = cellfun(@(x)x(1:T_after,:),eeg_fb_temp,'UniformOutput',false);
        eegbefore = cellfun(@(x)x(end-T_before:end-1,:),eeg_pre_temp,'UniformOutput',false);
        eegcomplex_temp= cell(1,length(eegbefore));
        
        for ii=1:length(eegbefore)
            eegcomplex_temp{1,ii} = [eegbefore{1,ii};eegafter{1,ii}];
        end
        
        % off-targets
        
        off_target_eegafter = cellfun(@(x)x(1:T_after,:),off_target_eeg_fb_temp,'UniformOutput',false);
        off_target_eegbefore = cellfun(@(x)x(end-T_before:end-1,:),off_target_eeg_pre_temp,'UniformOutput',false);
        off_target_eegcomplex_temp= cell(1,length(off_target_eegbefore));
        
        for ii=1:length(off_target_eegbefore)
            off_target_eegcomplex_temp{1,ii} = [off_target_eegbefore{1,ii};off_target_eegafter{1,ii}];
        end
        
        
    elseif T_before==0
        
        %targets
%         fprintf("just after \n")
        eegafter = cellfun(@(x)x(1:T_after,:),eeg_fb_temp,'UniformOutput',false);
        eegcomplex_temp = eegafter;
        
        %off targets
        
%         fprintf("just after \n")
        off_target_eegafter = cellfun(@(x)x(1:T_after,:),off_target_eeg_fb_temp,'UniformOutput',false);
        off_target_eegcomplex_temp = off_target_eegafter;
        
    elseif T_after==0
        
        %target
%         fprintf("just before \n")
        eegbefore = cellfun(@(x)x(end-T_before+1:end,:),eeg_pre_temp,'UniformOutput',false);
        eegcomplex_temp = eegbefore;
        
        %off target
        
%         fprintf("just before \n")
        off_target_eegbefore = cellfun(@(x)x(end-T_before+1:end,:),off_target_eeg_pre_temp,'UniformOutput',false);
        off_target_eegcomplex_temp = off_target_eegbefore;
    end
    %%%%%%%%%%%%%%%%%%%%  SPATIAL FILTERING
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%% remove channels
    if strcmp(remove_channels,'on')
        eegcomplex_temp = cellfun(@(x)x(:,selected_ch),eegcomplex_temp,'UniformOutput',false);
        off_target_eegcomplex_temp = cellfun(@(x)x(:,selected_ch),off_target_eegcomplex_temp,'UniformOutput',false);
        fprintf("\n some channels removed \n")
    end
    
    EEG_Trials_Complex = [EEG_Trials_Complex,eegcomplex_temp];
    Target_ind_Trials = [Target_ind_Trials, target_indices];
    POS_Trials = [POS_Trials,temp_pos];
    VEL_Trials = [VEL_Trials,temp_vel];
    % off-target
    off_target_EEG_Trials_Complex = [off_target_EEG_Trials_Complex,off_target_eegcomplex_temp];
    off_target_POS_Trials = [off_target_POS_Trials,off_target_temp_pos];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% define Binary Classes { class 1 = (T1 and T3) - class 2 = (T2 and T4) }

class_one_idx = sort([find((Target_ind_Trials==1)), find((Target_ind_Trials==3))]);
class_two_idx = sort([find((Target_ind_Trials==2)), find((Target_ind_Trials==4))]);

pos1 = POS_Trials(1,class_one_idx);
pos2 = POS_Trials(1,class_two_idx);
%%% from onset
EEG_class1 = EEG_Trials_Complex(1,class_one_idx);
EEG_class2 = EEG_Trials_Complex(1,class_two_idx);
EEG_class_off = off_target_EEG_Trials_Complex;
pos_off = off_target_POS_Trials;

%
% fbcsp_cv_svm(class1, class2,Fs,K, m,FFF,FSMETHOD,rep)
% rng(1)
rng("default")
m= 3;
minlength = min(length(EEG_class1),length(EEG_class2));
FFF = 42;
FSMETHOD = 'ardiMI'; %% 'mrmr', 'ardiMI', 'rlf', 'cna', 'ttest', 'entropy', 'wlx', 'roc'

% [PERF3,CONF_BOX,PERFbin,PredLabelBox,WCSP_BOX] = csp_cv_svm_single_three_aug(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs,10, m,30,svmkernel);

[PERF3,CONF_BOX,PERFbin,PredLabelBox,WCSP_BOX] = csp_cv_svm_single_three(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs,10, m,10,svmkernel);



PredLabelBox = PredLabelBox';
PredLabelBox = PredLabelBox(:);




PL = cell2mat(PredLabelBox);
C0 = confusionmat(PL(:,2),PL(:,1));
Ctot = C0./sum(C0,2)

fprintf("\n true positive rate for class 1 = %d %% \n", fix(Ctot(1,1)*100));
fprintf("\n true positive rate for class 2 = %d %% \n", fix(Ctot(2,2)*100));
fprintf("\n true positive rate for class 3 = %d %% \n", fix(Ctot(3,3)*100));
fprintf("\n")
fprintf("three-class accuracy = %d %%\n",fix(sum(PL(:,1)==PL(:,2))/size(PL,1)*100));
adasdadada
meanvector = mean(PERF3);
disp(["mean: ", num2str(mean(meanvector)*100)])
disp(["T_before  ",alignment, num2str(T_before/Fs)])
disp(["T_after  ", alignment, num2str(T_after/Fs)])
disp(["Fs: ", Fs])
disp(["sub number: " sub_no])

fprintf("end of first part \n")

% 
% IDXall = cell2mat(Selected_IDX_Box); 
% IDXall = IDXall(:);
% h = histogram(IDXall);
% grid on; grid minor;
% VAL = h.Values;
% sumval = sum(VAL);

%% drawing topoplot of Wcsp
load('locs53.mat')
[mm,nn] = size(WCSP_BOXsingle);
topoplot(WCSP_BOXsingle{1, 1}  (:,1),locs53)
title(["sub number: ", sub_no])
colorbar

W_row = reshape(WCSP_BOXsingle',1,mm*nn);
W_row_norm =  cellfun(@(x)(x./vecnorm(x)),W_row,'UniformOutput',false);
WTEMP = NaN(length(selected_ch),2*m,mm*nn);
for q =1:mm*nn

 WTEMP(:,:,q) =  W_row_norm {1,q};
%   WTEMP(:,:,q) =  W_row {1,q};
end
Wmean = mean(WTEMP,3);
subplot(231)
topoplot(Wmean(:,1),locs53);
title(["Wcsp(:,1) sub " sub_no])
colorbar
subplot(232)
topoplot(Wmean(:,2),locs53);
title(["Wcsp(:,2) sub " sub_no])
colorbar
subplot(233)
topoplot(Wmean(:,3),locs53);
title(["Wcsp(:,3) sub " sub_no])
colorbar
subplot(234)
topoplot(Wmean(:,4),locs53);
title(["Wcsp(:,4) sub " sub_no])
colorbar
subplot(235)
topoplot(Wmean(:,5),locs53);
title(["Wcsp(:,5) sub " sub_no])
colorbar
subplot(236)
topoplot(Wmean(:,6),locs53);
title(["Wcsp(:,6) sub " sub_no])
colorbar

asdasdasdasd
%%
% figure,
% plot(PERF1')
% figure,
% pwelch(COT_str.EEG.EEG_FB{1, 5}(:,7),[],[],[],Fs);figure(gcf)

% plot off-targets
% for i=1:length(EEG_class_off)
%     figure,
%     subplot(121)
%     plot(pos_off{i});box on; grid minor;
%     subplot(122)
%     scatter(pos_off{i}(:,1),pos_off{i}(:,2));box on; grid minor;
%     title(strcat("off-trial = ",num2str(i)));
% end
%% Score Analysis
%multiband
% [FEAT1, SCORE1, perf1,pred_off_1] = fbcsp_svm_analysis(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs, 3,FFF,FSMETHOD);
%single band
[FEAT1, SCORE1, perf1,pred_off_1,W] = csp_svm_analysis_single(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs, m);

score11 = SCORE1.score1;
score12 = SCORE1.score2;
score_off_1 = SCORE1.score_off;
% [~, idx] = sort(score11(:,1),'descend');
% score11 = score11(idx,:);
% thr_one_1 = score11(1,:);
% thr_one_2 = score11(end,:);
% 
% [~, idx] = sort(score12(:,1),'descend');
% score12 = score12(idx,:);
% thr_one_1 = score12(1,:);
% thr_one_2 = score12(end,:);

figure,
subplot(211)
scatter(score11(:,1),score11(:,2))
hold on
scatter(score12(:,1),score12(:,2))
scatter(score_off_1(:,1),score_off_1(:,2),'*g')
grid on;grid minor;box on;
title(strcat('subject - ',sub_no))
legend("1&3", '2&4', 'off targets')

figure,
boxplot(FEAT1.Ftrain1,'BoxStyle','outline', 'Colors','b','Symbol','b')
hold on
boxplot(FEAT1.Ftrain2,'BoxStyle','outline','Colors','k','Symbol','b')
box on;grid on; grid minor;
ylabel('Feature Values');
title(['subject: ', sub_no, ', six first fetures'])

% eeg avg

class1 = EEG_class1(1:minlength);
class2 = EEG_class2(1:minlength);
class_off = EEG_class_off;

Ntrials = length(class1);
trial_length = size(class1{1, 1},1); % length of each trial
N_off_trials = length(class_off);


[b, a] = butter(3,0.5/(0.5*Fs),'low');
cont1 = cell2mat(class1');
cont2 = cell2mat(class2');
cont3 = cell2mat(class_off');


temp1 = filtfilt(b,a,cont1);
temp1 = temp1*W;
class1 = mat2cell(temp1,trial_length*ones(1,Ntrials))';

temp2 = filtfilt(b,a,cont2);
temp2 = temp2*W;
class2= mat2cell(temp2,trial_length*ones(1,Ntrials))';

temp3 = filtfilt(b,a,cont3);
temp3 = temp3*W;
class3= mat2cell(temp3,trial_length*ones(1,N_off_trials))';

TRIALBOX_1 = NaN(trial_length,2*m,Ntrials) ;
TRIALBOX_2 = NaN(trial_length,2*m,Ntrials) ;

for j=1:Ntrials
    TRIALBOX_1(:,:,j) = class1{1,j};
     TRIALBOX_2(:,:,j) = class2{1,j};
end

cspsig_avg_1 = mean(TRIALBOX_1,3);
cspsig_avg_2 = mean(TRIALBOX_2,3);
asdsdadsd
%% define Binary Classes (second binary classification - positive and negative)
clear EEG_class1 EEG_class2
class_one_idx = sort([find((Target_ind_Trials==1)), find((Target_ind_Trials==2))]);
class_two_idx = sort([find((Target_ind_Trials==3)), find((Target_ind_Trials==4))]);

% pos1 = POS_Trials(1,class_one_idx);
% pos2 = POS_Trials(1,class_two_idx);
%%% from onset
EEG_class1 = EEG_Trials_Complex(1,class_one_idx);
EEG_class2 = EEG_Trials_Complex(1,class_two_idx);
minlength = min(length(EEG_class1),length(EEG_class2));
%% Score Analysis
[FEAT2, SCORE2, perf,pred_off_2] = fbcsp_svm_analysis(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs, 3,FFF,FSMETHOD);


score21 = SCORE2.score1;
score22 = SCORE2.score2;
score_off_2 = SCORE2.score_off;
subplot(212)
scatter(score21(:,1),score21(:,2))
hold on
scatter(score22(:,1),score22(:,2))
scatter(score_off_2(:,1),score_off_2(:,2),'*g')
grid on;grid minor;box on;
legend("1&2", '3&4', 'off targets')
%%
figure,
scatter3(FEAT1.Ftrain1(:,1),FEAT1.Ftrain1(:,2),FEAT1.Ftrain1(:,3))
hold on
scatter3(FEAT1.Ftrain2(:,1),FEAT1.Ftrain2(:,2),FEAT1.Ftrain2(:,3))
scatter3(FEAT1.F_off(:,1),FEAT1.F_off(:,2),FEAT1.F_off(:,3),'*g')
title("features from classsifcation one (1&3 - 2&4)")

figure,
scatter3(FEAT2.Ftrain1(:,1),FEAT2.Ftrain1(:,2),FEAT2.Ftrain1(:,3))
hold on
scatter3(FEAT2.Ftrain2(:,1),FEAT2.Ftrain2(:,2),FEAT2.Ftrain2(:,3))
scatter3(FEAT2.F_off(:,1),FEAT2.F_off(:,2),FEAT2.F_off(:,3),'*g')
title("features from classsifcation one (1&2 - 3&4)")


%% Fisher score
Xtr1 =FEAT1.Ftrain1;
Xtr2 =FEAT1.Ftrain2;

fscore1 = fisher_score(Xtr1,Xtr2,0);


%% Ardi appraoch
% D1 = EEG_class1(1:minlength);
% D2 = EEG_class2(1:minlength);
% 
% D1 = cellfun(@(x)(var(x)),D1,'UniformOutput',false);
% D1 = cell2mat(D1');
% D2 = cellfun(@(x)(var(x)),D2,'UniformOutput',false);
% D2 = cell2mat(D2');
% Dtot = [D1;D2];
% Dlabel = [zeros(size(D1,1),1);ones(size(D2,1),1) ];
% 
% Mdl = fitcdiscr(Dtot,Dlabel);
% 
% SWithin=Mdl.Sigma;
% SBetween=Mdl.BetweenSigma;
% [V,D] = eig(SBetween,SWithin);
% [~,idx2] = sort(diag(D),'descend');
% V = V(:,idx2);
% WFisher= [V(:,1:m),V(:,end-m+1:end)];
% 
% Z1 = D1*WFisher;
% Z2 = D2*WFisher;
% [fscore] = fisher_score(Z1,Z2,1);
% %%
% D1 = FEAT1.Ftrain1;
% D2 = FEAT1.Ftrain2;
% D3 = FEAT1.F_off;
% Dtot = [D1;D2];
% Dlabel = [zeros(size(D1,1),1);ones(size(D2,1),1) ];
% Mdl = fitcdiscr(Dtot,Dlabel);
% 
% SWithin=Mdl.Sigma;
% SBetween=Mdl.BetweenSigma;
% [V,D] = eig(SBetween,SWithin);
% [~,idx2] = sort(diag(D),'descend');
% V = V(:,idx2);
% WFisher= [V(:,1:m),V(:,end-m+1:end)];
% 
% Z1 = D1*WFisher;
% Z2 = D2*WFisher;
% Z3 = D3*WFisher;
% [fscore] = fisher_score(Z1,Z2,1);

%%
Xtr1 = FEAT1.Ftrain1;
Xtr2 = FEAT1.Ftrain2;
D3 = FEAT1.F_off;

[Z1,Z2,WFisher,fscore] = fisher_disc(Xtr1,Xtr2,3);
Z3 = D3*WFisher;

Zt = [Z1;Z2];
GroupTR=[zeros(1,size(Z1,1)),ones(1,size(Z2,1))]';
SVMStruct = fitcsvm(Zt(:,[1 2 3]),GroupTR,'Standardize',true,...
    'KernelFunction','rbf','KernelScale','auto'); %'polynomial'
[pred, score]= predict(SVMStruct,Zt(:,[1 2 3]));
[pred_off, score_off]= predict(SVMStruct,Z3(:,[1 2 3]));

figure,
scatter3(Z1(:,1),Z1(:,2),Z1(:,3))
hold on
scatter3(Z2(:,1),Z2(:,2),Z2(:,3))
scatter3(Z3(:,1),Z3(:,2),Z3(:,3),'g*')
title("features from classsifcation one (1&3 - 2&4)-CSP-Fisher")

%
% temp1 = cellfun(@(x)x*WFisher,D1,'UniformOutput',false);
% temp1 = cellfun(@(x)log10(var(x)),temp1,'UniformOutput',false);
% ZZ1 = cell2mat(temp1');
% 
% temp1 = cellfun(@(x)x*WFisher,D2,'UniformOutput',false);
% temp1 = cellfun(@(x)log10(var(x)),temp1,'UniformOutput',false);
% ZZ2 = cell2mat(temp1');
