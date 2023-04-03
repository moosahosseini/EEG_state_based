% this script generates classification results for each subject (Binary accuracy)
% reporting accuracy based on 10 repetition of K-fold
% classifying class 1 (movement along x-axis) and class 2 (movement along y-axis)
% Author: Seyyed Moosa Hosseini
% paper: State-Based Decoding of Continuous Hand Movements using EEG
% Signals (IEEE access)

%%

clc
clear
close all


rng("default")
saveresults = 'nnn'; % 'y' , 'n'
svmkernel = 'rbf';
remove_IDX = [1 2 3 4 5 33 34 61 62 63];
selected_ch = 1:63;
selected_ch(remove_IDX)=[];


sub_no = '6';
string_str = '\clean_data\COT_str_run_';
savedir = 'D:\Thesis\NBML_DATA\cot_analysis2\final_results\';
savename = strcat(savedir,'sub_',sub_no,'_cls_bin_default.mat');
alignment = 'go_cue'; %= 'go_cue'
run_numbers = [1 2 3 4];
sp_filter = 'off'; 
remove_channels ='on'; % [ 1 2 3 4 5 33 34 61 62 63]

Fs = 512;
% determine the interval of data for clasifcation
% 500 ms before go cue
T_before = floor(0.5*Fs); % time before alignement point
T_after = floor(0*Fs);  % time after alignement point


currentdir = 'D:\Spring402\NBML_DATA';

% LOCS = readlocs(locdir);
% LOCS = struct2table(LOCS);
% XLocs = LOCS.X;
% YLocs = LOCS.Y;
% ZLocs = LOCS.Z;

% % % loading ONSET sample times
% % onsetdir = strcat(currentdir,'\sub_',sub_no,'\clean_data\ONSET_IDX.mat');
% % load(onsetdir)
% % MOV_IDX = ONSET_IDX.MOV_IDX;


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
        %fprintf("just after \n")
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

m= 3;
minlength = min(length(EEG_class1),length(EEG_class2));
FFF = 10;
FSMETHOD = 'ardiMI'; %% feature selection method

% multiband
% [PERF1,~,Selected_IDX_Box ]= fbcsp_cv_svm(EEG_class1(1:minlength), EEG_class2(1:minlength),Fs,10, 3,FFF,FSMETHOD,10);
% single band

% [PERF1,WCSP_BOX] = Rcsp_cv_svm_single(EEG_class1(1:minlength), EEG_class2(1:minlength),Fs,10, m,10);
[PERF1,WCSP_BOXsingle,PL_BOX] = csp_cv_svm_single(EEG_class1(1:minlength), EEG_class2(1:minlength),Fs,10, m,10,svmkernel);
% [PERF1,~,Selected_IDX_Box ]= fbcsp_cv_svm(EEG_class1(1:minlength), EEG_class2(1:minlength),Fs,10, 3,FFF,FSMETHOD,10);

meanvector = mean(PERF1);
disp(["mean: ", num2str(mean(meanvector)*100)])
disp(["T_before  ",alignment, num2str(T_before/Fs)])
disp(["T_after  ", alignment, num2str(T_after/Fs)])
disp(["Fs: ", Fs])
disp(["sub number: " sub_no])
disp(["kernel: " svmkernel])

% cls_bin.sub_no = sub_no;
% cls_bin.Fs = Fs;
% cls_bin.PERF1 = PERF1;
% cls_bin.WCSP_BOXsingle = WCSP_BOXsingle;
% cls_bin.PL_BOX = PL_BOX;
% cls_bin.method = 'singleband [0, 0.5Hz]';
% cls_bin.kernel = svmkernel;
% cls_bin.date = sprintf('Date: %s',datestr(datetime('now')));
% if strcmp(saveresults,'y')
%    save(savename,'cls_bin') 
% end
% fprintf("end of first part \n")


%% Score Analysis
%



[FEAT1, SCORE1, perf1,pred_off_1,W] = csp_svm_analysis_single(EEG_class1(1:minlength), EEG_class2(1:minlength),EEG_class_off,Fs, m,svmkernel);

score11 = SCORE1.score1;
score12 = SCORE1.score2;
score_off_1 = SCORE1.score_off;
score_mean = mean(score_off_1);
score_std = std(score_off_1);


figure,

scatter(score11(:,1),score11(:,2))
hold on
scatter(score12(:,1),score12(:,2))
scatter(score_off_1(:,1),score_off_1(:,2),'*g')
grid on;grid minor;box on;
title(strcat('subject - ',sub_no))
legend("class1", 'class2', 'random targets')

%%
% density of class 1 and class 2 and random tragets distribution
% one dim score 
close all
figure,hold on

[f,xi]=ksdensity(score11(:,2) );         plot(xi,f,'b','LineWidth',2)
ind = dsearchn(xi',score11(:,2));    
ind = unique(ind,'stable'); scatter(xi(ind),f(ind),60,'b','filled');

[f,xi]=ksdensity(score12(:,2));          plot(xi,f,'r','LineWidth',2)
ind = dsearchn(xi',score12(:,2));    
ind = unique(ind,'stable'); scatter(xi(ind),f(ind),60,'r','filled');

[f,xi]=ksdensity(score_off_1(:,2));      plot(xi,f,':g','LineWidth',2)
ind = dsearchn(xi',score_off_1(:,2));    
ind = unique(ind,'stable'); scatter(xi(ind),f(ind),60,'g','filled');

line([max(score11(:,2)) max(score11(:,2))], [0 2],'LineWidth',2,'Color','b','LineStyle','-.')
line([min(score12(:,2)) min(score12(:,2))], [0 2],'LineWidth',2,'Color','r','LineStyle','-.')
axis tight
%%
% Boxplot of Features from class1 and class2

figure,
bh = boxplot(FEAT1.Ftrain1,'BoxStyle','outline', 'Colors','b','Symbol','b');
set(bh,'LineWidth', 1.5);
hold on
bh2 =boxplot(FEAT1.Ftrain2,'BoxStyle','outline','Colors','k','Symbol','b');
set(bh2,'LineWidth', 1.5);

box on;grid on; grid minor;
ylabel('Feature Values');
title(['subject: ', sub_no, ', six first fetures'])

disp(["boxplot of features from class 1 and class 2"])
% eeg avg
