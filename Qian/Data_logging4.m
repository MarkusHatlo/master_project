 clear; clc; clearvars
%close all
format long
%% Working condition parameters
Amain = pi.*0.25 .*( (69.6e-3)^2 - (60e-3)^2 );              % annular area at main flow inlet entrance
Rin_hole = ( (69.6e-3)^2 - (60e-3)^2 ) / ((1.8e-3)^2 * 80);  % Ratio of the velocity of main flow inlet and average velocity at holes

%% Input parameters: Velocity, Equivalence ratio, H2 power fraction
setup.flow.U1     =  40;                                        % Pilot flow inlet velocity (m/s)
setup.flow.ER     =  [0.81 0.0001];                        % Equivalence ratio of pilot flow and main flow(-)
setup.flow.U2     =  0;                                          % Main flow inlet velocity
setup.flow.U3     =  setup.flow.U2*Rin_hole;     % Main flow velocity at holes (m/s)
setup.flow.Ph2    =  [0  0];                                    % H2 power fraction of pilot flow and main flow(-)
setup.flow.T   =  [298.15 298.15];                        % Temperature at pilot flow inlet and main flow inlet(K)

%%%%%%%%%%%%%%%%%%%% Change !!! %%%%%%%%%%%%%%%%%%%               
setup.DAQ.t_samp = 20; %
LBO =0;
HS = 0;
sweep_number = 0;   % IF WE ARE SWEEPING
Mean_im = 1;
Long_videos = 0;
%%%%%%%%%%%%%%%%%%%% Change !!! %%%%%%%%%%%%%%%%%%%

% Calculate input parameters for the experiment -----------------------------------------------------
[Q,P,Re] = Flow_calculation4(setup.flow.U1,setup.flow.U2,setup.flow.U3,setup.flow.ER,setup.flow.Ph2,setup.flow.T);

[setup.flow.c(1),setup.flow.rho(1)] = CrhoMix( Q(1,:)); % calculate sound speed and density(kg/m^3) of the mixed pilot flow

clc;
fprintf(['Air flowrate of pilot flow:        ' num2str(round(100*Q(1,1))./100 ) ' SLPM \n'])
fprintf(['H2 flowrate of pilot flow:         ' num2str(round(100*Q(1,2))./100 ) ' SLPM \n'])
fprintf(['CH4 flowrate of pilot flow:        ' num2str(round(100*Q(1,3))./100 ) ' SLPM \n'])
fprintf('--------------------------------------------------------------------\n')
fprintf(['Thermal power of pilot flow:       ' num2str(P(1)) ' kW\n'])
fprintf(['Re of pilot flow:                  ' num2str(Re(1)) , '\n'])

setup.flow.Q = Q;          % Flowrate of different gas components
setup.flow.P = P;          % Thermal power (kW)
setup.flow.Re = Re;        % Reynold number (-)

%% Filename and folder

[setup,d1] = setup_func(setup) ;

% Saving parameters                                      
root_dir  = 'D:\Qian\202508Experiment_data_logging\05_09_D_120mm_90mm_Mean_image\';
% case_dir  = 'Trail';

% File name that is going to be saved
filename = ['Up_' num2str(setup.flow.U1) '_ERp_' num2str(setup.flow.ER(1)) '_PH2p_' num2str(setup.flow.Ph2(1)) ''];

if (LBO)
    filename = append('LBO_',filename);
    if isnan(sweep_number)
    else
        filename = ['LBO'];
        filename = append(filename, ['_Sweep_', num2str(sweep_number)]);
    end
end

if (HS)
    filename = append('HS_',filename);
    if isnan(sweep_number)
    else
        filename = append(filename, ['_Sweep_', num2str(sweep_number)]);
    end
end

if (Mean_im)
    filename = append('Mean_im_',filename);
end

if (Long_videos)
    filename = append('Long_videos_',filename);
end
log = 1;

root_dir = [root_dir];%, case_dir];

if(~isfolder(root_dir))
    mkdir(root_dir)
end

time_str = ['_', num2str(hour(datetime)), '_', num2str(minute(datetime)), '_', sprintf('%.0f',second(datetime))];% hour-min

filename = append(filename, time_str, '.mat');

%% Read off data
% input data to d1 
% d1: pressure and PMT signal

start( d1 ,'duration',seconds(setup.DAQ.t_samp));

[scanData1, time1] = read( d1 ,seconds(setup.DAQ.t_samp));

stop(d1);flush(d1);

F(1).data =scanData1;
F(1).time =time1;

stop(d1);flush(d1);

data.timestamp_fast = F(1).time;
data.time_fast = time2num( F(1).data.Time );

% Data of different dimension of F(1) is determined by the 'setup_func.m''s channel defination
data.PMT_OH_1 = F(1).data{:,1};
data.Cam_trig = F(1).data{:,5};% Pref and Cam trigger
data.P1   = F(1).data{:,2}./(setup.mics.s(1)*setup.mics.gain(1));  % Voltage to Pascal
data.P2   = F(1).data{:,3}./(setup.mics.s(2)*setup.mics.gain(2));
data.P3   = F(1).data{:,4}./(setup.mics.s(3)*setup.mics.gain(3));
data.Pref   = F(1).data{:,6};
% data.P4   = F(1).data{:,4}./(setup.mics.s(4)*setup.mics.gain(4)); 
% data.P5   = F(1).data{:,8}./(setup.mics.s(5)*setup.mics.gain(5));
% data.P6   = F(1).data{:,3}./(setup.mics.s(6)*setup.mics.gain(6));
%%
if log == 1
    out_dir = root_dir;
    if ~exist(out_dir, 'dir')
       mkdir(out_dir)
    end
    out = [out_dir filename];
    save(out, 'setup', 'data')
end

fprintf('\n\n--------------------------------------------------------------------\n')
fprintf('Finished taking data for case: Up = %.1f, Um = %.1f, ERp = %.2f,\n ERm = %.2f, Pp = %.2f, Pm = %.2f, Ph2p = %.1f, Ph2m = %.2f\n', setup.flow.U1, setup.flow.U2,setup.flow.ER(1),setup.flow.ER(2),setup.flow.P(1),setup.flow.P(2),setup.flow.Ph2(1),setup.flow.Ph2(2))
fprintf('--------------------------------------------------------------------\n')

%%
N = 2^12;

[PSD1,w] = PSD_Cfunc_amp( data.P1 ,data.PMT_OH_1, N , 0.5*N, 4*N, 51200 );

figure
subplot(3,1,1);plot(data.PMT_OH_1);title('PMT')
subplot(3,1,2);plot(data.P1);title('P1');
subplot(3,1,3);plot(data.Cam_trig);title('Camera trigger')


