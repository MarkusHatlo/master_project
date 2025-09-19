function [setup , d1] = setup_func(setup) 

%% DAQ setup
setup.DAQ.f_samp1 = 51200;    


%% Define microphone properties
 % Number of microphones to be used in MMM
setup.mics.n_mic  = 3;  

% Location of main microphones
setup.mics.x_pos(1) =  -0.0653;
setup.mics.x_pos(2) =  -0.0902;
setup.mics.x_pos(3) =  -0.1506;

% Location of secondary microphones
% setup.mics.x_pos(4) =  -0.143;
% setup.mics.x_pos(5) =  -0.163;
% setup.mics.x_pos(5) =  -0.208;

% sensor
setup.mics.sensor{1} =  '8354-2-72';
setup.mics.sensor{3} =  'MXX3';
setup.mics.sensor{3} =  '8354-2-79';
% setup.mics.sensor{4} =  '7484-1-122';
% setup.mics.sensor{5} =  'Old_kulite_MX3';
% setup.mics.sensor{6} =  'old_kulite';

% Sensitivity (V/pa)
setup.mics.s(1) =  400.8./(1e3*1e5);
setup.mics.s(2) =  400.8./(1e3*1e5);
setup.mics.s(3) =  400.8./(1e3*1e5);
% setup.mics.s(4) =  400.8./(1e3*1e5);% correct!!!
% setup.mics.s(5) =  400.8./(1e3*1e5)*(1/1.115);
% setup.mics.s(6) =  400.8./(1e3*1e5);

% Gain amplifier
setup.mics.gain(1) =  300;
setup.mics.gain(2) =  300;
setup.mics.gain(3) =  300;
% setup.mics.gain(4) =  100;
% setup.mics.gain(5) =  100;
% setup.mics.gain(6) =  100;

%% Define Thermocouple properties
% Location
% setup.Temp.x_pos(1) =  101;

%% DAQ setup: d1
% High speed
d1 = daq('ni');
d1.Rate = 51200;%setup.DAQ.f_samp1;

warning('off')
ch1   = addinput(d1, 'cDAQ1Mod1', 'ai0',   'voltage'); % PMT
ch2   = addinput(d1, 'cDAQ1Mod1', 'ai1',   'voltage'); % P1
ch3   = addinput(d1, 'cDAQ1Mod1', 'ai2',   'voltage'); % P2
ch4   = addinput(d1, 'cDAQ1Mod1', 'ai3',   'voltage'); % P3

ch5   = addinput(d1, 'cDAQ1Mod2', 'ai2',   'voltage'); %  Camera trigger
ch6   = addinput(d1, 'cDAQ1Mod4', 'ai3',   'voltage'); %  Pref
% ch7   = addinput(d1, 'AnnularDAQ2', 'ai2',   'voltage'); %  P3
% ch8   = addinput(d1, 'AnnularDAQ2', 'ai3',   'voltage'); %  P5

warning('on')

set(ch1,   'Coupling', 'DC')    % Force DC read
set(ch2,   'Coupling', 'DC')    % Force DC read
set(ch3,   'Coupling', 'DC')    % Force DC read
set(ch4,   'Coupling', 'DC')    % Force DC read
set(ch5,   'Coupling', 'DC')    % Force DC read
set(ch6,   'Coupling', 'DC')    % Force DC read
% set(ch7,   'Coupling', 'DC')    % Force DC read
% set(ch8,   'Coupling', 'DC')    % Force DC read


%% d2
% % Low speed.
% d2 = daq('ni');
% d2.Rate = setup.DAQ.f_samp2;
% addinput(d2, 'cDAQ3Mod4', 'ai0', 'Thermocouple');d2.Channels(1).ThermocoupleType = 'K'; % thermocouple
% % addinput(d2, 'cDAQ3Mod4', 'ai1', 'Thermocouple');d2.Channels(2).ThermocoupleType = 'K'; %  
% % addinput(d2, 'cDAQ3Mod4', 'ai2', 'Thermocouple');d2.Channels(3).ThermocoupleType = 'K'; %  
% % addinput(d2, 'cDAQ3Mod4', 'ai3', 'Thermocouple');d2.Channels(4).ThermocoupleType = 'K'; %  

fprintf('ADDED INPUTS, wait 5 seconds for syncing...\n')

end