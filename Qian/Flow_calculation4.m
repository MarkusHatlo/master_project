function [Q,P,Re] = Flow_calculation4(U1,U2,U3,ER,Ph2, T)

% Inputs: velocity, H2 power fraction and equivalence of main flow and pilot flow
% Output: H2, CH4 and air flow rate of  in main flow and pilot flow, Reynold number
% -------------------------------------------------------------------------------------------------------------

% Exit area
Amain = pi.*0.25 .*( (69.6e-3)^2 - (60e-3)^2 );
Rin_hole = ( (69.6e-3)^2 - (60e-3)^2 ) / ((1.8e-3)^2 * 80);  % Ratio of the velocity of main flow inlet and average velocity at holes
Apilot  = pi.*0.25 .*( (19e-3)^2 - (13e-3)^2 );

% Gas density... kg/m^3
rhoair = 1.184;%1.225;%  1.225的取值好像是15℃的取值，25℃的取值为1.184
rhoh2  = 0.08988;
rhoch4 = 0.656;
vair = 15.6e-6;%14.7e-6; % 空气运动粘度    14.7E-6为15℃取值，若按25℃取，则为15.6E-6

% Lower heating values:
LHh2   = 33.3 *60^2 ;  % kj/kg
LHch4 = 13.9 *60^2 ;  % kj/kg

%Calculate flow rates based on chemical kinetics + the volume fraction relationship.

% Pilot flame ------------------------------------------------------------------
VH2Infuel_p = Ph2(1)/(Ph2(1)+(1-Ph2(1))*rhoh2*LHh2/rhoch4/LHch4);
VairTofuel_p = (0.5*VH2Infuel_p+2*(1-VH2Infuel_p))/0.21/ER(1);
Qair_p = U1/T(1)*298.15*Apilot*60000*(VairTofuel_p/(VairTofuel_p+1));
Qch4_p = Qair_p/VairTofuel_p*(1-VH2Infuel_p);
Qh2_p = Qair_p/VairTofuel_p*VH2Infuel_p;
P(1) = Qch4_p/60000*rhoch4*LHch4+Qh2_p/60000*rhoh2*LHh2;

Q(1,:) = [ Qair_p, Qh2_p, Qch4_p ;];

Vair_p = Qair_p/(Qair_p + Qch4_p + Qh2_p);
Re_p = U1*(19-13)/1000/vair;

% Main flow-----------------------------------------------------------------
VH2Infuel_m = Ph2(2)/(Ph2(2)+(1-Ph2(2))*rhoh2*LHh2/rhoch4/LHch4);
VairTofuel_m = (0.5*VH2Infuel_m+2*(1-VH2Infuel_m))/0.21/ER(2);
Qair_m = U2/T(2)*298.15*Amain*60000*(VairTofuel_m/(VairTofuel_m+1));
Qch4_m = Qair_m/VairTofuel_m*(1-VH2Infuel_m);
Qh2_m = Qair_m/VairTofuel_m*VH2Infuel_m;
P(2) = Qch4_m/60000*rhoch4*LHch4+Qh2_m/60000*rhoh2*LHh2;

Q(2,:) = [ Qair_m, Qh2_m, Qch4_m ;];

Re_m = U2*(69.6-60)/1000/vair;
Re_main_hole = U3*1.8/1000/vair;

Re = [Re_p,Re_m,Re_main_hole];
end

