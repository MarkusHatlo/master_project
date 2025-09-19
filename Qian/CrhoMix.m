function [c,rho] = CrhoMix( y )

y = y./(sum(y));

Mw = [28.9647, 16.04, 2.016];  %g/mole

t = 273 + 25; %k
r0 = 8.314; %j/k mole
p = 1.012E+05;

%cp = [1.005 , 20.4 , 2.22 ]; %j/gk
cp = [1.005  , 2.226 , 14.31 ];

c = sqrt( r0 *t * ( sum(  y.*cp.*Mw ) ./ ( 1e-3* sum(y.*Mw) .* sum(y.*(cp.*Mw-r0) ) ) ) );

rho =  1E-03.*( sum(  y.*Mw ) * p) ./( r0 .*t);
end

%plot(Y(:,1),Y(:,2),'x');