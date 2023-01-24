close all
clear all
clc
%
% Results for u-velocity along Vertical Line 
% through Geometric Center of Cavity.
%
% By Allan P. Engsig-Karup.
%

% Re
y100 = [ 0.0000 0.0547 0.0625 0.0703 0.1016 0.1719 0.2813 0.4531 0.500 ...
      0.6172 0.7344 0.8516 0.9531 0.9609 0.9688 0.9766 1.0000];
u100 = [ 0.0000 -0.03717 -0.04192 -0.04775 -0.06434 -0.10150 -0.15662 -0.21090 ...
     -0.20581 -0.13641 0.00332 0.23151 0.68717 0.73722 0.78871 0.84123 1.00000];

 plot(u100,y100)
 