function NavierStokes2DLidDrivenCavity
%clear all; 
format compact; format short; clc; clf; close all
% Numerical solution of the 2D incompressible Navier-Stokes on a
% Square Domain [0,1]x[0,1] using a Pseudospectral legendre method
% and Crank-Nicolson timestepping. 
%
% By Allan P. Engsig-Karup.

LoadICON = 1; % Load Initial Condition (IC) (based on Reynolds Number)

Nx  = 35;
Ny  = 35;
Re  = 400;
tol = 1e-10; % error for reaching steady state solution
Lx  = 1;
Ly  = 1;
CFL = 4*0.5;
Px  = Nx-2;
Py  = Ny-2;

%% Reference grid
[rx]    = JacobiGL(0,0,Nx-1); %rx = 0.5*(rx+1)*Lx;
[ry]    = JacobiGL(0,0,Ny-1); %ry = 0.5*(ry+1)*Ly;
[RX,RY] = meshgrid(rx,ry);
Ntol    = min(diff(rx))/2;
iWest   = find(abs(RX+1)<=Ntol);
iEast   = find(abs(RX-1)<=Ntol);
iSouth  = find(abs(RY+1)<=Ntol);
iNorth  = find(abs(RY-1)<=Ntol);
iAll    = [iWest(:); iEast(:); iNorth(:); iSouth(:)];
iInterior = setdiff(1:length(RX(:)),iAll);

Vx   = Vandermonde1D(Px+1,rx);
Vy   = Vandermonde1D(Py+1,ry);
Vrx  = GradVandermonde1D(Px+1,rx);
Vry  = GradVandermonde1D(Py+1,ry);

%% Interpolate 2D solution to new grid
% Physical grid
xP  = 0.5*( rx(2:end-1) + 1 ) * Lx;
yP  = 0.5*( ry(2:end-1) + 1 ) * Ly;
[XP,YP] = meshgrid(xP,yP);

% Reference grid to construct operators
rxP  = rx(2:end-1)/rx(end-1);
ryP  = ry(2:end-1)/ry(end-1);
VxP  = Vandermonde1D(Px-1,rxP);
VyP  = Vandermonde1D(Py-1,ryP);
VrxP = GradVandermonde1D(Px-1,rxP);
VryP = GradVandermonde1D(Py-1,ryP);

% 1D operators
Dx   = Vrx  / Vx; 
Dy   = Vry  / Vy; 
DxP  = VrxP / VxP; 
DyP  = VryP / VyP; 
Ix   = eye(Nx);
Iy   = eye(Ny);
IxP  = eye(Nx-2);
IyP  = eye(Ny-2);

rx2 = rx/rx(end-1);%linspace(-1.2,1.2,40);
ry2 = ry/ry(end-1);%linspace(-2,2,30);
if 0
    u = sin(pi*XP).*sin(pi*YP);
    % Reference grid which is called to define where interpolation is to be
    % done
    % Do the interpolation via operations in reference domain
    [X3,Y3,u3] = Interpolate2DGrid(XP,YP,u,rx2,ry2,Px,Py,VxP,VyP);
    mesh(X3,Y3,u3)
    return
end

%% Physical grid
x     = 0.5*(rx+1)*Lx;
y     = 0.5*(ry+1)*Ly;
xP    = x(2:end-1);
yP    = y(2:end-1);
LxP   = xP(end)-xP(1);
LyP   = yP(end)-yP(1);
[X,Y] = meshgrid(x,y);

% 1D operators
Dx  = Dx  * 2 / Lx  ;
Dy  = Dy  * 2 / Ly  ;
DxP = DxP * 2 / LxP ;
DyP = DyP * 2 / LyP ;
Ix  = eye(Nx)       ;
Iy  = eye(Ny)       ;
IxP = eye(Nx-2)     ;
IyP = eye(Ny-2)     ;

% 2D operators
DX  = kron(Dx,Iy);
DY  = kron(Ix,Dy);
DXP = kron(DxP,IyP);
DYP = kron(IxP,DyP);

% Initialize variables
filename = sprintf('LidDriven2D%d.mat',Re);
if LoadICON
    if exist(filename,'file')
        load(filename)
        disp([filename ' loaded.'])
    else
        U = X*0; V = X*0; P = XP*0;
    end
%    load LidDriven2D
else
    U = X*0; V = X*0; P = XP*0;
end
b2 = 5; % beta^2 from artificial compressibility term

ux = X*0; vy = X*0;

XX = X(iNorth)-0.5;
uWall = (16*(XX-0.5).^2.*(XX+0.5));
uWall = uWall / max(uWall) * 0 +1;
U(iNorth) = uWall*0+1;
%plot(XX,uWall,'k')
%return

PX = P*0; PY = P*0;
PX1 = P*0; PY1 = P*0;
PX2 = P*0; PY2 = P*0;
PX3 = P*0; PY3 = P*0;

switch Re
    case 100
        yRe = [ 0.0000 0.0547 0.0625 0.0703 0.1016 0.1719 0.2813 0.4531 0.500 ...
            0.6172 0.7344 0.8516 0.9531 0.9609 0.9688 0.9766 1.0000];
        uRe = [ 0.0000 -0.03717 -0.04192 -0.04775 -0.06434 -0.10150 -0.15662 -0.21090 ...
            -0.20581 -0.13641 0.00332 0.23151 0.68717 0.73722 0.78871 0.84123 1.00000];
    case 400
        yRe = [ 0.0000 0.0547 0.0625 0.0703 0.1016 0.1719 0.2813 0.4531 0.500 ...
            0.6172 0.7344 0.8516 0.9531 0.9609 0.9688 0.9766 1.0000];
        uRe = [ 0.00000 -0.08186 -0.09266 -0.10338 -0.14612 -0.24299 -0.32726 ...
            -0.17119 -0.11477 0.02135 0.16256 0.29093 0.55892 0.61756 0.68439 0.75837 1.0000];
    case 1000
        yRe = [ 0.0000 0.0547 0.0625 0.0703 0.1016 0.1719 0.2813 0.4531 0.500 ...
            0.6172 0.7344 0.8516 0.9531 0.9609 0.9688 0.9766 1.0000];
        uRe = [ 0.00000 -0.18109 -0.20196 -0.22220 -0.29730 -0.38289 -0.27805 -0.10648 ...
            -0.06080 0.05702 0.18719 0.33304 0.46604 0.51117 0.57492 0.65928 1.00000];
    case 5000
        yRe = [ 0.0000 0.0547 0.0625 0.0703 0.1016 0.1719 0.2813 0.4531 0.500 ...
            0.6172 0.7344 0.8516 0.9531 0.9609 0.9688 0.9766 1.0000];
        uRe = [ 0.0000 0.42447 0.43329 0.43648 0.42951 0.35368 0.28066 0.27280 0.00945 ...
            -0.30018 -0.36214 -0.41442 -0.52876 -0.55408 -0.55069 -0.49774 0.00000 ];
end
dt = 0.0005;

dx = min(diff(x)); dy = min(diff(y));
iter = 0;
figure
while iter < 100000
    iter = iter + 1;

    umax = max(U(:)); vmax = max(V(:));
    lambdaX = (abs(umax) + sqrt(umax^2 + b2))/dx + 1/(Re*dx^2);
    lambdaY = (abs(vmax) + sqrt(vmax^2 + b2))/dy + 1/(Re*dy^2);
    dt = CFL / (lambdaX + lambdaY);

    PX(:) = DXP*P(:);
    PY(:) = DYP*P(:);
    [X3,Y3,tPX] = Interpolate2DGrid(XP,YP,PX,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [X3,Y3,tPY] = Interpolate2DGrid(XP,YP,PY,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [RHSp,RHSu,RHSv] = ComputeResiduals(P,tPX,tPY,U,V,DX,DY,Re,b2);
    RHSp = RHSp(iInterior);
    RHSu([iAll]) = 0;
    RHSv([iAll]) = 0;
    P1 = P(:) + (1/4)*dt*RHSp;
    U1 = U(:) + (1/4)*dt*RHSu;
    V1 = V(:) + (1/4)*dt*RHSv;
    U1(iNorth) = uWall;

    PX1(:) = DXP*P1(:);
    PY1(:) = DYP*P1(:);
    [X3,Y3,tPX1] = Interpolate2DGrid(XP,YP,PX1,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [X3,Y3,tPY1] = Interpolate2DGrid(XP,YP,PY1,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [RHSp,RHSu,RHSv] = ComputeResiduals(P1,tPX1,tPY1,U1,V1,DX,DY,Re,b2);
    RHSp = RHSp(iInterior);
    RHSu([iAll]) = 0;
    RHSv([iAll]) = 0;
    P2 = P(:) + (1/3)*dt*RHSp;
    U2 = U(:) + (1/3)*dt*RHSu;
    V2 = V(:) + (1/3)*dt*RHSv;
    U2(iNorth) = uWall;

    PX2(:) = DXP*P2(:);
    PY2(:) = DYP*P2(:);
    [X3,Y3,tPX2] = Interpolate2DGrid(XP,YP,PX2,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [X3,Y3,tPY2] = Interpolate2DGrid(XP,YP,PY2,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [RHSp,RHSu,RHSv] = ComputeResiduals(P2,tPX2,tPY2,U2,V2,DX,DY,Re,b2);
    RHSp = RHSp(iInterior);
    RHSu([iAll]) = 0;
    RHSv([iAll]) = 0;
    P3 = P(:) + (1/2)*dt*RHSp;
    U3 = U(:) + (1/2)*dt*RHSu;
    V3 = V(:) + (1/2)*dt*RHSv;
    U3(iNorth) = uWall;

    PX3(:) = DXP*P3(:);
    PY3(:) = DYP*P3(:);
    [X3,Y3,tPX3] = Interpolate2DGrid(XP,YP,PX3,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [X3,Y3,tPY3] = Interpolate2DGrid(XP,YP,PY3,rx2,ry2,Px-1,Py-1,VxP,VyP);
    [RHSp,RHSu,RHSv] = ComputeResiduals(P3,tPX3,tPY3,U3,V3,DX,DY,Re,b2);
    RHSp = RHSp(iInterior);
    RHSu([iAll]) = 0;
    RHSv([iAll]) = 0;
    P(:) = P(:) + dt*RHSp;
    U(:) = U(:) + dt*RHSu;
    V(:) = V(:) + dt*RHSv;
    U(iNorth) = uWall;
    
    if mod(iter,20)==0
        disp(iter)
        disp(dt)
        clf
        subplot(2,3,1)
        runix = linspace(-1,1,51);
        runiy = linspace(-1,1,51);        
    [XXX,YYY,UUU] = Interpolate2DGrid(X,Y,U,runix,runiy,Px+1,Py+1,Vx,Vy);
    [XXX,YYY,VVV] = Interpolate2DGrid(X,Y,V,runix,runiy,Px+1,Py+1,Vx,Vy);        
        streamslice(XXX,YYY,UUU,VVV,2.5)
        set(gca,'ylim',[0,Ly])
        set(gca,'xlim',[0,Lx])
        xlabel('x')
        ylabel('y')
        title('Streamlines inside driven cavity (steady state)')
        subplot(2,3,2)
        mesh(X,Y,U)
        title('U')
        subplot(2,3,3)
        mesh(X,Y,V)
        title('V')
        subplot(2,3,5)
        title(sprintf('u profile, x=0.5, Re=%d',Re))
        plot(uRe,yRe,'k.',UUU(:,26),YYY(:,26),'b','markersize',20)
        drawnow
%        pause
    end
    
    % backup
    if mod(iter,1000)==0
%        save(filename,'U','V','P')
%        disp([filename '.mat stored.'])
    end
end

return

function [RX3,RY3,u3] = Interpolate2DGrid(RX,RY,u,rx2,ry2,Px,Py,Vx,Vy)
%% Interpolate 2D solution to new grid
%  Px, Py: order of polynomial on current grid
%
%u = sin(pi*RX).*sin(pi*RY);
%subplot(1,3,1)
%mesh(RX,RY,u)
% first x-grid
coef = Vx \ u';
%rx2 = linspace(-1,1,40);
%ry2 = linspace(-1,1,30);
Vx2   = Vandermonde1D(Px,rx2);
Vy2   = Vandermonde1D(Py,ry2);
u2  = (Vx2 * coef)';
RX2 = (Vx2 * (Vx \ RX'))';
RY2 = (Vx2 * (Vx \ RY'))';
%subplot(1,3,2)
%mesh(RX2,RY2,u2)
% then y-grid
coef = Vy \ u2;
u3  = (Vy2 * coef);
RX3 = (Vy2 * (Vy \ RX2));
RY3 = (Vy2 * (Vy \ RY2));
%subplot(1,3,3)
%mesh(RX3,RY3,u3)
return

function [RHSp,RHSu,RHSv] = ComputeResiduals(P,px,py,U,V,DX,DY,Re,b2)

    ux  = DX*U(:);
    uxx = DX*ux(:);

    uy  = DY*U(:);
    uyy = DY*uy(:);
    
    vx  = DX*V(:);
    vxx = DX*vx(:);
    
    vy  = DY*V(:);
    vyy = DY*vy(:);
    
%    px  = DX*P(:);
%    py  = DY*P(:);
    
    RHSp = -b2*( ux + vy );
    RHSu = -(U(:).*ux + V(:).*uy ) - px(:) + ( uxx + uyy )/Re;
    RHSv = -(U(:).*vx + V(:).*vy ) - py(:) + ( vxx + vyy )/Re;

return
