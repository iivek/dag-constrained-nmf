function [a] = gnmf_solvebynewton(c, a0)
% GNMF_SOLVEBYNEWTON		Solves C = Log(A) -  Psi(A) + 1  for A by Newton-Raphson
%
%  [] = gnmf_solvebynewton()
%
% Inputs :
%	:
%
% Outputs :
%	:
%
% Usage Example : [] = gnmf_solvebynewton();
%
%
% Note	:
% See also

% Uses :

% Change History :
% Date		Time		Prog	Note
% 29-Feb-2008	 4:19 PM	ATC	Created under MATLAB 6.5.0 (R13)

% ATC = Ali Taylan Cemgil,
% SPCL - Signal Processing and Communications Lab., University of Cambridge, Department of Engineering
% e-mail : atc27@cam.ac.uk

    if nargin<2,
        a0 = 0.1*ones(size(c));
    end;

    
    
    [M N] = size(a0);
    [Mc Nc] = size(c);
    
    if M==Mc & N == Nc, % No tie
    
        a = a0;
        cond = 1;
        
    elseif Mc==1 & Nc>1,  % Tie rows
        cond = 2;

        a = a0(1,:);
        
    elseif Mc>1 & Nc==1,  % Tie cols
        cond = 3;
        a = a0(:,1);
    
    elseif Mc==1 & Nc==1,  % Tie all 
        cond = 4;
        a = a0(1,1);
    end
        
    for i=1:10, % Should be enough
        a2 = a - (log(a) - psi(0, a) + 1 - c )./(1./a - psi(1,a));
        idx = find(a2(:)<0);
        if ~isempty(idx),
            %            idx
            a2(idx) = a(idx)/2;
        end;
        a = a2;
    end;
    
    switch cond,
        %      case 1, do nothing
      case 2, % tie rows
        a = repmat(a, [M 1]);
      case 3, % tie cols
        a = repmat(a, [1 N]);
      case 4,
        a = a*ones(M, N);
    end;