clc
clear
close all

randn('state',100)
NUM=xlsread('data_five');
x=NUM(1:end,1);
dt=1;
%% 分区间

xmax=max(x);xmin=min(x);bins=40;
binlen=(xmax-xmin)/bins;

%% f的估计值
for j=1:bins
    tx(1,j)=xmin+j*binlen-1/2*binlen;%
    num(j)=0;
    sumx(j)=0;
    for i=1:size(x,1)-1
        if x(i)>= xmin+(j-1)*binlen && x(i)< xmin+j*binlen
            num(j)=num(j)+1;
            sumx(j)=sumx(j)+(x(i+1)-x(i))/dt;             
        else
            ;
        end
    end
    tx(2,j)=(sumx(j))/num(j);
end

%% Select the basis functions to build the library
u=(tx(1,:))';
Phi=[ones(size(u,1),1),u,u.^2,u.^3,u.^4,u.^5,u.^(0.5),u.^(-0.5),1./u,log(u),log(u)./u,exp(1./u)];
Phi_name={'1','t','t.^2','t.^3','t.^4','t.^5','t.^(0.5)','t.^(-0.5)','1./t','log(t)','log(t)./t','exp(1./t)'};

%% For the identification of the drift term
     dr.thre=0.001;
     Drif =TMSBL ( Phi, (tx(2,:))', 'prune_gamma',1e-4, 'lambda',1e-1, 'learn_lambda',1, 'matrix_reg', zeros(1));
     %Drif(abs(Drif)<dr.thre)=0;

threshold=10^-10;
fprintf('Drift: f(x)=');
for i = 1:size(Drif,1)
    if abs(Drif(i))<threshold
            ;
    else
       if Drif(i)<0
           fprintf('%.4f*%s', Drif(i),Phi_name{i});
       else
           fprintf('+');
           fprintf('%.4f*%s', Drif(i),Phi_name{i});
        end
    end
end
fprintf('\n')

   %% 第一种方法求出的G值（在求出f的基础上）
  for j=1:bins
    tx2(1,j)=xmin+j*binlen-1/2*binlen;
    num(j)=0;
    sumx(j)=0;
    for i=1:size(x,1)-1
        if x(i)>= xmin+(j-1)*binlen && x(i)< xmin+j*binlen
            num(j)=num(j)+1;
            sumx(j)=sumx(j)+(x(i+1)-x(i)-sum(Drif'.*PhiMatrix(x(i)),2)*dt).^2/dt;
        else
            ;
        end
    end
    tx2(2,j)=(sumx(j))/num(j); %%欧拉方法中的G
  end

%% For the identification of the diffusion term
     di.thre=0.001;
     Diff =TMSBL (Phi,sqrt(tx2(2,:))', 'prune_gamma',1e-4, 'lambda',1e-1 , 'learn_lambda', 0, 'matrix_reg', zeros(1));
     %Diff(abs(Diff)<di.thre)=0;

fprintf('Diffusion: G(x)=',1,1);
for i = 1:size(Diff,1)
    if abs(Diff(i))<threshold
            ;
    else
       if Diff(i)<0
           fprintf('%.6f*%s', Diff(i),Phi_name{i});
       else
           fprintf('+');
           fprintf('%.6f*%s', Diff(i),Phi_name{i});
        end
    end
end
fprintf('\n')

  %%  The employed library
    
    function M=PhiMatrix(u)
          M=[ones(size(u,1),1),u,u.^2,u.^3,u.^4,u.^5,u.^(0.5),u.^(-0.5),1./u,log(u),log(u)./u,exp(1./u)];
    end

