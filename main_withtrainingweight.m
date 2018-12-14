%%
% load training data and testing data

load emnist-digits.mat;
train_data = dataset.train.images; %n*d
train_label = dataset.train.labels;%n*1

%colormap(gray);
% select 4 9 out of original matrix
digit1 = 4;
digit2 = 9;
total_num = 500;
d = 28*28;
xt = double(train_data(train_label ==digit1 | train_label == digit2,:));
yt = double(train_label(train_label ==digit1 | train_label == digit2,:));
xt = xt(1:total_num,:);
xt = xt/max(max(xt));
yt = yt(1:total_num,:);
%% HyperParameter
label_num = 20;
sigma = 10;
e = 0.1;
sigmad = ones(d,1)*sigma;
% Learning rate
lr = sigma/10^4;
ite_num = 10;
%%

xt2 = xt.*xt;
xtinnp = xt*(xt');
xt2sum = sum(xt2,2);
wt2s = xt2sum+xt2sum';
W = exp(-(wt2s-2*xtinnp)/sigma^2);


xl = xt(1:label_num,:);
xu = xt(label_num+1:end,:);
yl = yt(1:label_num,:);
yu = yt(label_num+1:end,:);
%%
Wll = W(1:label_num,1:label_num);
Wlu = W(1:label_num,label_num+1:end);
Wul = W(label_num+1:end,1:label_num);
Wuu = W(label_num+1:end,label_num+1:end);
D = diag(sum(W,1));
Duu = D(label_num+1:end,label_num+1:end);


%%
error_arr = zeros(ite_num,1);
Dcheck = zeros(ite_num+1,d);
Hcheck = zeros(ite_num,d);
Wcheck = zeros(ite_num,d);
Pcheck = zeros(ite_num,d);
Dcheck(1,:) = sigmad';
for i=1:ite_num
    
    P = D\W;
    U = 1/length(P(:,1))*ones(length(P(:,1)),length(P(1,:)));
    Ptelta = e*U + (1-e)*P;
    fl = (yl==digit2);
    fux = (Duu-Wuu)\Wul*fl;
    fu = (fux-min(fux))/(max(fux)-min(fux));
    fu = max(1e-10, min(1 - 1e-10, fu));
    H_current = func_h(fu)/(total_num-label_num)
    fudedmat = zeros(total_num-label_num,d);
    Hded = zeros(d,1);
    for k = 1:d
        xs = xt(:,k).*xt(:,k);
        Wij_ded = 2*W.*((xs + xs')-2*xt(:,k)*xt(:,k)')/(sigmad(k))^3;
        Wi = zeros(length(W(:,1)),1) + sum(W,2)';
        Wi_ded = zeros(length(W(:,1)),1) + sum(Wij_ded,1);
        Wcheck(i,k) = sum(sum(Wij_ded));
        Pij_ded = (Wij_ded - P.*Wi_ded)./Wi;% this term will become small
        Pcheck(i,k) = sum(sum(Pij_ded));
        Pteltaij_ded = (1-e)*Pij_ded;
        fu_ded = (eye(total_num-label_num)-Ptelta(label_num+1:end,label_num+1:end))\(Pteltaij_ded(label_num+1:end,label_num+1:end)*fu + Pteltaij_ded(label_num+1:end,1:label_num)*fl);
        
        Hded(k) = 1/(total_num-label_num)*sum(log((1-fu)./fu).*fu_ded);
        fudedmat(:,k) = fu_ded;
    end
    sigmad = sigmad - lr*Hded;
    Dcheck(i+1,:) = sigmad';
    sigmadt = sigmad';
    xt2 = (xt./(sigmadt + zeros(total_num,1))).*(xt./(sigmadt + zeros(total_num,1)));
    xtinnp = (xt./(sigmadt + zeros(total_num,1)))*((xt./(sigmadt + zeros(total_num,1)))');
    xt2sum = sum(xt2,2);
    wt2s = xt2sum+xt2sum';
    W = exp(-(wt2s-2*xtinnp));
    Wll = W(1:label_num,1:label_num);
    Wlu = W(1:label_num,label_num+1:end);
    Wul = W(label_num+1:end,1:label_num);
    Wuu = W(label_num+1:end,label_num+1:end);
    D = diag(sum(W,1));
    Duu = D(label_num+1:end,label_num+1:end);
    Hcheck(i,:) = Hded';
    if norm(Hded)<1e-9
        break;
    end
    fl = (yl==digit2);
    fux = (Duu-Wuu)\Wul*fl;
    fu = fux-mean(fux);
    fu = (fu>0)*(digit2-digit1)+digit1;
    error_arr(i)  = sum(fu~=yu)/length(fu);
end


fl = (yl==digit2);

fux = (Duu-Wuu)\Wul*fl;
fu = fux-mean(fux);
fu = (fu>0)*(digit2-digit1)+digit1;
errrate  = sum(fu~=yu)/length(fu);
