% load training data and testing data
load emnist-digits.mat;
train_data = dataset.train.images; %n*d
train_label = dataset.train.labels;%n*1

x = double(train_data(1,:));
x = reshape(x,28,28);
imagesc(x);
%colormap(gray);
% select k numbers out of original matrix
digitarr = [];% can directly define here
K = 10;
t = randperm(10)-1;
digitarr = t(1:K);% or use randomly chosen number

total_num = 5000;

for i = 1:K
    if i == 1
        xt = [double(train_data(train_label == digitarr(i),:))];
        yt = [double(train_label(train_label == digitarr(i),:))];
    else
        xt = [xt;double(train_data(train_label == digitarr(i),:))];
        yt = [yt;double(train_label(train_label == digitarr(i),:))];
    end
end
select = randperm(length(xt(:,1)));
select = select(1:total_num);
xt = xt(select,:);
yt = yt(select,:);
var = 3800;
label_num = 20*K;% define the first label_num data as label data
% It is random permutation, so we can not guarantee the labeled number of each
% digit

xt2 = xt.*xt;
xtinnp = xt*(xt');
xt2sum = sum(xt2,2);
wt2s = xt2sum+xt2sum';
W = exp(-(wt2s-2*xtinnp)/var/784);

xl = xt(1:label_num,:);
xu = xt(label_num+1:end,:);
yl = yt(1:label_num,:);
yu = yt(label_num+1:end,:);

Wll = W(1:label_num,1:label_num);
Wlu = W(1:label_num,label_num+1:end);
Wul = W(label_num+1:end,1:label_num);
Wuu = W(label_num+1:end,label_num+1:end);
for i =1:K
    if i == 1
        fl = (yl==digitarr(i));
    else
        fl = [fl (yl==digitarr(i))];
    end
end
D = diag(sum(W,1));
Duu = D(label_num+1:end,label_num+1:end);
fux = (Duu-Wuu)\Wul*fl;
fux = (fux-mean(fux))./std(fux);
[~, p] = max(fux');
idx = p';
fu = zeros(length(idx),1);
for i =1:length(idx)
    fu(i) = digitarr(idx(i));
end
errrate  = sum(fu~=yu)/length(fu);