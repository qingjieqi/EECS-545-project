% load training data and testing data
load emnist-digits.mat;
train_data = dataset.train.images; %n*d
train_label = dataset.train.labels;%n*1

x = double(train_data(1,:));
x = reshape(x,28,28);
imagesc(x);
%colormap(gray);
% select 4 9 out of original matrix
digit1 = 4;
digit2 = 8;
total_num = 4800;
xt = double(train_data(train_label ==digit1 | train_label == digit2,:));
yt = double(train_label(train_label ==digit1 | train_label == digit2,:));
xt = xt(1:total_num,:);
yt = yt(1:total_num,:);
var = 3800;

xt2 = xt.*xt;
xtinnp = xt*(xt');
xt2sum = sum(xt2,2);
wt2s = xt2sum+xt2sum';
W = exp(-(wt2s-2*xtinnp)/var/784);

label_num = 20;
xl = xt(1:label_num,:);
xu = xt(label_num+1:end,:);
yl = yt(1:label_num,:);
yu = yt(label_num+1:end,:);

Wll = W(1:label_num,1:label_num);
Wlu = W(1:label_num,label_num+1:end);
Wul = W(label_num+1:end,1:label_num);
Wuu = W(label_num+1:end,label_num+1:end);
fl = (yl==digit2);
D = diag(sum(W,1));
Duu = D(label_num+1:end,label_num+1:end);
fux = (Duu-Wuu)\Wul*fl;
fu = fux-mean(fux);
fu = (fu>0)*(digit2-digit1)+digit1;
errrate  = sum(fu~=yu)/length(fu);
