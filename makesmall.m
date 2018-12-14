%make the number of data pixel smaller than 28*28


load emnist-digits.mat;
train_data = dataset.train.images; %n*d
train_label = dataset.train.labels;%n*1
n = length(train_data(:,1));
x = double(train_data(1,:));
x = reshape(x,28,28);
x = x/max(max(x));
lambda = 0.2;
C = [0 -lambda 0; -lambda lambda*4+1 -lambda; 0 -lambda 0];
xc = conv2(x,C);
xc(xc<0) = 0;
xc(xc>1) = 1;
edge = [2 6];
xc = xc(edge(1)+1:end - edge(1), edge(2)+1:end - edge(2));

figure(1);
imagesc(x);
figure(2);
imagesc(xc);
train_data_small = zeros(n,26*18);

for i = 1 : n
    x = double(train_data(i,:));
    x = reshape(x,28,28);
    x = x/max(max(x));
    lambda = 0.2;
    C = [0 -lambda 0; -lambda lambda*4+1 -lambda; 0 -lambda 0];
    xc = conv2(x,C);
    xc(xc<0) = 0;
    xc(xc>1) = 1;
    edge = [2 6];
    xc = xc(edge(1)+1:end - edge(1), edge(2)+1:end - edge(2));
    train_data_small(i,:) = reshape(xc,1,26*18);
end

save(['training_data_small'],'train_data_small','train_label');