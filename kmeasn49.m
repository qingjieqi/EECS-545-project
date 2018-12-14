load emnist-digits.mat;
rng(4);
train_data = dataset.train.images; 
train_label = dataset.train.labels;
digit1 = 4;
digit2 = 9;
j = 0;
x = 10:10:100;
y = zeros(1,10);

for total_num = 10:10:100
j = j+1;
train_x = double(train_data(train_label ==digit1 | train_label == digit2,:));
train_y = double(train_label(train_label ==digit1 | train_label == digit2,:));
train_x = train_x(1:total_num,:);
train_y = train_y(1:total_num,:);
[~,idx]=sort(train_y);
train_y = train_y(idx);
train_x = train_x(idx,:);
pos = find(train_y==digit2);
[result,C] = kmeans(train_x,2);
y(j) = min((length(find(result(1:pos(1)-1)==2))+ length(find(result(pos(1):end)==1)))/total_num, 1-((length(find(result(1:pos(1)-1)==2))+ length(find(result(pos(1):end)==1)))/total_num));
end

figure
plot(x,y,'b-o')
title('4 & 9 Digit Classification using K-means')
xlabel('label set size')
ylabel('error rate');
hold on