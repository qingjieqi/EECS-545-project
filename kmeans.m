load emnist-digits.mat;
rng(1);
train_num = 20000;
train_data = dataset.train.images; 
train_label = dataset.train.labels;
train_data = double(train_data);
train_x = train_data(1:train_num,:);
train_y = train_label(1:train_num);
[~,idx]=sort(train_y);
train_y = train_y(idx);
train_x = train_x(idx,:);
[result,C] = kmeans(train_x,10);
idx_array=zeros(1,11);
idx_array(11)=length(result);
map_array=zeros(1,10);

for i = 1:10
    pos = find(train_y==i-1);
    idx_array(i) = pos(1);
end


for i = 1:10
    map_array(i)=mode(result(idx_array(i):idx_array(i+1)-1));
end

   
  
miss_array=zeros(1,10);

for i = 1:10
    miss_array(i)=length(find(result(idx_array(i):idx_array(i+1)-1)==map_array(i)));
    miss_array(i)=1-miss_array(i)/(idx_array(i+1)-idx_array(i));
end


miss_array

for i = 1:10
 subplot(2,5,i)
 imagesc(reshape(C(i,:),[28,28]))
end
