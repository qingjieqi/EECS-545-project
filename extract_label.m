function [data_r, label_r] = extract_label(data,label,num,random_flag)
% This function randomly extracts data to feed in the algorithm as labeled
% data and treat others as unlabeled.
% Input:
%   data: input data
%   label: input label
%   num: required number of labeled data
%   random_flag: if we want the labeled data to be randomly distributed
% Output:
%   data_r: random data
%   label_r: corresponding label

rng(0);
digit = 0:1:9;

%% Generate the sample size of each digit
if random_flag
    digit_n = rand(10,1);
    digit_n = digit_n./sum(digit_n);
    digit_n(1:9) = floor(digit_n(1:9)*num);
    digit_n(10) = num-sum(digit_n(1:9));
else
    digit_n = ones(10,1).*num/10;
end
%%
idx_r = zeros(num,1);
idx = 1;
for i = 1:10
    current_digit_idx = find(label==i-1);
    sample = datasample(current_digit_idx,digit_n(i),'Replace',false);
    idx_r(idx:idx+digit_n(i)-1) = sample;
    idx = idx+digit_n(i);
end

data_r = data(idx_r,:);
label_r = label(idx_r);
end