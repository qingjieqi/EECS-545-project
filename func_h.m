function H = func_h(p)
    H = -1*sum(p.*log(p) + (1-p).*log(1-p));
end