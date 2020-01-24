function H = sample_khatri_rao(A, factor_idx)

    kr_els = sort(1:size(A,2),'descend');

    
    H = A{kr_els(1)}(factor_idx(:,kr_els(1)),:);
    
    for i = kr_els(2:end)
        H = H.* A{i}(factor_idx(:,i),:);
    end
end