function val = soft_thresh(in, thresh)

[nr, nc] = size(in);

for ii=1:nr
    for jj=1:nc
        if (abs(in(ii,jj)) <= thresh) 
            val(ii,jj) = 0;
        elseif (in(ii,jj) > thresh)
            val(ii,jj) = in(ii,jj) - thresh;
        else
            val(ii,jj) = in(ii,jj) + thresh;
        end
    end
end
    