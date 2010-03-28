import indicator 	as ind;
import r_mor;
from scipy.misc	    import imshow;

reload(ind);
reload(r_mor);

A = ind.load_img("shape2.png");
imshow(A);

B = ind.load_img("shape3.png");
imshow(B);

imshow(r_mor.cpad(B, (256,256)));

A_star_B = r_mor.conv(A,B);
imshow(A_star_B);

A_cor_B = r_mor.hg_conv(A,B);
imshow(A_cor_B);

A_sum_B = r_mor.mink_sum(A,B);
imshow(A_sum_B);

A_diff_B = r_mor.mink_diff(A,B);
imshow(A_diff_B);

A_hg_sum_B = r_mor.hg_mink_sum(A,B);
imshow(A_hg_sum_B);

A_hg_ldiff_B = r_mor.hg_mink_ldiff(A,B);
imshow(A_hg_ldiff_B);

A_hg_rdiff_B = r_mor.hg_mink_rdiff(A,B);
imshow(A_hg_rdiff_B);

