import indicator 	as ind;
from scipy.misc	    import imshow;

A = ind.load_img("shape1.png");
imshow(A);

B = ind.load_img("shape2.png");
imshow(B);

A_U_B = ind.union(A, B);
imshow(A_U_B);

A_N_B = ind.intersect(A, B);
imshow(A_N_B);

A_C = ind.complement(A);
imshow(A_C);

A_S_B = ind.subtract(A, B);
imshow(A_S_B);


