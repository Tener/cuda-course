// constant variables in device space
__constant__ float arb_poly_const_coeff[3*(18+1)];
__constant__ float arb_poly_const_coeff_der[3*(18+1)];
__constant__ int arb_poly_const_size[3];

float arb_poly_host_coeff[3*(18+1)];
float arb_poly_host_coeff_der[3*(18+1)];
int arb_poly_host_size[3];
