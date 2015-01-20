
#define min(a, b) ((a) < (b) ? (a) : (b))

/* just a dummy calculation to boost gpu, calulation has no meanings */
__kernel void dummy_calc (
    __global const float * restrict A,
    __global const float * restrict B,
    __global float * restrict C,
	int width,
	int height,
	int lookaround
)
{
	int idx = get_global_id(0);
	int iy = idx / width;
	int ix = idx - iy * width;

	float r_coef = 1.0f / sqrt((float)(lookaround * lookaround * 2));
	
	lookaround = min(lookaround, ix);
	lookaround = min(lookaround, iy);
	lookaround = min(lookaround, width - ix - 1);
	lookaround = min(lookaround, height - iy - 1);

	float new_value = 0.0f;
	float weight_sum = 0.0f;
	for (int jy = - lookaround; jy <= lookaround; jy++) {
		for (int jx = - lookaround; jx <= lookaround; jx++) {
			int jdx = ((iy + jy) * width) + (ix + jx);
			float r = sqrt((float)(jy * jy + iy * iy)) * r_coef;
			float weight = (1 - r) * (1 - r) * (1 + r) * (1 + r);
			new_value += weight * (A[jdx] - B[jdx]);
			weight_sum += weight;
		}
	}
	C[idx] = new_value / weight_sum + A[idx];
}
