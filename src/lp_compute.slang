// Slang compute shader for lamination parameter computation.
// Dispatched over M laminates, each thread computes LP for one laminate.

// Constants for the LP computation (set from Python)
struct LPParams {
    float invN;
    float N2;
    float N3;
    int N;        // number of layers
    int M;        // number of laminates in batch
};

// Buffer bindings
RWStructuredBuffer<float> resultLP    : register(u0);  // [M, 12] output
RWStructuredBuffer<float> inputAngles : register(u1);  // [M, N] input angles (radians)
RWStructuredBuffer<float> Z2_buffer   : register(u2);  // [N]
RWStructuredBuffer<float> Z3_buffer   : register(u3);  // [N]
RWStructuredBuffer<float> lp_t_buffer : register(u4);  // [12] target LP

// Helper: compute LP for a single laminate (angles at offset idx*N)
void compute_lp(int idx, int N, float invN, float N2, float N3,
                in RWStructuredBuffer<float> angles,
                in RWStructuredBuffer<float> Z2,
                in RWStructuredBuffer<float> Z3,
                out float lp[12])
{
    float cos2 = 0, sin2 = 0, cos4 = 0, sin4 = 0;
    float dotZ2_c2 = 0, dotZ2_s2 = 0, dotZ2_c4 = 0, dotZ2_s4 = 0;
    float dotZ3_c2 = 0, dotZ3_s2 = 0, dotZ3_c4 = 0, dotZ3_s4 = 0;

    int base = idx * N;
    for (int i = 0; i < N; i++) {
        float ang = angles[base + i];
        float a2 = ang * 2.0;
        float a4 = ang * 4.0;
        float c2 = cos(a2);
        float s2 = sin(a2);
        float c4 = cos(a4);
        float s4 = sin(a4);

        cos2 += c2;  sin2 += s2;  cos4 += c4;  sin4 += s4;
        float z2 = Z2[i];
        float z3 = Z3[i];
        dotZ2_c2 += z2 * c2;  dotZ2_s2 += z2 * s2;
        dotZ2_c4 += z2 * c4;  dotZ2_s4 += z2 * s4;
        dotZ3_c2 += z3 * c2;  dotZ3_s2 += z3 * s2;
        dotZ3_c4 += z3 * c4;  dotZ3_s4 += z3 * s4;
    }

    lp[0]  = cos2 * invN;
    lp[1]  = sin2 * invN;
    lp[2]  = cos4 * invN;
    lp[3]  = sin4 * invN;
    lp[4]  = dotZ2_c2 * N2;
    lp[5]  = dotZ2_s2 * N2;
    lp[6]  = dotZ2_c4 * N2;
    lp[7]  = dotZ2_s4 * N2;
    lp[8]  = dotZ3_c2 * N3;
    lp[9]  = dotZ3_s2 * N3;
    lp[10] = dotZ3_c4 * N3;
    lp[11] = dotZ3_s4 * N3;
}

// Compute gradient of 0.5 * sum((lp - lp_t)^2) w.r.t. each angle
void compute_grad(int idx, int N, float invN, float N2, float N3,
                  in RWStructuredBuffer<float> angles,
                  in RWStructuredBuffer<float> Z2,
                  in RWStructuredBuffer<float> Z3,
                  in RWStructuredBuffer<float> lp_t,
                  out float grad[12])
{
    // First compute LP
    float lp[12];
    compute_lp(idx, N, invN, N2, N3, angles, Z2, Z3, lp);

    // Then gradient: d/d(theta_k) of 0.5 * sum((lp - lp_t)^2)
    // Uses the same trig arrays
    int base = idx * N;
    for (int k = 0; k < N; k++) {
        float ang = angles[base + k];
        float a2 = ang * 2.0;
        float a4 = ang * 4.0;
        float c2 = cos(a2), s2 = sin(a2);
        float c4 = cos(a4), s4 = sin(a4);

        // lp_d = lp_t - lp  (original convention)
        // grad contribution: -2 * sin2 * lp_d[0] + 2 * cos2 * lp_d[1] + ...
        // Then multiply by -2 at the end (descent direction)
        float val = (-2 * s2 * (lp_t[0] - lp[0]) +
                      2 * c2 * (lp_t[1] - lp[1]) +
                     -2 * s4 * (lp_t[2] - lp[2]) +
                      2 * c4 * (lp_t[3] - lp[3]) +
                     -2 * s2 * Z2[k] * (lp_t[4] - lp[4]) +
                      2 * c2 * Z2[k] * (lp_t[5] - lp[5]) +
                     -2 * s4 * Z2[k] * (lp_t[6] - lp[6]) +
                      2 * c4 * Z2[k] * (lp_t[7] - lp[7]) +
                     -2 * s2 * Z3[k] * (lp_t[8] - lp[8]) +
                      2 * c2 * Z3[k] * (lp_t[9] - lp[9]) +
                     -2 * s4 * Z3[k] * (lp_t[10] - lp[10]) +
                      2 * c4 * Z3[k] * (lp_t[11] - lp[11]));
        grad[k] = -val * 2;
    }
}

// Entry point: compute LP for batch of laminates
// Dispatch dimensions: ceil(M / 64) x 1 x 1
[numthreads(64, 1, 1)]
void compute_lp_kernel(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    int idx = int(tid.x);
    if (idx >= M) return;

    float lp[12];
    compute_lp(idx, N, invN, N2, N3, inputAngles, Z2_buffer, Z3_buffer, lp);

    int out_base = idx * 12;
    for (int j = 0; j < 12; j++)
        resultLP[out_base + j] = lp[j];
}

// Entry point: compute LP + gradient for batch
[numthreads(64, 1, 1)]
void compute_lp_grad_kernel(uint3 tid : SV_DispatchThreadID)
{
    int idx = int(tid.x);
    if (idx >= M) return;

    // First compute LP
    float lp[12];
    compute_lp(idx, N, invN, N2, N3, inputAngles, Z2_buffer, Z3_buffer, lp);

    // Write LP
    int out_base = idx * 12;
    for (int j = 0; j < 12; j++)
        resultLP[out_base + j] = lp[j];

    // Compute and write gradient
    float grad[12];
    compute_grad(idx, N, invN, N2, N3, inputAngles, Z2_buffer, Z3_buffer, lp_t_buffer, grad);

    int grad_base = idx * N;
    for (int j = 0; j < N; j++)
        resultLP[grad_base + j + M * 12] = grad[j];
}
