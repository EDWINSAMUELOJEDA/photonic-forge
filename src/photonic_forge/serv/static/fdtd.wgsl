/*
 * PhotonicForge WebGPU FDTD Kernel
 * 
 * Simple 2D TMz mode FDTD solver.
 * Grid Layout: 
 *   Hz at (i+0.5, j+0.5)
 *   Ex at (i+0.5, j)
 *   Ey at (i, j+0.5)
 */

struct Params {
    nx: u32,
    ny: u32,
    dt: f32,
    dx: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> epsilon: array<f32>;
@group(0) @binding(2) var<storage, read_write> hz: array<f32>;
@group(0) @binding(3) var<storage, read_write> ex: array<f32>;
@group(0) @binding(4) var<storage, read_write> ey: array<f32>;

// Helper to get index
fn get_idx(x: u32, y: u32) -> u32 {
    return y * params.nx + x;
}

@compute @workgroup_size(16, 16)
fn update_h(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= params.nx - 1 || j >= params.ny - 1) {
        return;
    }

    // Standard Yee update for Hz
    // Hz(t+0.5) = Hz(t-0.5) - (dt/mu*dx) * [ (Ey(i+1) - Ey(i)) - (Ex(j+1) - Ex(j)) ]
    // Assuming mu = 1.0 (non-magnetic)
    
    let idx = get_idx(i, j);
    let idx_ex_j = get_idx(i, j);
    let idx_ex_j1 = get_idx(i, j + 1);
    let idx_ey_i = get_idx(i, j);
    let idx_ey_i1 = get_idx(i + 1, j);

    // Curl E
    let dEy = ey[idx_ey_i1] - ey[idx_ey_i];
    let dEx = ex[idx_ex_j1] - ex[idx_ex_j];

    let factor = params.dt / params.dx;
    
    hz[idx] -= factor * (dEy - dEx);
}

@compute @workgroup_size(16, 16)
fn update_e(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= params.nx || j >= params.ny) {
        return;
    }

    let idx = get_idx(i, j);
    let eps = epsilon[idx];
    let factor = (params.dt / eps) / params.dx;

    // Update Ex (requires Hz(j) and Hz(j-1))
    if (j > 0 && j < params.ny - 1 && i < params.nx - 1) {
        // Ex(t+1) = Ex(t) + (dt/eps*dx) * (Hz(j) - Hz(j-1))
        let hz_now = hz[get_idx(i, j)];
        let hz_prev = hz[get_idx(i, j - 1)];
        ex[idx] += factor * (hz_now - hz_prev);
    }

    // Update Ey (requires Hz(i) and Hz(i-1))
    if (i > 0 && i < params.nx - 1 && j < params.ny - 1) {
        // Ey(t+1) = Ey(t) - (dt/eps*dx) * (Hz(i) - Hz(i-1))
        let hz_now = hz[get_idx(i, j)];
        let hz_prev = hz[get_idx(i - 1, j)];
        ey[idx] -= factor * (hz_now - hz_prev);
    }
}

// Source injection kernel (simplified soft source)
@group(0) @binding(5) var<uniform> source_val: f32;
// In real impl, we'd pass source pos as uniform
@compute @workgroup_size(1)
fn update_source(@builtin(global_invocation_id) global_id: vec3<u32>) {
     // Hardcoded center source for test
     // ideally passed as params
     let cx = params.nx / 4;
     let cy = params.ny / 2;
     let idx = get_idx(cx, cy);
     hz[idx] += source_val; 
}
