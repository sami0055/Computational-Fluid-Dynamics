/// Particle properties for two cylinders

#include <vector> // vector containers
#include <cmath> // mathematical library
#include <iostream> // for the use of 'cout'
#include <fstream> // file streams
#include <sstream> // string streams
#include <cstdlib> // standard library
#define SQ(x) ((x) * (x)) 
/// Fluid/lattice properties
using namespace std;

const int Nx = 100; // number of lattice nodes along the x-axis (periodic)
const int Ny = 42; // number of lattice nodes along the y-axis (including two wall nodes)
const double tau = 0.55; // relaxation time
const int t_num = 40000; // number of time steps (running from 1 to t_num)
const int t_disk = 200; // disk write time step (data will be written to the disk every t_disk step)
const int t_info = 1000; // info time step (screen message will be printed every t_info step)
const double gravity = 0.00001; // force density due to gravity (in positive x-direction)
const double wall_vel_bottom = 0; // velocity of the bottom wall (in positive x-direction)
const double wall_vel_top = 0; // velocity of the top wall (in positive x-direction)

#include <cuda_runtime.h>

// const int num_cylinders = 2;
// const int particle_num_nodes[num_cylinders] = {36, 36};
// const double particle_radius[num_cylinders] = {8, 8};
// const double particle_stiffness[num_cylinders] = {0.1, 0.1};
// const double particle_center_x[num_cylinders] = {20, 50};
// const double particle_center_y[num_cylinders] = {20, 20};

#define NUM_CYLINDERS 2

__constant__ int num_cylinders = NUM_CYLINDERS;
__constant__ int particle_num_nodes[NUM_CYLINDERS] = {36, 36};
__constant__ double particle_radius[NUM_CYLINDERS] = {8, 8};
__constant__ double particle_stiffness[NUM_CYLINDERS] = {0.1, 0.1};
__constant__ double particle_center_x[NUM_CYLINDERS] = {20, 50};
__constant__ double particle_center_y[NUM_CYLINDERS] = {20, 20};

#define RIGID_CYLINDER

const double omega = 1. / tau; // relaxation frequency (inverse of relaxation time)
__device__ double pop_eq[9]; // equilibrium populations
__device__ const double weight[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.}; // lattice weights

__device__ double *pop, *pop_old; // LBM populations (old and new)
__device__ double *density; // fluid density
__device__ double *velocity_x; // fluid velocity (x-component)
__device__ double *velocity_y; // fluid velocity (y-component)
__device__ double *force_x; // fluid force (x-component)
__device__ double *force_y; // fluid force (y-component)

struct node_struct {
  double x; // current x-position
  double y; // current y-position
  double x_ref; // reference x-position
  double y_ref; // reference y-position
  double vel_x; // node velocity (x-component)
  double vel_y; // node velocity (y-component)
  double force_x; // node force (x-component)
  double force_y; // node force (y-component)
};

struct particle_struct {
  int num_nodes; // number of surface nodes
  double radius; // object radius
  double stiffness; // stiffness modulus
  node_struct center; // center node
  node_struct *node; // list of nodes
};

particle_struct *particles; // Array of particle_struct for each cylinder

// Device function declarations
__device__ void equilibrium(double den, double vel_x, double vel_y, double *pop_eq);
__global__ void lbm_kernel(double *pop, double *pop_old, double *density, double *velocity_x, double *velocity_y, double *force_x, double *force_y);
__global__ void compute_particle_forces_kernel(particle_struct *particles);
__global__ void spread_kernel(double *force_x, double *force_y, particle_struct *particles);
__global__ void interpolate_kernel(double *velocity_x, double *velocity_y, double *force_x, double *force_y, double *density, particle_struct *particles);
__global__ void update_particle_position_kernel(particle_struct *particles);
__global__ void initialize_populations_kernel(double *pop, double *pop_old, double *density, double *velocity_x, double *velocity_y);
  __global__ void initialize_particles_kernel(particle_struct *particles);
__global__ void momenta_kernel(double *pop, double *density, double *velocity_x, double *velocity_y);

// Host function declarations
void initialize();
//void momenta();
void write_fluid_vtk(int);
void write_particle_vtk(int);
void write_data(int);

int main() {
  // Preparations
  void initialize_constants_for_device();
  initialize(); // allocate memory and initialize variables

  // Compute derived quantities
  const double D = Ny - 2; // inner channel diameter
  const double nu = (tau - 0.5) / 3; // lattice viscosity
  const double umax = gravity / (2 * nu) * SQ(0.5 * D); // expected max velocity for Poiseuille flow without object
  const double Re = D * umax / nu; // Reynolds number for Poiseuille flow without object

  // Report derived parameters
  cout << "simulation parameters" << endl;
  cout << "=====================" << endl;
  cout << "D = " << D << endl;
  cout << "nu = " << nu << endl;
  cout << "umax = " << umax << endl;
  cout << "Re = " << Re << endl;
  cout << endl;

  // Starting simulation
  cout << "starting simulation" << endl;
  srand(1);

  for(int t = 1; t <= t_num; ++t) { // run over all times between 1 and t_num
    compute_particle_forces_kernel<<<(NUM_CYLINDERS + 255) / 256, 256>>>(particles);
    spread_kernel<<<(Nx * (Ny - 2) + 255) / 256, 256>>>(force_x, force_y, particles);
    lbm_kernel<<<(Nx * (Ny - 2) + 255) / 256, 256>>>(pop, pop_old, density, velocity_x, velocity_y, force_x, force_y);
    interpolate_kernel<<<(NUM_CYLINDERS * 36 + 255) / 256, 256>>>(velocity_x, velocity_y, force_x, force_y, density, particles);
    update_particle_position_kernel<<<(NUM_CYLINDERS + 255) / 256, 256>>>(particles);

    // Write to VTK files and data file
    if(t % t_disk == 0) {
      write_fluid_vtk(t);
      write_particle_vtk(t);
      write_data(t);
    }

    // Report end of time step
    if(t % t_info == 0) {
      cout << "completed time step " << t << " in [1, " << t_num << "]" << endl;
    }
  }

  // End of simulation
  cout << "simulation complete" << endl;

  return 0;
}

void initialize() {
  /// Create folders, delete data file
  int ignore; // ignore return value of system calls
  ignore = system("mkdir -p vtk_fluid"); // create folder if not existing
  ignore = system("mkdir -p vtk_particle"); // create folder if not existing
  ignore = system("rm -f data.dat"); // delete file if existing

  /// Allocate memory on the device
  cudaMalloc(&pop, sizeof(double) * Nx * Ny * 9);
  cudaMalloc(&pop_old, sizeof(double) * Nx * Ny * 9);
  cudaMalloc(&density, sizeof(double) * Nx * Ny);
  cudaMalloc(&velocity_x, sizeof(double) * Nx * Ny);
  cudaMalloc(&velocity_y, sizeof(double) * Nx * Ny);
  cudaMalloc(&force_x, sizeof(double) * Nx * Ny);
  cudaMalloc(&force_y, sizeof(double) * Nx * Ny);

  /// Initialize the fluid density and velocity on the device
  double *temp_density, *temp_velocity_x, *temp_velocity_y;
  cudaMallocHost(&temp_density, sizeof(double) * Nx * Ny);
  cudaMallocHost(&temp_velocity_x, sizeof(double) * Nx * Ny);
  cudaMallocHost(&temp_velocity_y, sizeof(double) * Nx * Ny);

  // Start with unit density and zero velocity
  for (int i = 0; i < Nx * Ny; ++i) {
    temp_density[i] = 1;
    temp_velocity_x[i] = 0;
    temp_velocity_y[i] = 0;
  }

  // Transfer the initial data to the device
  cudaMemcpy(density, temp_density, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_x, temp_velocity_x, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
  cudaMemcpy(velocity_y, temp_velocity_y, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);

  cudaFreeHost(temp_density);
  cudaFreeHost(temp_velocity_x);
  cudaFreeHost(temp_velocity_y);

  /// Initialize the populations on the device
  initialize_populations_kernel<<<(Nx * Ny + 255) / 256, 256>>>(pop, pop_old, density, velocity_x, velocity_y);

  // Allocate memory for particles on the device
  cudaMalloc(&particles, sizeof(particle_struct) * num_cylinders);

  // Initialize particles on the device
  initialize_particles_kernel<<<(num_cylinders + 255) / 256, 256>>>(particles);

  return;
}

__device__ void equilibrium(double den, double vel_x, double vel_y, double *pop_eq) {
  pop_eq[0] = weight[0] * den * (1                                                     - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[1] = weight[1] * den * (1 + 3 * (  vel_x        ) + 4.5 * SQ(  vel_x        ) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[2] = weight[2] * den * (1 + 3 * (- vel_x        ) + 4.5 * SQ(- vel_x        ) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[3] = weight[3] * den * (1 + 3 * (          vel_y) + 4.5 * SQ(          vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[4] = weight[4] * den * (1 + 3 * (        - vel_y) + 4.5 * SQ(        - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[5] = weight[5] * den * (1 + 3 * (  vel_x + vel_y) + 4.5 * SQ(  vel_x + vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[6] = weight[6] * den * (1 + 3 * (- vel_x - vel_y) + 4.5 * SQ(- vel_x - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[7] = weight[7] * den * (1 + 3 * (  vel_x - vel_y) + 4.5 * SQ(  vel_x - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[8] = weight[8] * den * (1 + 3 * (- vel_x + vel_y) + 4.5 * SQ(- vel_x + vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
}

__global__ void lbm_kernel(double *pop, double *pop_old, double *density, double *velocity_x, double *velocity_y, double *force_x, double *force_y) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;

  if (X < Nx && Y >= 1 && Y < Ny - 1) {
    int idx = X * Ny + Y;

    // Compute equilibrium populations
    equilibrium(density[idx], velocity_x[idx] + (force_x[idx] + gravity) * tau / density[idx], velocity_y[idx] + (force_y[idx]) * tau / density[idx], pop_eq);

    // Compute new populations
    pop[idx * 9 + 0] = pop_old[idx * 9 + 0] * (1 - omega) + pop_eq[0] * omega;
    pop[(((X + 1) % Nx) * Ny + Y) * 9 + 1] = pop_old[idx * 9 + 1] * (1 - omega) + pop_eq[1] * omega;
    pop[(((X - 1 + Nx) % Nx) * Ny + Y) * 9 + 2] = pop_old[idx * 9 + 2] * (1 - omega) + pop_eq[2] * omega;
    pop[(X * Ny + (Y + 1)) * 9 + 3] = pop_old[idx * 9 + 3] * (1 - omega) + pop_eq[3] * omega;
    pop[(X * Ny + (Y - 1)) * 9 + 4] = pop_old[idx * 9 + 4] * (1 - omega) + pop_eq[4] * omega;
    pop[(((X + 1) % Nx) * Ny + (Y + 1)) * 9 + 5] = pop_old[idx * 9 + 5] * (1 - omega) + pop_eq[5] * omega;
    pop[(((X - 1 + Nx) % Nx) * Ny + (Y - 1)) * 9 + 6] = pop_old[idx * 9 + 6] * (1 - omega) + pop_eq[6] * omega;
    pop[(((X + 1) % Nx) * Ny + (Y - 1)) * 9 + 7] = pop_old[idx * 9 + 7] * (1 - omega) + pop_eq[7] * omega;
    pop[(((X - 1 + Nx) % Nx) * Ny + (Y + 1)) * 9 + 8] = pop_old[idx * 9 + 8] * (1 - omega) + pop_eq[8] * omega;
  }
}


__global__ void momenta_kernel(double *pop, double *density, double *velocity_x, double *velocity_y) {
  int X = blockIdx.x * blockDim.x + threadIdx.x;
  int Y = blockIdx.y * blockDim.y + threadIdx.y;

  if (X < Nx && Y >= 1 && Y < Ny - 1) {
    int idx = X * Ny + Y;
    density[idx] = pop[idx * 9 + 0] + pop[idx * 9 + 1] + pop[idx * 9 + 2] + pop[idx * 9 + 3] + pop[idx * 9 + 4] + pop[idx * 9 + 5] + pop[idx * 9 + 6] + pop[idx * 9 + 7] + pop[idx * 9 + 8];
    velocity_x[idx] = (pop[idx * 9 + 1] - pop[idx * 9 + 2] + pop[idx * 9 + 5] - pop[idx * 9 + 6] + pop[idx * 9 + 7] - pop[idx * 9 + 8]) / density[idx];
    velocity_y[idx] = (pop[idx * 9 + 3] - pop[idx * 9 + 4] + pop[idx * 9 + 5] - pop[idx * 9 + 6] - pop[idx * 9 + 7] + pop[idx * 9 + 8]) / density[idx];
  }
}
__global__ void compute_particle_forces_kernel(particle_struct *particles) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (p < num_cylinders) {
      particle_struct &particle = particles[p];
  
      // Reset forces
      for (int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].force_x = 0;
        particle.node[n].force_y = 0;
      }
  
      // Compute rigid forces for RIGID_CYLINDER
  #ifdef RIGID_CYLINDER
      const double area = 2 * M_PI * particle.radius / particle.num_nodes; // area belonging to a node
  
      for (int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].force_x += -particle.stiffness * (particle.node[n].x - particle.node[n].x_ref) * area;
        particle.node[n].force_y += -particle.stiffness * (particle.node[n].y - particle.node[n].y_ref) * area;
      }
  #endif
    }
  }

  __global__ void spread_kernel(particle_struct *particles, double *force_x, double *force_y) {
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (X < Nx && Y >= 1 && Y < Ny - 1) {
      int idx = X * Ny + Y;
      force_x[idx] = 0;
      force_y[idx] = 0;
  
      for (int p = 0; p < num_cylinders; p++) {
        particle_struct &particle = particles[p];
  
        for (int n = 0; n < particle.num_nodes; ++n) {
          int x_int = (int)(particle.node[n].x - 0.5 + Nx) % Nx;
          int y_int = (int)(particle.node[n].y + 0.5);
  
          for (int X_nbr = x_int; X_nbr <= x_int + 1; ++X_nbr) {
            for (int Y_nbr = y_int; Y_nbr <= y_int + 1; ++Y_nbr) {
              const double dist_x = particle.node[n].x - 0.5 - X_nbr;
              const double dist_y = particle.node[n].y + 0.5 - Y_nbr;
  
              const double weight_x = 1 - abs(dist_x);
              const double weight_y = 1 - abs(dist_y);
  
              const int idx_nbr = (X_nbr + Nx) % Nx * Ny + Y_nbr;
              force_x[idx_nbr] += (particle.node[n].force_x * weight_x * weight_y);
              force_y[idx_nbr] += (particle.node[n].force_y * weight_x * weight_y);
            }
          }
        }
      }
    }
  }
  
  __global__ void interpolate_kernel(particle_struct *particles, double *density, double *velocity_x, double *velocity_y, double *force_x, double *force_y) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (p < num_cylinders) {
      particle_struct &particle = particles[p];
  
      for (int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].vel_x = 0;
        particle.node[n].vel_y = 0;
  
        int x_int = (int)(particle.node[n].x - 0.5 + Nx) % Nx;
        int y_int = (int)(particle.node[n].y + 0.5);
  
        for (int X_nbr = x_int; X_nbr <= x_int + 1; ++X_nbr) {
          for (int Y_nbr = y_int; Y_nbr <= y_int + 1; ++Y_nbr) {
            const double dist_x = particle.node[n].x - 0.5 - X_nbr;
            const double dist_y = particle.node[n].y + 0.5 - Y_nbr;
  
            const double weight_x = 1 - abs(dist_x);
            const double weight_y = 1 - abs(dist_y);
  
            const int idx_nbr = (X_nbr + Nx) % Nx * Ny + Y_nbr;
            particle.node[n].vel_x += ((velocity_x[idx_nbr] + 0.5 * (force_x[idx_nbr] + gravity) / density[idx_nbr]) * weight_x * weight_y);
            particle.node[n].vel_y += ((velocity_y[idx_nbr] + 0.5 * (force_y[idx_nbr]) / density[idx_nbr]) * weight_x * weight_y);
          }
        }
      }
    }
  }
  
  __global__ void update_particle_position_kernel(particle_struct *particles) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (p < num_cylinders) {
      particle_struct &particle = particles[p];
  
      particle.center.x = 0;
      particle.center.y = 0;
  
      for (int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].x += particle.node[n].vel_x;
        particle.node[n].y += particle.node[n].vel_y;
        particle.center.x += particle.node[n].x / particle.num_nodes;
        particle.center.y += particle.node[n].y / particle.num_nodes;
      }
  
      if (particle.center.x < 0) {
        particle.center.x += Nx;
        for (int n = 0; n < particle.num_nodes; ++n) {
          particle.node[n].x += Nx;
        }
      }
      else if (particle.center.x >= Nx) {
        particle.center.x -= Nx;
        for (int n = 0; n < particle.num_nodes; ++n) {
          particle.node[n].x -= Nx;
        }
      }
    }
  }
  
  __global__ void initialize_populations_kernel(double *pop, double *pop_old, double *density, double *velocity_x, double *velocity_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (idx < Nx * Ny) {
      int X = idx / Ny;
      int Y = idx % Ny;
  
      if (Y >= 1 && Y < Ny - 1) {
        equilibrium(density[idx], velocity_x[idx], velocity_y[idx], pop_eq);
  
        for (int c_i = 0; c_i < 9; ++c_i) {
          pop_old[idx * 9 + c_i] = pop_eq[c_i];
          pop[idx * 9 + c_i] = pop_eq[c_i];
        }
      }
    }
  }
  
  __global__ void initialize_particles_kernel(particle_struct *particles) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (p < num_cylinders) {
      particle_struct &particle = particles[p];
  
      particle.num_nodes = particle_num_nodes[p];
      particle.radius = particle_radius[p];
      particle.stiffness = particle_stiffness[p];
      particle.center.x = particle_center_x[p];
      particle.center.y = particle_center_y[p];
      particle.center.x_ref = particle_center_x[p];
      particle.center.y_ref = particle_center_y[p];
  
      for (int n = 0; n < particle.num_nodes; ++n) {
  #if defined RIGID_CYLINDER || defined DEFORMABLE_CYLINDER
        particle.node[n].x = particle.center.x + particle.radius * sin(2. * M_PI * (double)n / particle.num_nodes);
        particle.node[n].x_ref = particle.center.x + particle.radius * sin(2. * M_PI * (double)n / particle.num_nodes);
        particle.node[n].y = particle.center.y + particle.radius * cos(2. * M_PI * (double)n / particle.num_nodes);
        particle.node[n].y_ref = particle.center.y + particle.radius * cos(2. * M_PI * (double)n / particle.num_nodes);
  #endif
      }
    }
  }

void write_fluid_vtk(int time) {
    // Create filename
    stringstream output_filename;
    output_filename << "vtk_fluid/fluid_t" << time << ".vtk";
    ofstream output_file;

    // Open file
    output_file.open(output_filename.str().c_str());

    // Allocate device memory for density, velocity_x, velocity_y, and force_x, force_y
    double *dev_density, *dev_velocity_x, *dev_velocity_y, *dev_force_x, *dev_force_y;
    cudaMalloc(&dev_density, sizeof(double) * Nx * Ny);
    cudaMalloc(&dev_velocity_x, sizeof(double) * Nx * Ny);
    cudaMalloc(&dev_velocity_y, sizeof(double) * Nx * Ny);
    cudaMalloc(&dev_force_x, sizeof(double) * Nx * Ny);
    cudaMalloc(&dev_force_y, sizeof(double) * Nx * Ny);

    // Flatten the arrays for cudaMemcpy
    double *flat_density = new double[Nx * Ny];
    double *flat_velocity_x = new double[Nx * Ny];
    double *flat_velocity_y = new double[Nx * Ny];
    double *flat_force_x = new double[Nx * Ny];
    double *flat_force_y = new double[Nx * Ny];

    // Flatten the arrays
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int idx = i * Ny + j;
            flat_density[idx] = density[i * Ny + j];
            flat_velocity_x[idx] = velocity_x[i * Ny + j];
            flat_velocity_y[idx] = velocity_y[i * Ny + j];
            flat_force_x[idx] = force_x[i * Ny + j];
            flat_force_y[idx] = force_y[i * Ny + j];
        }
    }

    // Copy data from host to device
    cudaMemcpy(dev_density, flat_density, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_velocity_x, flat_velocity_x, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_velocity_y, flat_velocity_y, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_force_x, flat_force_x, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_force_y, flat_force_y, sizeof(double) * Nx * Ny, cudaMemcpyHostToDevice);

    // Write VTK header
    output_file << "# vtk DataFile Version 3.0\n";
    output_file << "fluid_state\n";
    output_file << "ASCII\n";
    output_file << "DATASET RECTILINEAR_GRID\n";
    output_file << "DIMENSIONS " << Nx << " " << Ny - 2 << " 1\n";
    output_file << "X_COORDINATES " << Nx << " float\n";

    for (int X = 0; X < Nx; ++X) {
        output_file << X + 0.5 << " ";
    }

    output_file << "\n";
    output_file << "Y_COORDINATES " << Ny - 2 << " float\n";

    for (int Y = 1; Y < Ny - 1; ++Y) {
        output_file << Y - 0.5 << " ";
    }

    output_file << "\n";
    output_file << "Z_COORDINATES " << 1 << " float\n";
    output_file << 0 << "\n";
    output_file << "POINT_DATA " << Nx * (Ny - 2) << "\n";

    // Write density difference
    output_file << "SCALARS density_difference float 1\n";
    output_file << "LOOKUP_TABLE default\n";

    double *temp_density = new double[Nx * Ny];
    cudaMemcpy(temp_density, dev_density, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);

    for (int Y = 1; Y < Ny - 1; ++Y) {
        for (int X = 0; X < Nx; ++X) {
            output_file << temp_density[X * Ny + Y] - 1 << "\n";
        }
    }

    delete[] temp_density;

    // Write velocity
    output_file << "VECTORS velocity_vector float\n";

    double *temp_velocity_x = new double[Nx * Ny];
    double *temp_velocity_y = new double[Nx * Ny];
    double *temp_force_x = new double[Nx * Ny];
    double *temp_force_y = new double[Nx * Ny];

    cudaMemcpy(temp_velocity_x, dev_velocity_x, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_velocity_y, dev_velocity_y, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_force_x, dev_force_x, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_force_y, dev_force_y, sizeof(double) * Nx * Ny, cudaMemcpyDeviceToHost);

    for (int Y = 1; Y < Ny - 1; ++Y) {
        for (int X = 0; X < Nx; ++X) {
            int idx = X * Ny + Y;
            output_file << temp_velocity_x[idx] + 0.5 * (temp_force_x[idx] + gravity) / temp_density[idx] << " "
                        << temp_velocity_y[idx] + 0.5 * (temp_force_y[idx]) / temp_density[idx] << " 0\n";
        }
    }

    delete[] temp_velocity_x;
    delete[] temp_velocity_y;
    delete[] temp_force_x;
    delete[] temp_force_y;

    // Free device memory
    cudaFree(dev_density);
    cudaFree(dev_velocity_x);
    cudaFree(dev_velocity_y);
    cudaFree(dev_force_x);
    cudaFree(dev_force_y);

    // Close file
    output_file.close();

    // Free host memory
    delete[] flat_density;
    delete[] flat_velocity_x;
    delete[] flat_velocity_y;
    delete[] flat_force_x;
    delete[] flat_force_y;
}

void write_particle_vtk(int time) {
// Create filename
stringstream output_filename;
output_filename << "vtk_particle/particles_t" << time << ".vtk";
ofstream output_file;

// Open file
output_file.open(output_filename.str().c_str());

// Allocate device memory for particles
particle_struct *dev_particles;
cudaMalloc(&dev_particles, sizeof(particle_struct) * num_cylinders);

// Copy particles data from host to device
cudaMemcpy(dev_particles, particles, sizeof(particle_struct) * num_cylinders, cudaMemcpyHostToDevice);

// Write VTK header
output_file << "# vtk DataFile Version 3.0\n";
output_file << "particles_state\n";
output_file << "ASCII\n";
output_file << "DATASET POLYDATA\n";

// Count total number of nodes for all particles
int total_num_nodes = 0;
for (int p = 0; p < num_cylinders; p++) {
  total_num_nodes += particles[p].num_nodes;
}

// Write node positions
output_file << "POINTS " << total_num_nodes << " float\n";

double *temp_nodes = new double[total_num_nodes * 3];
int node_idx = 0;

for (int p = 0; p < num_cylinders; p++) {
  particle_struct particle;
  cudaMemcpy(&particle, &dev_particles[p], sizeof(particle_struct), cudaMemcpyDeviceToHost);

  for (int n = 0; n < particle.num_nodes; ++n) {
    temp_nodes[node_idx * 3] = particle.node[n].x;
    temp_nodes[node_idx * 3 + 1] = particle.node[n].y;
    temp_nodes[node_idx * 3 + 2] = 0;
    output_file << temp_nodes[node_idx * 3] << " " << temp_nodes[node_idx * 3 + 1] << " " << temp_nodes[node_idx * 3 + 2] << "\n";
    node_idx++;
  }
}

delete[] temp_nodes;

// Write lines between neighboring nodes
int line_count = 0;
output_file << "LINES " << total_num_nodes << " " << 3 * total_num_nodes << "\n";

for (int p = 0; p < num_cylinders; p++) {
  particle_struct particle;
  cudaMemcpy(&particle, &dev_particles[p], sizeof(particle_struct), cudaMemcpyDeviceToHost);

  for (int n = 0; n < particle.num_nodes; ++n) {
    output_file << "2 " << line_count << " " << (line_count + 1) % (line_count + particle.num_nodes) << "\n";
    line_count++;
  }
}

// Write vertices
output_file << "VERTICES 1 " << total_num_nodes + 1 << "\n";
output_file << total_num_nodes << " ";
for (int n = 0; n < total_num_nodes; ++n) {
  output_file << n << " ";
}

// Close file
output_file.close();

// Free device memory
cudaFree(dev_particles);
}



void write_data(int time) {
// Create filename
string output_filename("data.dat");
ofstream output_file;

// Open file
output_file.open(output_filename.c_str(), fstream::app);

// Allocate device memory for particles
particle_struct *dev_particles;
cudaMalloc(&dev_particles, sizeof(particle_struct) * num_cylinders);

// Copy particles data from host to device
cudaMemcpy(dev_particles, particles, sizeof(particle_struct) * num_cylinders, cudaMemcpyHostToDevice);

// Loop over each cylinder and compute quantities
for (int p = 0; p < num_cylinders; p++) {
  particle_struct particle;
  cudaMemcpy(&particle, &dev_particles[p], sizeof(particle_struct), cudaMemcpyDeviceToHost);

  double force_tot_x = 0;
  double force_tot_y = 0;
  double vel_center_x = 0;
  double vel_center_y = 0;

  for (int i = 0; i < particle.num_nodes; ++i) {
    force_tot_x += particle.node[i].force_x;
    force_tot_y += particle.node[i].force_y;
    vel_center_x += particle.node[i].vel_x;
    vel_center_y += particle.node[i].vel_y;
  }

  // Normalize center velocity
  vel_center_x /= particle.num_nodes;
  vel_center_y /= particle.num_nodes;

  // Write data for each particle
  output_file << time << " "; // time step
  output_file << force_tot_x << " "; // total force x-component
  output_file << force_tot_y << " "; // total force y-component
  output_file << particle.center.x << " "; // center position x-component
  output_file << particle.center.y << " "; // center position y-component
  output_file << vel_center_x << " "; // center velocity x-component
  output_file << vel_center_y << "\n"; // center velocity y-component
}

// Close file
output_file.close();

// Free device memory
cudaFree(dev_particles);
}




