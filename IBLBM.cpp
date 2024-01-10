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

const int num_cylinders = 2; // Number of cylinders
const int particle_num_nodes[num_cylinders] = {36, 36}; // Number of surface nodes for each cylinder
const double particle_radius[num_cylinders] = {8, 8}; // Radius of each cylinder
const double particle_stiffness[num_cylinders] = {0.1, 0.1}; // Stiffness modulus for each cylinder
const double particle_center_x[num_cylinders] = {20, 50}; // Center position x-component for each cylinder
const double particle_center_y[num_cylinders] = {20, 20}; // Center position y-component for each cylinder
#define RIGID_CYLINDER
/// Structure for surface nodes of multiple cylinders
// The following code should not be modified when it is first used.

const double omega = 1. / tau; // relaxation frequency (inverse of relaxation time)
double ***pop, ***pop_old; // LBM populations (old and new)
double **density; // fluid density
double **velocity_x; // fluid velocity (x-component)
double **velocity_y; // fluid velocity (y-component)
double **force_x; // fluid force (x-component)
double **force_y; // fluid force (y-component)
double pop_eq[9]; // equilibrium populations
const double weight[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.}; // lattice weights

struct node_struct {
  node_struct() : x(0), y(0), x_ref(0), y_ref(0), vel_x(0), vel_y(0), force_x(0), force_y(0) {}

  double x; // current x-position
  double y; // current y-position
  double x_ref; // reference x-position
  double y_ref; // reference y-position
  double vel_x; // node velocity (x-component)
  double vel_y; // node velocity (y-component)
  double force_x; // node force (x-component)
  double force_y; // node force (y-component)
};

node_struct cylinder_nodes[num_cylinders][36]; // Array of nodes for each cylinder

struct particle_struct {
  // Constructor
  particle_struct(int id) {
    num_nodes = particle_num_nodes[id];
    radius = particle_radius[id];
    stiffness = particle_stiffness[id];
    center.x = particle_center_x[id];
    center.y = particle_center_y[id];
    center.x_ref = particle_center_x[id];
    center.y_ref = particle_center_y[id];
    node = new node_struct[num_nodes];

    // Initialize the nodes for the cylinder
    for(int n = 0; n < num_nodes; ++n) {
      #if defined RIGID_CYLINDER || defined DEFORMABLE_CYLINDER
        node[n].x = center.x + radius * sin(2. * M_PI * (double) n / num_nodes);
        node[n].x_ref = center.x + radius * sin(2. * M_PI * (double) n / num_nodes);
        node[n].y = center.y + radius * cos(2. * M_PI * (double) n / num_nodes);
        node[n].y_ref = center.y + radius * cos(2. * M_PI * (double) n / num_nodes);
      #endif
      // ... (rest of your initialization code)
    }
  }

  // Elements
  int num_nodes; // number of surface nodes
  double radius; // object radius
  double stiffness; // stiffness modulus
  node_struct center; // center node
  node_struct *node; // list of nodes
};

// Array of particle_struct for each cylinder
particle_struct particles[num_cylinders] = {particle_struct(0), particle_struct(1)};

// The following functions are used in the simulation code.

void initialize(); // allocate memory and initialize variables
void LBM(int); // perform LBM operations
void momenta(); // compute fluid density and velocity from the populations
void equilibrium(double, double, double); // compute the equilibrium populations from the fluid density and velocity
void compute_particle_forces(int); // compute the forces acting on the object nodes
void spread(); // spread node forces to fluid lattice
void interpolate(); // interpolate node velocities from fluid velocity
void update_particle_position(); // update object center position
void write_fluid_vtk(int); // write the fluid state to the disk as VTK file
void write_particle_vtk(int); // write the particle state to the disk as VTK file
void write_data(int); // write data to the disk (drag/lift, center position)


int main() {

  // Preparations
  initialize(); // allocate memory and initialize variables
  // Now the particles are initialized within the initialize() function

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

    compute_particle_forces(t); // compute forces for all particles
    spread(); // spread forces from all particles
    LBM(t); // perform LBM operations
    for (int p = 0; p < num_cylinders; p++) {
      interpolate(); // interpolate velocity for each particle
      update_particle_position(); // update position for all particles
    }

    // Write to VTK files and data file
    if(t % t_disk == 0) {
      write_fluid_vtk(t);
      for (int p = 0; p < num_cylinders; p++) {
        write_particle_vtk(t); // write particle VTK for each particle
      }
      write_data(t); // write data for all particles
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
  // Make sure that the VTK folders exist.
  // Old file data.dat is deleted, if existing.

  int ignore; // ignore return value of system calls
  ignore = system("mkdir -p vtk_fluid"); // create folder if not existing
  ignore = system("mkdir -p vtk_particle"); // create folder if not existing
  ignore = system("rm -f data.dat"); // delete file if existing

  /// Allocate memory for the fluid density, velocity, and force

  density = new double*[Nx];
  velocity_x = new double*[Nx];
  velocity_y = new double*[Nx];
  force_x = new double*[Nx];
  force_y = new double*[Nx];

  for(int X = 0; X < Nx; ++X) {
    density[X] = new double[Ny];
    velocity_x[X] = new double[Ny];
    velocity_y[X] = new double[Ny];
    force_x[X] = new double[Ny];
    force_y[X] = new double[Ny];
  }

  /// Initialize the fluid density and velocity
  // Start with unit density and zero velocity.

  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      density[X][Y] = 1;
      velocity_x[X][Y] = 0;
      velocity_y[X][Y] = 0;
      force_x[X][Y] = 0;
      force_y[X][Y] = 0;
    }
  }

  /// Allocate memory for the populations

  pop = new double**[9];
  pop_old = new double**[9];

  for(int c_i = 0; c_i < 9; ++c_i) {
    pop[c_i] = new double*[Nx];
    pop_old[c_i] = new double*[Nx];

    for(int X = 0; X < Nx; ++X) {
      pop[c_i][X] = new double[Ny];
      pop_old[c_i][X] = new double[Ny];

      for(int Y = 0; Y < Ny; ++Y) {
        pop[c_i][X][Y] = 0;
        pop_old[c_i][X][Y] = 0;
      }
    }
  }

  /// Initialize the populations
  // Use the equilibrium populations corresponding to the initialized fluid density and velocity.

  for(int X = 0; X < Nx; ++X) {
    for(int Y = 0; Y < Ny; ++Y) {
      equilibrium(density[X][Y], velocity_x[X][Y], velocity_y[X][Y]);

      for(int c_i = 0; c_i < 9; ++c_i) {
        pop_old[c_i][X][Y] = pop_eq[c_i];
        pop[c_i][X][Y] = pop_eq[c_i];
      }
    }
  }

  for (int i = 0; i < num_cylinders; i++) {
    particles[i] = particle_struct(i); // Initialize each particle using the constructor
  }

  return;
}


/// *******************
/// COMPUTE EQUILIBRIUM
/// *******************

// This function computes the equilibrium populations from the fluid density and velocity.
// It computes the equilibrium only at a specific lattice node: Function has to be called at each lattice node.
// The standard quadratic euilibrium is used.
// reminder: SQ(x) = x * x

void equilibrium(double den, double vel_x, double vel_y) {
  pop_eq[0] = weight[0] * den * (1                                                     - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[1] = weight[1] * den * (1 + 3 * (  vel_x        ) + 4.5 * SQ(  vel_x        ) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[2] = weight[2] * den * (1 + 3 * (- vel_x        ) + 4.5 * SQ(- vel_x        ) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[3] = weight[3] * den * (1 + 3 * (          vel_y) + 4.5 * SQ(          vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[4] = weight[4] * den * (1 + 3 * (        - vel_y) + 4.5 * SQ(        - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[5] = weight[5] * den * (1 + 3 * (  vel_x + vel_y) + 4.5 * SQ(  vel_x + vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[6] = weight[6] * den * (1 + 3 * (- vel_x - vel_y) + 4.5 * SQ(- vel_x - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[7] = weight[7] * den * (1 + 3 * (  vel_x - vel_y) + 4.5 * SQ(  vel_x - vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));
  pop_eq[8] = weight[8] * den * (1 + 3 * (- vel_x + vel_y) + 4.5 * SQ(- vel_x + vel_y) - 1.5 * (SQ(vel_x) + SQ(vel_y)));

  return;
}


void LBM(int time) {

  /// Swap populations
  // The present code used old and new populations which are swapped at the beginning of each time step.
  // This is sometimes called 'double-buffered' or 'ping-pong' algorithm.
  // This way, the old populations are not overwritten during propagation.
  // The resulting code is easier to write and to debug.
  // The memory requirement for the populations is twice as large.

  double ***swap_temp = pop_old;
  pop_old = pop;
  pop = swap_temp;

  /// Lattice Boltzmann equation
  // The lattice Boltzmann equation is solved in the following.
  // The algorithm includes
  // - computation of the lattice force
  // - combined collision and propagation (faster than first collision and then propagation)

  for(int X = 0; X < Nx; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {

      /// Compute equilibrium
      // The equilibrium populations are computed.
      // Forces are coupled via Shan-Chen velocity shift.

      equilibrium(density[X][Y], (velocity_x[X][Y] + (force_x[X][Y] + gravity) * tau / density[X][Y]), (velocity_y[X][Y] + (force_y[X][Y]) * tau / density[X][Y]));

      /// Compute new populations
      // This is the lattice Boltzmann equation (combined collision and propagation) including external forcing.
      // Periodicity of the lattice in x-direction is taken into account by the %-operator.

      pop[0][X]                [Y]     = pop_old[0][X][Y] * (1 - omega) + pop_eq[0] * omega;
      pop[1][(X + 1) % Nx]     [Y]     = pop_old[1][X][Y] * (1 - omega) + pop_eq[1] * omega;
      pop[2][(X - 1 + Nx) % Nx][Y]     = pop_old[2][X][Y] * (1 - omega) + pop_eq[2] * omega;
      pop[3][X]                [Y + 1] = pop_old[3][X][Y] * (1 - omega) + pop_eq[3] * omega;
      pop[4][X]                [Y - 1] = pop_old[4][X][Y] * (1 - omega) + pop_eq[4] * omega;
      pop[5][(X + 1) % Nx]     [Y + 1] = pop_old[5][X][Y] * (1 - omega) + pop_eq[5] * omega;
      pop[6][(X - 1 + Nx) % Nx][Y - 1] = pop_old[6][X][Y] * (1 - omega) + pop_eq[6] * omega;
      pop[7][(X + 1) % Nx]     [Y - 1] = pop_old[7][X][Y] * (1 - omega) + pop_eq[7] * omega;
      pop[8][(X - 1 + Nx) % Nx][Y + 1] = pop_old[8][X][Y] * (1 - omega) + pop_eq[8] * omega;
    }
  }

  /// Bounce-back
  // Due to the presence of the rigid walls at y = 0 and y = Ny - 1, the populations have to be bounced back.
  // Ladd's momentum correction term is included for moving walls (wall velocity parallel to x-axis).
  // Periodicity of the lattice in x-direction is taken into account via the %-operator.

  for(int X = 0; X < Nx; ++X) {

    /// Bottom wall (y = 0)

    pop[3][X][1] = pop[4][X]                [0];
    pop[5][X][1] = pop[6][(X - 1 + Nx) % Nx][0] + 6 * weight[6] * density[X][1] * wall_vel_bottom;
    pop[8][X][1] = pop[7][(X + 1) % Nx]     [0] - 6 * weight[7] * density[X][1] * wall_vel_bottom;

    /// Top wall (y = Ny - 1)

    pop[4][X][Ny - 2] = pop[3][X]                [Ny - 1];
    pop[6][X][Ny - 2] = pop[5][(X + 1) % Nx]     [Ny - 1] - 6 * weight[5] * density[X][Ny - 2] * wall_vel_top;
    pop[7][X][Ny - 2] = pop[8][(X - 1 + Nx) % Nx][Ny - 1] + 6 * weight[8] * density[X][Ny - 2] * wall_vel_top;
  }

  /// Compute fluid density and velocity
  // The fluid density and velocity are obtained from the populations.

  momenta();

  return;
}

void momenta() {
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      density[X][Y] = pop[0][X][Y] + pop[1][X][Y] + pop[2][X][Y] + pop[3][X][Y] + pop[4][X][Y] + pop[5][X][Y] + pop[6][X][Y] + pop[7][X][Y] + pop[8][X][Y];
      velocity_x[X][Y] = (pop[1][X][Y] - pop[2][X][Y] + pop[5][X][Y] - pop[6][X][Y] + pop[7][X][Y] - pop[8][X][Y]) / density[X][Y];
      velocity_y[X][Y] = (pop[3][X][Y] - pop[4][X][Y] + pop[5][X][Y] - pop[6][X][Y] - pop[7][X][Y] + pop[8][X][Y]) / density[X][Y];
    }
  }

  return;
}

void compute_particle_forces(int time) {
  for (int p = 0; p < num_cylinders; p++) {
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
  return;
}


void spread() {
  // Reset forces on the fluid lattice
  for(int X = 0; X < Nx; ++X) {
    for(int Y = 1; Y < Ny - 1; ++Y) {
      force_x[X][Y] = 0;
      force_y[X][Y] = 0;
    }
  }

  // Loop over each cylinder
  for (int p = 0; p < num_cylinders; p++) {
    particle_struct &particle = particles[p];

    // Spread forces from each cylinder
    for(int n = 0; n < particle.num_nodes; ++n) {
      int x_int = (int) (particle.node[n].x - 0.5 + Nx) % Nx;
      int y_int = (int) (particle.node[n].y + 0.5);

      // Loop over neighboring fluid nodes
      for(int X = x_int; X <= x_int + 1; ++X) {
        for(int Y = y_int; Y <= y_int + 1; ++Y) {
          const double dist_x = particle.node[n].x - 0.5 - X;
          const double dist_y = particle.node[n].y + 0.5 - Y;

          const double weight_x = 1 - abs(dist_x);
          const double weight_y = 1 - abs(dist_y);

          // Apply the forces to the fluid lattice
          force_x[(X + Nx) % Nx][Y] += (particle.node[n].force_x * weight_x * weight_y);
          force_y[(X + Nx) % Nx][Y] += (particle.node[n].force_y * weight_x * weight_y);
        }
      }
    }
  }

  return;
}

void interpolate() {
  for (int p = 0; p < num_cylinders; p++) {
    particle_struct &particle = particles[p];

    // Run over all object nodes
    for(int n = 0; n < particle.num_nodes; ++n) {
      // Reset node velocity first since '+=' is used.
      particle.node[n].vel_x = 0;
      particle.node[n].vel_y = 0;

      // Identify the lowest fluid lattice node in interpolation range
      int x_int = (int) (particle.node[n].x - 0.5 + Nx) % Nx;
      int y_int = (int) (particle.node[n].y + 0.5);

      // Run over all neighboring fluid nodes (2x2 for bi-linear interpolation)
      for(int X = x_int; X <= x_int + 1; ++X) {
        for(int Y = y_int; Y <= y_int + 1; ++Y) {
          // Compute distance between object node and fluid lattice node
          const double dist_x = particle.node[n].x - 0.5 - X;
          const double dist_y = particle.node[n].y + 0.5 - Y;

          // Compute interpolation weights
          const double weight_x = 1 - abs(dist_x);
          const double weight_y = 1 - abs(dist_y);

          // Compute node velocities
          particle.node[n].vel_x += ((velocity_x[(X + Nx) % Nx][Y] + 0.5 * (force_x[(X + Nx) % Nx][Y] + gravity) / density[(X + Nx) % Nx][Y]) * weight_x * weight_y);
          particle.node[n].vel_y += ((velocity_y[(X + Nx) % Nx][Y] + 0.5 * (force_y[(X + Nx) % Nx][Y]) / density[(X + Nx) % Nx][Y]) * weight_x * weight_y);
        }
      }
    }
  }
  return;
}



void update_particle_position() {
  for (int p = 0; p < num_cylinders; p++) {
    particle_struct &particle = particles[p];

    // Reset center position
    particle.center.x = 0;
    particle.center.y = 0;

    // Update node and center positions
    for(int n = 0; n < particle.num_nodes; ++n) {
      particle.node[n].x += particle.node[n].vel_x;
      particle.node[n].y += particle.node[n].vel_y;
      particle.center.x += particle.node[n].x / particle.num_nodes;
      particle.center.y += particle.node[n].y / particle.num_nodes;
    }

    // Check for periodicity along the x-axis
    if(particle.center.x < 0) {
      particle.center.x += Nx;
      for(int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].x += Nx;
      }
    }
    else if(particle.center.x >= Nx) {
      particle.center.x -= Nx;
      for(int n = 0; n < particle.num_nodes; ++n) {
        particle.node[n].x -= Nx;
      }
    }
  }
  return;
}

void write_fluid_vtk(int time) {

  /// Create filename

  stringstream output_filename;
  output_filename << "vtk_fluid/fluid_t" << time << ".vtk";
  ofstream output_file;

  /// Open file

  output_file.open(output_filename.str().c_str());

  /// Write VTK header

  output_file << "# vtk DataFile Version 3.0\n";
  output_file << "fluid_state\n";
  output_file << "ASCII\n";
  output_file << "DATASET RECTILINEAR_GRID\n";
  output_file << "DIMENSIONS " << Nx << " " << Ny - 2 << " 1" << "\n";
  output_file << "X_COORDINATES " << Nx << " float\n";

  for(int X = 0; X < Nx; ++X) {
    output_file << X + 0.5 << " ";
  }

  output_file << "\n";
  output_file << "Y_COORDINATES " << Ny - 2 << " float\n";

  for(int Y = 1; Y < Ny - 1; ++Y) {
    output_file << Y - 0.5 << " ";
  }

  output_file << "\n";
  output_file << "Z_COORDINATES " << 1 << " float\n";
  output_file << 0 << "\n";
  output_file << "POINT_DATA " << Nx * (Ny - 2) << "\n";

  /// Write density difference

  output_file << "SCALARS density_difference float 1\n";
  output_file << "LOOKUP_TABLE default\n";

  for(int Y = 1; Y < Ny - 1; ++Y) {
    for(int X = 0; X < Nx; ++X) {
      output_file << density[X][Y] - 1 << "\n";
    }
  }

  /// Write velocity

  output_file << "VECTORS velocity_vector float\n";

  for(int Y = 1; Y < Ny - 1; ++Y) {
    for(int X = 0; X < Nx; ++X) {
      output_file << velocity_x[X][Y] + 0.5 * (force_x[X][Y] + gravity) / density[X][Y] << " " << velocity_y[X][Y] + 0.5 * (force_y[X][Y]) / density[X][Y] << " 0\n";
    }
  }

  /// Close file

  output_file.close();

  return;
}


void write_particle_vtk(int time) {
  // Create filename
  stringstream output_filename;
  output_filename << "vtk_particle/particles_t" << time << ".vtk";
  ofstream output_file;

  // Open file
  output_file.open(output_filename.str().c_str());

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
  for (int p = 0; p < num_cylinders; p++) {
    for(int n = 0; n < particles[p].num_nodes; ++n) {
      output_file << particles[p].node[n].x << " " << particles[p].node[n].y << " 0\n";
    }
  }

  // Write lines between neighboring nodes
  int line_count = 0;
  output_file << "LINES " << total_num_nodes << " " << 3 * total_num_nodes << "\n";
  for (int p = 0; p < num_cylinders; p++) {
    for(int n = 0; n < particles[p].num_nodes; ++n) {
      output_file << "2 " << line_count << " " << (line_count + 1) % (line_count + particles[p].num_nodes) << "\n";
      line_count++;
    }
  }

  // Write vertices
  output_file << "VERTICES 1 " << total_num_nodes + 1 << "\n";
  output_file << total_num_nodes << " ";
  for(int n = 0; n < total_num_nodes; ++n) {
    output_file << n << " ";
  }

  // Close file
  output_file.close();

  return;
}

void write_data(int time) {
  // Create filename
  string output_filename("data.dat");
  ofstream output_file;

  // Open file
  output_file.open(output_filename.c_str(), fstream::app);

  // Loop over each cylinder and compute quantities
  for (int p = 0; p < num_cylinders; p++) {
    particle_struct &particle = particles[p];

    double force_tot_x = 0;
    double force_tot_y = 0;
    double vel_center_x = 0;
    double vel_center_y = 0;

    for(int i = 0; i < particle.num_nodes; ++i) {
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

  return;
}




