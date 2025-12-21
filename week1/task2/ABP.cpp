#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std; 

double pi = 3.141592653589793;

int radius = 1;
float v0 = 1;
float dt = 0.01;
int steps = 10000;
int N = 500;
double phi = 0.5; 
double L = sqrt(N * pi * radius * radius / phi);
int Pe = 10; //Peclet number 
double Dr = v0 / (radius * Pe);


struct Particle {
    int id;
    std::array<double, 2> position; 
    std::array<double, 2> velocity; 
    std::array<double, 2> force;
    double theta;                    // orientation angle
    std::array<double, 2> n;         // orientation vector

};

struct Box{
    double phi ; //Area fraction
    int N; //number of particles
    double L; //box length
};

struct Neighbor {
    int id;                     // index of neighbor
    std::array<double, 2> dr;  // {dx, dy}
    double r2;                 // squared distance
};

class System
{
public:

    System(Box& box) : _box{box}, _numparticles{0} { } //constructor initialiser list

    std::vector<Particle>& get_particles() { 
        return _particles; 
    }
    int size() const { return _numparticles; }
    
    std::vector<std::vector<Neighbor>> neighbors;       // r < rc
    std::vector<std::vector<Neighbor>> skin_neighbors;  // r < rc + r_skin


    void build_neighbor_lists_full();
    void build_neighbor_lists_core();
    double max_displacement() const;

    inline void apply_pbc(std::array<double,2>& r) const { // apply periodic boundary conditions
        r[0] -= _box.L * std::floor(r[0] / _box.L);
        r[1] -= _box.L * std::floor(r[1] / _box.L);
    }

    inline void set_cutoffs(double rc, double rskin) {
        r_c = rc;
        r_skin = rskin;
    }

    inline double get_r_skin() const {
        return r_skin;
    }

private:

    Box& _box;                           // Simulation box
    std::vector<Particle> _particles;    // Standard library vector containing all particles
    int _numparticles;                   // Total number of particles in the system
    std::vector<std::array<double,2>> old_positions; //updated when skin neighbors are rebuilt
    double r_c;
    double r_skin;
    std::vector<std::array<double,2>> forces;

    inline double min_image(double dx) const{   //minimum image convention
        if (dx >  0.5 * _box.L) dx -= _box.L;
        if (dx < -0.5 * _box.L) dx += _box.L;
        return dx;
    }
};

void System::build_neighbor_lists_full()
{
    const int N = _particles.size();
    _numparticles = N;

    neighbors.assign(N, {});  // clear previous neighbor lists
    skin_neighbors.assign(N, {});
    old_positions.assign(N, {});

    const double rc2      = r_c * r_c;
    const double rskin    = r_c + r_skin;
    const double rskin2   = rskin * rskin;

    for (int i = 0; i < N; ++i) {
        const std::array<double,2>& ri = _particles[i].position;
        old_positions[i] = _particles[i].position;

        for (int j = i + 1; j < N; ++j) {
            const std::array<double,2>& rj = _particles[j].position;

            double dx = rj[0] - ri[0];
            double dy = rj[1] - ri[1];

            // minimum image
            dx = min_image(dx);
            dy = min_image(dy);

            // checking only particles within required sub_box
            if (std::abs(dx) > rskin || std::abs(dy) > rskin)
                continue;

            double r2 = dx*dx + dy*dy;

            // core neighbor list
            if (r2 < rc2) {
                neighbors[i].push_back({j, { dx,  dy}, r2});
                neighbors[j].push_back({i, {-dx, -dy}, r2});
            }

            // skin neighbor list
            if (r2 < rskin2) {
                skin_neighbors[i].push_back({j, { dx,  dy}, r2});
                skin_neighbors[j].push_back({i, {-dx, -dy}, r2});
            }
        }
    }
}

void System::build_neighbor_lists_core()
{
    const double rc2 = r_c * r_c;

    neighbors.assign(_numparticles, {});

    for (int i = 0; i < _numparticles; ++i) {
        const auto& ri = _particles[i].position;

        for (const auto& particle : skin_neighbors[i]) {
            int j = particle.id;
            const auto& rj = _particles[j].position;

            double dx = min_image(rj[0] - ri[0]);
            double dy = min_image(rj[1] - ri[1]);

            double r2 = dx*dx + dy*dy;

            if (r2 < rc2) {
                neighbors[i].push_back({j, { dx,  dy}, r2});
            }
        }
    }
}

double System::max_displacement() const{

    double max2 = 0.0;

    for (int i = 0; i < _numparticles; ++i) {
        double dx = min_image(_particles[i].position[0] - old_positions[i][0]);
        double dy = min_image(_particles[i].position[1] - old_positions[i][1]);

        double dr2 = dx*dx + dy*dy;
        if (dr2 > max2) max2 = dr2;
    }
    return std::sqrt(max2);
}



class evolver{

private:
    System& _system;
    std::mt19937 rng;
    std::normal_distribution<double> normal;


public:
    evolver(System& system): _system{system}, rng(1234), normal(0.0, 1.0) {} //constructor initialiser list

    void step_euler(){
        auto& particles = _system.get_particles();
        for(auto& p : particles){

            p.theta += std::sqrt(2.0 * Dr * dt) * normal(rng);

            // orientation vector
            p.n[0] = std::cos(p.theta);
            p.n[1] = std::sin(p.theta);
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            _system.apply_pbc(p.position);
        }
    }

    void update_neighbors(){
        if(_system.max_displacement() > _system.get_r_skin() / 2.0){
            _system.build_neighbor_lists_full();
        }
        else{
            _system.build_neighbor_lists_core();
        }
    }

    void step() {
        step_euler();       
        update_neighbors(); 
        // later:
        // compute_forces();
        // apply_forces();
    }

    

};


namespace py = pybind11;

PYBIND11_MODULE(abp_sim, m) {
    py::class_<Particle>(m, "Particle")
        .def_readwrite("position", &Particle::position)
        .def_readwrite("theta", &Particle::theta);

    py::class_<System>(m, "System")
        .def("get_particles", &System::get_particles);

    py::class_<evolver>(m, "Evolver")
        .def("step", &evolver::step);
}
