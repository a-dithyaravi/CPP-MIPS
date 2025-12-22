#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std; 

double pi = 3.141592653589793;

int radius = 1;
double v0 = 1.0; // Self-propulsion speed
double dt = 0.01; // Time step
double k = 100.0; // mobility* Spring constant for repulsive force


struct Particle {
    int id;
    std::array<double, 2> position; 
    std::array<double, 2> position_unwrapped; //actual dist travelled without pbc
    std::array<double, 2> force; // (fx,fy)
    double theta;                   // orientation angle
    std::array<double, 2> n;        // orientation vector

};


struct Neighbor {
    int id;                    // index of neighbor
    std::array<double, 2> dr;  // {dx, dy}
    double r2;                 // squared distance
};

class System
{
public:

    System(int N, double L, double Pe) : _numparticles(N), L(L), Pe(Pe) {
        _particles.resize(N);
    } //constructor initialiser list

    std::vector<Particle>& get_particles() { 
        return _particles; 
    }
    int size() const { return _numparticles; }
    
    std::vector<std::vector<Neighbor>> neighbors;       // r < rc
    std::vector<std::vector<Neighbor>> skin_neighbors;  // r < rc + r_skin

    void build_neighbor_lists_full();
    void build_neighbor_lists_core();
    double max_displacement() const;
    
    void initialize_particles(
    const std::vector<std::array<double,2>>& positions,
    const std::vector<double>& thetas) 
    {

        for (int i = 0; i < _numparticles; ++i) {
            _particles[i].position_unwrapped = positions[i];
            _particles[i].position = positions[i];
            _particles[i].theta = thetas[i];

            _particles[i].n[0] = std::cos(thetas[i]);
            _particles[i].n[1] = std::sin(thetas[i]);

        }
    }


    inline void apply_pbc(std::array<double,2>& r) const { // apply periodic boundary conditions
        r[0] -= L * std::floor(r[0] / L);
        r[1] -= L * std::floor(r[1] / L);
    }

    inline double min_image(double dx) const{   //minimum image convention
        if (dx >  0.5 * L) dx -= L;
        if (dx < -0.5 * L) dx += L;
        return dx;
    }

    inline void set_cutoff(){ 
        r_c = 2*radius;
        r_skin = 0.5*radius; }

    inline double get_r_skin() const { return r_skin; }

    inline void set_Dr(){
        Dr = 1.0 / Pe; 
    }
    inline double get_Dr() const {
        return Dr;
    }


    bool neighbors_initialized = false;

private:
    std::vector<Particle> _particles;    // Standard library vector containing all particles
    double L;
    int _numparticles;                   // Total number of particles in the system
    double r_c; // cutoff radius
    double r_skin; // skin thickness
    std::vector<std::array<double,2>> old_positions; //updated when skin neighbors are rebuilt
    double Pe; // Peclet number
    double Dr; // rotational diffusion constant
    
};

void System::build_neighbor_lists_full()
{
    set_cutoff();
    const int N = size();

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
    set_cutoff();
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

    void compute_forces() {
        auto& particles = _system.get_particles();

        // reset forces 
        for (auto& p : particles) {
            p.force[0] = 0.0;
            p.force[1] = 0.0;
        }

        for (int i = 0; i < _system.size(); ++i) {
            for (const auto& nb : _system.neighbors[i]) {

                int j = nb.id;
                if (j <= i) continue;  // avoid double counting

                double dx = nb.dr[0];
                double dy = nb.dr[1];
                double r2 = nb.r2;

                if (r2 < (2.0 * radius) * (2.0 * radius)) {
                    double r = std::sqrt(r2) + 1e-12; // to avoid division by zero
                    double f = k * (2.0 * radius - r) / r;

                    particles[i].force[0] -= f * dx;
                    particles[i].force[1] -= f * dy;

                    particles[j].force[0] += f * dx;
                    particles[j].force[1] += f * dy;
                }
            }
        }
    }


    void step_euler(){
        compute_forces();
        auto& particles = _system.get_particles();
        _system.set_Dr();
        for(auto& p : particles){

            p.theta += std::sqrt(2.0 * _system.get_Dr()* dt) * normal(rng);

            // orientation vector
            p.n[0] = std::cos(p.theta);
            p.n[1] = std::sin(p.theta);

            p.position_unwrapped[0] += (v0*(p.n[0]) + p.force[0])* dt;
            p.position_unwrapped[1] += (v0*(p.n[1]) + p.force[1])* dt;
            p.position[0] += (v0*(p.n[0]) + p.force[0])* dt;
            p.position[1] += (v0*(p.n[1]) + p.force[1])* dt;
            _system.apply_pbc(p.position);
            
        }
    }

    void update_neighbors(){

        if (!_system.neighbors_initialized) {
            _system.build_neighbor_lists_full();
            _system.neighbors_initialized = true;
            return;
        
        }   
        if(_system.max_displacement() > _system.get_r_skin()/2.0){
            _system.build_neighbor_lists_full();
        }
        else{
            _system.build_neighbor_lists_core();
        }
    }

    void step() {
        update_neighbors();
        step_euler();       
    }

};


namespace py = pybind11;

PYBIND11_MODULE(abp_sim, m) {
    py::class_<Particle>(m, "Particle")
        .def_readwrite("position", &Particle::position)
        .def_readwrite("position_unwrapped", &Particle::position_unwrapped)
        .def_readwrite("theta", &Particle::theta);

    py::class_<System>(m, "System")
    .def(py::init<int, double>())
    .def("initialize_particles", &System::initialize_particles)
    .def("get_particles", &System::get_particles,
         py::return_value_policy::reference_internal);

    py::class_<evolver>(m, "Evolver")
    .def(py::init<System&>())
    .def("step", &evolver::step);

}
