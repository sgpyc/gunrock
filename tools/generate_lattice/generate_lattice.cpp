#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <omp.h>

using namespace std;

int main()
{
    typedef uint32_t VertexT;
    typedef uint32_t SizeT;
    typedef uint32_t ValueT;
 
    vector<int> dimensions{11000, 11000};
    bool directed = true;
    bool warrped = false;
    bool has_edge_weights = true;
    ValueT edge_weight = 1;
    int num_threads = 16; 
    const vector<vector<int> > neighbor_displacements
        {{-1, -1}, {-1, 0}, {0, -1}, {0, 1}, {1, 0}, {1, 1}};
    
    int num_dimensions = dimensions.size();
    int num_neightbors = neighbor_displacements.size();
    vector<pair<VertexT, VertexT> > *thread_edges 
        = new vector<pair<VertexT, VertexT> >[num_threads];
    vector<ValueT> *thread_edge_weights = new vector<ValueT>[num_threads];

    SizeT num_vertices = 1;
    for (auto &dim : dimensions)
        num_vertices *= dim;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_num = omp_get_thread_num();
        auto &t_edges = thread_edges[thread_num];
        auto &t_edge_weights = thread_edge_weights[thread_num];

        VertexT start_v = num_vertices / num_threads * thread_num;
        VertexT end_v   = num_vertices / num_threads * (thread_num + 1);
        VertexT v = start_v;
        
        vector<int> v_coordinates, u_coordinates;
        v_coordinates.resize(num_dimensions);
        u_coordinates.resize(num_dimensions);
        for (auto i = num_dimensions -1; i>=0; i--)
        {
            v_coordinates[i] = v % dimensions[i];
            v = v / dimensions[i]; 
        }

        for (v = start_v; v < end_v; v++)
        {
            for (auto t = 0; t < num_neightbors; t++)
            {
                auto &displacement = neighbor_displacements[t];
                for (auto i = 0; i< num_dimensions; i ++)
                    u_coordinates[i] = v_coordinates[i] + displacement[i];
                VertexT u = 0;
                bool u_valid = true;
                for (auto i = 0; i< num_dimensions; i++)
                {
                    if (i != 0)
                        u *= dimensions[i-1];
                    if ((u_coordinates[i] < 0 || u_coordinates[i] >= dimensions[i]) &&
                        !warrped)
                    {
                        u_valid = false;
                        break;
                    }

                    if (u_coordinates[i] < 0)
                        u_coordinates[i] += dimensions[i];
                    if (u_coordinates[i] >= dimensions[i])
                        u_coordinates[i] -= dimensions[i];
                    u += u_coordinates[i];
                }

                if (!directed && v > u)
                    u_valid = false;
                if (!u_valid)
                    continue;

                t_edges.push_back(make_pair(v, u));
                if (has_edge_weights)
                    t_edge_weights.push_back(edge_weight);
            }
            //cout << thread_num << " : " << v << endl;

            int pos = num_dimensions - 1;
            v_coordinates[pos] ++;
            while (pos > 0 && v_coordinates[pos] == dimensions[pos])
            {
                v_coordinates[pos - 1] ++;
                v_coordinates[pos] = 0;
                pos --;
            } 
        }
    }

    SizeT num_edges = 0;
    for (auto t = 0; t < num_threads; t++)
        num_edges += thread_edges[t].size();

    string filename = "lattice_";
    for (unsigned int i = 0; i < dimensions.size(); i++)
    {
        if (i != 0)
            filename += "x";
        filename += to_string(dimensions[i]);
    } 
    filename += ".mtx";

    ofstream fout;
    fout.open(filename.c_str());
    if (!fout.is_open())
    {
        cerr << "Error: can't open " << filename << " for writing." << endl;
        return 1;
    }

    fout << "%%MatrixMarket matrix coordinate integer" << endl;
    fout << "% Generated lattice" << endl;
    fout << "% Dimensions = ";
    for (unsigned int i = 0; i < dimensions.size(); i++)
        fout << (i == 0 ? "" : " x ") << dimensions[i];
    fout << endl;
    fout << "% Directed = " << (directed ? "True" : "False") << endl;
    fout << "% Warrped = " << (warrped ? "True" : "False") << endl;
    fout << "% Neighbor_displacements = {";
    for (unsigned int i = 0; i < neighbor_displacements.size(); i++)
    {
        fout << (i == 0 ? "" : ", ") << "{";
        for (auto j = 0; j < num_dimensions; j++)
            fout << (j == 0 ? "" : ", ") << neighbor_displacements[i][j];
        fout << "}";
    }
    fout << endl;

    fout << num_vertices << " " << num_vertices << " " << num_edges << endl;
    for (auto t = 0; t < num_threads; t++)
    {
        auto &t_edges = thread_edges[t];
        auto &t_edge_weights = thread_edge_weights[t];
        for (SizeT i = 0; i < t_edges.size(); i++)
        {
            fout << t_edges[i].first + 1 << " " << t_edges[i].second + 1;
            if (has_edge_weights)
                fout << " " << t_edge_weights[i];
            fout << endl;
        }
        t_edges.clear();
        t_edge_weights.clear();
    }
    fout.close();
    return 0;
}

