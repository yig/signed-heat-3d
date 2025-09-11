#include "signed_heat_tet_solver.h"

#include "polyscope/volume_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include <unordered_map>

SignedHeatTetSolver::SignedHeatTetSolver() {}

Eigen::MatrixXd SignedHeatTetSolver::getVertices() const {
    return vertices;
}

Eigen::MatrixXi SignedHeatTetSolver::getTets() const {
    return tets;
}

namespace {
    Vector3 eigenToGC(const Eigen::Vector3d& eigenVec) {
        return Vector3{eigenVec(0), eigenVec(1), eigenVec(2)};
    }
}


// =============== ALGORITHM

Vector<double> SignedHeatTetSolver::computeDistance(VertexPositionGeometry& geometry,
                                                    const SignedHeat3DOptions& options) {

    bool isConforming = false;
    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building tet mesh..." << std::endl;
        double meanFaceArea = 0.;
        SurfaceMesh& mesh = geometry.mesh;
        setFaceVectorAreas(geometry, surfaceFaceAreas, surfaceFaceNormals);
        for (Face f : mesh.faces()) meanFaceArea += surfaceFaceAreas[f];
        meanFaceArea /= mesh.nFaces();
        double length = options.resolution[0] > 0 ? 1. / options.resolution[0] : 1. / 32;
        length *= 16;
        double maxVolume = length * length * length / (6. * std::sqrt(2.));
        TETFLAGS = TET_PREFIX + std::to_string(maxVolume);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(maxVolume) + "Y";
        Vector3 bboxMin, bboxMax;
        if (!isBoundingBoxValid(options.bboxMin, options.bboxMax)) {
            std::tie(bboxMin, bboxMax) = computeBBox(geometry);
        } else {
            bboxMin = options.bboxMin;
            bboxMax = options.bboxMax;
        }
        if (mesh.isTriangular()) isConforming = tetmeshDomain(geometry, bboxMin, bboxMax);
        if (!isConforming) {
            size_t nPts = mesh.nVertices();
            cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
            pointcloud::PointData<Vector3> pointPositions = pointcloud::PointData<Vector3>(*cloud);
            for (size_t i = 0; i < nPts; i++) pointPositions[i] = geometry.vertexPositions[i];
            pointPolyGeom = std::unique_ptr<pointcloud::PointPositionGeometry>(
                new pointcloud::PointPositionGeometry(*cloud, pointPositions));
            tetmeshPointCloud(*pointPolyGeom, bboxMin, bboxMax);
        }
        // With direct convolution in R^n, it's not clear what we should pick as our timestep. Just use the
        // tetmesh/trimesh as a proxy.
        if (VERBOSE) std::cerr << "Computing tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = dualLaplacian();
        if (VERBOSE) std::cerr << "Tet mesh (re)built" << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
    Eigen::MatrixXd Yt = Eigen::MatrixXd::Zero(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    SurfaceMesh& mesh = geometry.mesh;
    size_t F = mesh.nFaces();
    // Integrate contributions (single-point quadrature)
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions (single-point quadrature)
        Vector3 X = {0, 0, 0};
        for (Face f : mesh.faces()) {
            Vector3 p = {0, 0, 0};
            for (Vertex v : f.adjacentVertices()) p += geometry.vertexPositions[v];
            p /= f.degree();
            Vector3 n = surfaceFaceNormals[f];
            X += yukawaPotential(p, q, lambda) * n * surfaceFaceAreas[f];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    Vector<double> phi;
    if (isConforming) {
        phi = options.fastIntegration ? integrateVectorFieldGreedily(geometry, Yt, options)
                                      : integrateVectorField(geometry, Yt, options);
    } else {
        pointPolyGeom->requireTuftedTriangulation();
        pointPolyGeom->tuftedGeom->requireVertexDualAreas();
        phi = options.fastIntegration ? integrateVectorFieldGreedily(*pointPolyGeom, Yt, options)
                                      : integrateVectorField(*pointPolyGeom, Yt, options);
        pointPolyGeom->unrequireTuftedTriangulation();
        pointPolyGeom->tuftedGeom->unrequireVertexDualAreas();
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    return phi;
}

Vector<double> SignedHeatTetSolver::computeDistance(pointcloud::PointPositionNormalGeometry& pointGeom,
                                                    const SignedHeat3DOptions& options) {

    pointGeom.requireTuftedTriangulation();
    pointGeom.tuftedGeom->requireVertexDualAreas();

    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building tet mesh..." << std::endl;
        double meanArea = 0.;
        for (size_t i = 0; i < pointGeom.cloud.nPoints(); i++) meanArea += pointGeom.tuftedGeom->vertexDualAreas[i];
        meanArea /= pointGeom.cloud.nPoints();
        double length = options.resolution[0] > 0 ? 1. / options.resolution[0] : 1. / 32;
        length *= 16;
        double maxVolume = length * length * length / (6. * std::sqrt(2.));
        TETFLAGS = TET_PREFIX + std::to_string(maxVolume);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(maxVolume) + "Y";
        Vector3 bboxMin, bboxMax;
        if (!isBoundingBoxValid(options.bboxMin, options.bboxMax)) {
            std::tie(bboxMin, bboxMax) = computeBBox(pointGeom);
        } else {
            bboxMin = options.bboxMin;
            bboxMax = options.bboxMax;
        }
        tetmeshPointCloud(pointGeom, bboxMin, bboxMax);
        // With direct convolution in R^n, it's not clear what we should pick as our timestep. Just use the
        // tetmesh/trimesh as a proxy.
        if (VERBOSE) std::cerr << "Computing tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        if (VERBOSE) std::cerr << "Building Laplacian..." << std::endl;
        laplaceMat = dualLaplacian();
        if (VERBOSE) std::cerr << "Tet mesh (re)built" << std::endl;
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Pre-compute time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;

    // Evaluate vectors at tet barycenters.
    size_t P = pointGeom.cloud.nPoints();
    Eigen::MatrixXd Yt(nTets, 3);
    double lambda = std::sqrt(1. / shortTime);
    for (size_t i = 0; i < nTets; i++) {
        // Compute query point.
        Vector3 q = {0, 0, 0};
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
        }
        q /= 4.;
        // Integrate contributions.
        Vector3 X = {0, 0, 0};
        for (size_t pIdx = 0; pIdx < P; pIdx++) {
            Vector3 p = pointGeom.positions[pIdx];
            Vector3 n = pointGeom.normals[pIdx];
            X += yukawaPotential(p, q, lambda) * n * pointGeom.tuftedGeom->vertexDualAreas[pIdx];
        }
        X /= X.norm();
        for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
    }
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    if (VERBOSE) std::cerr << "Step 3..." << std::endl;
    Vector<double> phi = options.fastIntegration ? integrateVectorFieldGreedily(pointGeom, Yt, options)
                                                 : integrateVectorField(pointGeom, Yt, options);
    if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

    pointGeom.tuftedGeom->unrequireVertexDualAreas();
    pointGeom.unrequireTuftedTriangulation();

    return phi;
}


Vector<double> SignedHeatTetSolver::integrateVectorField(VertexPositionGeometry& geometry, const Eigen::MatrixXd& Yt,
                                                         const SignedHeat3DOptions& options) {

    if (options.useCrouzeixRaviart) return integrateVectorFieldToFaces(geometry, Yt, options);

    SurfaceMesh& mesh = geometry.mesh;
    Vector<double> div = vertexDivergence(Yt);
    Vector<double> phi;
    if (options.levelSetConstraint == LevelSetConstraint::ZeroSet) {
        // Since the tet mesh conforms to the surface, preserving zero can be done via Dirichlet boundary conditions.
        Vector<bool> setAMembership = Vector<bool>::Ones(nVertices);
        for (size_t i = 0; i < mesh.nVertices(); i++) setAMembership[i] = false;
        int nB = nVertices - setAMembership.cast<int>().sum();
        Vector<double> bcVals = Vector<double>::Zero(nB);
        BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceMat, setAMembership, true);
        Vector<double> rhsValsA, rhsValsB;
        decomposeVector(decomp, div, rhsValsA, rhsValsB);
        Vector<double> combinedRHS = rhsValsA;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> Aresult = AMGCL_solve(decomp.AA, combinedRHS, VERBOSE);
        #else
        Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
        #endif
        // clang-format on
        phi = reassembleVector(decomp, Aresult, bcVals);
    } else if (options.levelSetConstraint == LevelSetConstraint::Multiple) {
        // Determine the connected components of the mesh. Do simple depth-first search.
        std::vector<Eigen::Triplet<double>> triplets;
        SparseMatrix<double> A;
        size_t m = 0;
        size_t V = mesh.nVertices();
        VertexData<bool> marked(mesh, false);
        geometry.requireVertexIndices();
        for (Vertex v : mesh.vertices()) {
            if (marked[v]) continue;
            marked[v] = true;
            std::vector<Vertex> queue = {v};
            size_t v0 = geometry.vertexIndices[v];
            Vertex curr;
            while (!queue.empty()) {
                curr = queue.back();
                queue.pop_back();
                for (Vertex w : curr.adjacentVertices()) {
                    if (marked[w]) continue;
                    triplets.emplace_back(m, geometry.vertexIndices[w], -1);
                    triplets.emplace_back(m, v0, 1);
                    marked[w] = true;
                    queue.push_back(w);
                    m++;
                }
            }
        }
        geometry.unrequireVertexIndices();
        A.resize(m, nVertices);
        A.setFromTriplets(triplets.begin(), triplets.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(nVertices + m);
        RHS.head(nVertices) = div;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> soln = AMGCL_solve(LHS, RHS, VERBOSE);
        #else 
        Vector<double> soln = solveSquare(LHS, RHS);
        #endif
        // clang-format on
        phi = soln.head(nVertices);
        double shift = averageVertexDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nVertices);
    } else {
        if (options.rebuild || poissonSolver == nullptr) {
            if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
            poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
        }
        phi = poissonSolver->solve(div);
        double shift = averageVertexDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nVertices);
    }

    return phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorFieldToFaces(VertexPositionGeometry& geometry,
                                                                const Eigen::MatrixXd& Yt,
                                                                const SignedHeat3DOptions& options) {

    geometry.requireFaceIndices();

    SurfaceMesh& mesh = geometry.mesh;
    Vector<double> div = faceDivergence(Yt);
    Vector<double> phi;
    laplaceCR = buildCrouzeixRaviartLaplacian();
    if (options.levelSetConstraint == LevelSetConstraint::ZeroSet) {
        // Since the tet mesh conforms to the surface, preserving zero can be done via Dirichlet boundary conditions.
        Vector<bool> setAMembership = Vector<bool>::Ones(nFaces);
        for (const int& fIdx : surfaceFaces) setAMembership[abs(fIdx)] = false;
        int nB = nFaces - setAMembership.cast<int>().sum();
        Vector<double> bcVals = Vector<double>::Zero(nB);
        BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceCR, setAMembership, true);
        Vector<double> rhsValsA, rhsValsB;
        decomposeVector(decomp, div, rhsValsA, rhsValsB);
        Vector<double> combinedRHS = rhsValsA;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> Aresult = AMGCL_solve(decomp.AA, combinedRHS, VERBOSE);
        #else
        Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
        #endif
        // clang-format on
        phi = reassembleVector(decomp, Aresult, bcVals);
    } else if (options.levelSetConstraint == LevelSetConstraint::Multiple) {
        // Determine the connected components of the mesh. Do simple depth-first search.
        std::vector<Eigen::Triplet<double>> triplets;
        SparseMatrix<double> A;
        size_t m = 0;
        size_t F = mesh.nFaces();
        FaceData<bool> marked(mesh, false);
        geometry.requireFaceIndices();
        for (Face f : mesh.faces()) {
            if (marked[f]) continue;
            marked[f] = true;
            std::vector<Face> queue = {f};
            size_t f0 = geometry.faceIndices[f];
            Face curr;
            while (!queue.empty()) {
                curr = queue.back();
                queue.pop_back();
                for (Face g : curr.adjacentFaces()) {
                    if (marked[g]) continue;
                    triplets.emplace_back(m, geometry.faceIndices[g], -1);
                    triplets.emplace_back(m, f0, 1);
                    marked[g] = true;
                    queue.push_back(g);
                    m++;
                }
            }
        }
        geometry.unrequireFaceIndices();
        A.resize(m, nFaces);
        A.setFromTriplets(triplets.begin(), triplets.end());
        SparseMatrix<double> Z(m, m);
        SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceCR, A.transpose()});
        SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
        SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
        Vector<double> RHS = Vector<double>::Zero(nFaces + m);
        RHS.head(nFaces) = div;
        // clang-format off
        #ifndef SHM_NO_AMGCL
        Vector<double> soln = AMGCL_solve(LHS, RHS, VERBOSE);
        #else
        Vector<double> soln = solveSquare(LHS, RHS);
        #endif
        // clang-format on
        phi = soln.head(nFaces);
        double shift = averageFaceDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nFaces);
    } else {
        if (options.rebuild || poissonSolverCR == nullptr) {
            if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
            poissonSolverCR.reset(new PositiveDefiniteSolver<double>(laplaceCR));
        }
        phi = poissonSolverCR->solve(div);
        double shift = averageFaceDataOnSource(geometry, phi);
        phi -= shift * Vector<double>::Ones(nFaces);
    }

    if (options.rebuild || projectionSolver == nullptr) {
        massMat = buildCrouzeixRaviartMassMatrix();
        avgMat = buildAveragingMatrix();
        SparseMatrix<double> P = avgMat.transpose() * massMat * avgMat;
        projectionSolver.reset(new SquareSolver<double>(P));
    }
    phi = projectOntoVertices(phi);

    geometry.unrequireFaceIndices();

    return -phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorField(pointcloud::PointPositionGeometry& pointGeom,
                                                         const Eigen::MatrixXd& Yt,
                                                         const SignedHeat3DOptions& options) {

    Vector<double> phi;
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            if (options.rebuild || poissonSolver == nullptr) {
                if (VERBOSE) std::cerr << "\tFactorizing..." << std::endl;
                poissonSolver.reset(new PositiveDefiniteSolver<double>(laplaceMat));
            }
            Vector<double> div = vertexDivergence(Yt);
            phi = poissonSolver->solve(div);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            Vector<double> div = vertexDivergence(Yt);
            size_t P = pointGeom.cloud.nPoints();
            Vector<bool> setAMembership = Vector<bool>::Ones(nVertices);
            for (size_t i = 0; i < P; i++) setAMembership[i] = false;
            int nB = nVertices - setAMembership.cast<int>().sum();
            Vector<double> bcVals = Vector<double>::Zero(nB);
            BlockDecompositionResult<double> decomp = blockDecomposeSquare(laplaceMat, setAMembership, true);
            Vector<double> rhsValsA, rhsValsB;
            decomposeVector(decomp, div, rhsValsA, rhsValsB);
            Vector<double> combinedRHS = rhsValsA;
            // clang-format off
            #ifndef SHM_NO_AMGCL
            Vector<double> Aresult = AMGCL_solve(decomp.AA, combinedRHS, VERBOSE);
            #else
            Vector<double> Aresult = solvePositiveDefinite(decomp.AA, combinedRHS);
            #endif
            // clang-format on
            phi = reassembleVector(decomp, Aresult, bcVals);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            Vector<double> div = vertexDivergence(Yt);
            std::vector<Eigen::Triplet<double>> triplets;
            SparseMatrix<double> A;
            size_t m = 0;
            size_t P = pointGeom.cloud.nPoints();
            VertexData<bool> marked(pointGeom.tuftedGeom->mesh, Vector<bool>::Zero(P));
            pointGeom.tuftedGeom->requireVertexIndices();
            for (Vertex v : pointGeom.tuftedGeom->mesh.vertices()) {
                if (marked[v]) continue;
                marked[v] = true;
                std::vector<Vertex> queue = {v};
                size_t v0 = pointGeom.tuftedGeom->vertexIndices[v];
                Vertex curr;
                while (!queue.empty()) {
                    curr = queue.back();
                    queue.pop_back();
                    for (Vertex w : curr.adjacentVertices()) {
                        if (marked[w]) continue;
                        triplets.emplace_back(m, pointGeom.tuftedGeom->vertexIndices[w], -1);
                        triplets.emplace_back(m, v0, 1);
                        marked[w] = true;
                        queue.push_back(w);
                        m++;
                    }
                }
            }
            pointGeom.tuftedGeom->unrequireVertexIndices();
            A.resize(m, nVertices);
            A.setFromTriplets(triplets.begin(), triplets.end());
            SparseMatrix<double> Z(m, m);
            SparseMatrix<double> LHS1 = horizontalStack<double>({laplaceMat, A.transpose()});
            SparseMatrix<double> LHS2 = horizontalStack<double>({A, Z});
            SparseMatrix<double> LHS = verticalStack<double>({LHS1, LHS2});
            Vector<double> RHS = Vector<double>::Zero(nVertices + m);
            RHS.head(nVertices) = div;
            // clang-format off
            #ifndef SHM_NO_AMGCL
            Vector<double> soln = AMGCL_solve(LHS, RHS, VERBOSE);
            #else
            Vector<double> soln = solveSquare(LHS, RHS);
            #endif
            #// clang-format on
            phi = soln.head(nVertices);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
    }
    return phi;
}

/* Integrate using breadth-first search. */
Vector<double> SignedHeatTetSolver::integrateVectorFieldGreedily(VertexPositionGeometry& geometry,
                                                                 const Eigen::MatrixXd& Yt,
                                                                 const SignedHeat3DOptions& options) {

    Vector<double> phi(nVertices);
    SurfaceMesh& mesh = geometry.mesh;
    size_t V = mesh.nVertices();
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            phi[0] = 0;
            visited[0] = true;
            integrateGreedily(Yt, visited, phi);
            double shift = averageVertexDataOnSource(geometry, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            // Fix solution values on source geometry.
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            for (size_t i = 0; i < V; i++) {
                phi[i] = 0;
                visited[i] = true;
            }
            integrateGreedily(Yt, visited, phi);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            phi = integrateGreedilyMultipleLevelSets(geometry, Yt);
            break;
        }
    }
    return phi;
}

Vector<double> SignedHeatTetSolver::integrateVectorFieldGreedily(pointcloud::PointPositionGeometry& pointGeom,
                                                                 const Eigen::MatrixXd& Yt,
                                                                 const SignedHeat3DOptions& options) {

    Vector<double> phi(nVertices);
    size_t P = pointGeom.cloud.nPoints();
    switch (options.levelSetConstraint) {
        case (LevelSetConstraint::None): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            phi[0] = 0;
            visited[0] = true;
            integrateGreedily(Yt, visited, phi);
            double shift = averageVertexDataOnSource(pointGeom, phi);
            phi -= shift * Vector<double>::Ones(nVertices);
            break;
        }
        case (LevelSetConstraint::ZeroSet): {
            Vector<bool> visited = Vector<bool>::Zero(nVertices);
            for (size_t i = 0; i < P; i++) {
                phi[i] = 0;
                visited[i] = true;
            }
            integrateGreedily(Yt, visited, phi);
            break;
        }
        case (LevelSetConstraint::Multiple): {
            phi = integrateGreedilyMultipleLevelSets(*(pointGeom.tuftedGeom), Yt);
            break;
        }
    }
    return phi;
}

void SignedHeatTetSolver::integrateGreedily(const Eigen::MatrixXd& Yt, Vector<bool>& visited,
                                            Vector<double>& phi) const {

    // Start queue with one of the surface vertices; we're assuming that the tetmesh domain is connected.
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        Eigen::Vector3d p = vertices.row(curr);
        queue.pop();
        for (size_t tIdx : vertexTet[curr]) {
            for (int j = 0; j < 4; j++) {
                size_t neighbor = tets(tIdx, j);
                if (visited[neighbor]) continue;
                Eigen::Vector3d q = vertices.row(neighbor);
                Eigen::Vector3d edge = q - p;
                Eigen::Vector3d Y = Yt.row(tIdx);
                phi[neighbor] = phi[curr] + Y.dot(edge);
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
}

Vector<double> SignedHeatTetSolver::integrateGreedilyMultipleLevelSets(IntrinsicGeometryInterface& geometry,
                                                                       const Eigen::MatrixXd& Yt) const {

    // Determine mesh components.
    SurfaceMesh& mesh = geometry.mesh;
    geometry.requireVertexIndices();
    std::vector<int> meshComponent(mesh.nVertices(), -1);
    Vector<bool> visited = Vector<bool>::Zero(nVertices);
    Vector<double> phi(nVertices);
    size_t cptIdx = 0;
    for (Vertex v : mesh.vertices()) {
        size_t vIdx = geometry.vertexIndices[v];
        if (meshComponent[vIdx] != -1) continue;
        meshComponent[vIdx] = cptIdx;
        std::vector<Vertex> queue = {v};
        if (cptIdx == 0) phi[vIdx] = 0;
        while (!queue.empty()) {
            Vertex curr = queue.back();
            queue.pop_back();
            for (Vertex w : curr.adjacentVertices()) {
                size_t wIdx = geometry.vertexIndices[w];
                if (meshComponent[wIdx] != -1) continue;
                meshComponent[wIdx] = cptIdx;
                if (cptIdx == 0) phi[wIdx] = 0;
                queue.push_back(w);
            }
        }
        cptIdx++;
    }
    geometry.unrequireVertexIndices();

    // integrate
    size_t V = mesh.nVertices();
    std::vector<bool> componentVisited(cptIdx, false);
    std::vector<double> componentValue(cptIdx);
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
        size_t curr = queue.front();
        Eigen::Vector3d p = vertices.row(curr);
        queue.pop();
        for (size_t tIdx : vertexTet[curr]) {
            for (int j = 0; j < 4; j++) {
                size_t neighbor = tets(tIdx, j);
                if (visited[neighbor]) continue;
                if ((neighbor < V) && componentVisited[meshComponent[neighbor]]) {
                    phi[neighbor] = componentValue[meshComponent[neighbor]];
                } else {
                    Eigen::Vector3d q = vertices.row(neighbor);
                    Eigen::Vector3d edge = q - p;
                    Eigen::Vector3d Y = Yt.row(tIdx);
                    phi[neighbor] = phi[curr] + Y.dot(edge);
                    if (neighbor < V) {
                        componentVisited[meshComponent[neighbor]] = true;
                        componentValue[meshComponent[neighbor]] = phi[neighbor];
                    }
                }
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
    return phi;
}

double SignedHeatTetSolver::averageFaceDataOnSource(VertexPositionGeometry& geometry, const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0.;
    for (const auto& fIdx : surfaceFaces) {
        size_t i = abs(fIdx);
        Eigen::Vector3d a = vertices.row(faces(i, 0));
        Eigen::Vector3d b = vertices.row(faces(i, 1));
        Eigen::Vector3d c = vertices.row(faces(i, 2));
        double A = 0.5 * ((a - c).cross(b - c)).norm();
        shift += A * phi[i];
        totalArea += A;
    }
    shift /= totalArea;
    return shift;
}

double SignedHeatTetSolver::averageVertexDataOnSource(VertexPositionGeometry& geometry,
                                                      const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0.;
    geometry.requireVertexDualAreas();
    for (size_t i = 0; i < geometry.mesh.nVertices(); i++) {
        double A = geometry.vertexDualAreas[i];
        shift += A * phi[i];
        totalArea += A;
    }
    shift /= totalArea;
    geometry.unrequireVertexDualAreas();
    return shift;
}

double SignedHeatTetSolver::averageVertexDataOnSource(pointcloud::PointPositionGeometry& pointGeom,
                                                      const Vector<double>& phi) const {

    double shift = 0.;
    double totalArea = 0;
    size_t P = pointGeom.cloud.nPoints();
    for (size_t pIdx = 0; pIdx < P; pIdx++) {
        double A = pointGeom.tuftedGeom->vertexDualAreas[pIdx];
        shift += A * phi[pIdx];
        totalArea += A;
    }
    shift /= totalArea;
    return shift;
}

/*
 * Given a piecewise-constant vector field defined on tets, compute FEM integrated divergence per face.
 */
Vector<double> SignedHeatTetSolver::faceDivergence(const Eigen::MatrixXd& X) const {

    Vector<double> divX = Vector<double>::Zero(nFaces);
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            int sfIdx = tetFace(i, j);
            int fIdx = abs(sfIdx);
            Eigen::Vector3d N = areaWeightedNormalVector(sfIdx);
            divX[fIdx] += N.dot(X.row(i));
        }
    }
    return divX;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartLaplacian() const {

    SparseMatrix<double> L(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        for (int j = 0; j < 4; j++) {
            int sfA = tetFace(i, j);
            int fA = abs(sfA);
            Eigen::Vector3d nA = areaWeightedNormalVector(sfA);
            for (int k = j + 1; k < 4; k++) {
                int sfB = tetFace(i, k);
                int fB = abs(sfB);
                Eigen::Vector3d nB = areaWeightedNormalVector(sfB);
                double w = (nA.dot(nB)) / vol;
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
                triplets.emplace_back(fA, fA, -w);
                triplets.emplace_back(fB, fB, -w);
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    return L;
}

SparseMatrix<double> SignedHeatTetSolver::buildCrouzeixRaviartMassMatrix() const {

    SparseMatrix<double> M(nFaces, nFaces);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < nTets; i++) {
        double vol = computeTetVolume(i);
        // Iterate over all pairs of adjacent faces.
        double w = -0.05 * vol;
        for (int j = 0; j < 4; j++) {
            int fA = abs(tetFace(i, j));
            for (int k = j + 1; k < 4; k++) {
                int fB = abs(tetFace(i, k));
                triplets.emplace_back(fA, fB, w);
                triplets.emplace_back(fB, fA, w);
            }
            triplets.emplace_back(fA, fA, 0.4 * vol);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

/*
 * Compute the circumcenter of a tetrahedron, given its vertex positions.
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
void tetCircumcenter(const Eigen::Matrix<double, 4, 3>& t, Eigen::Vector3d& c) {

    Eigen::Matrix3d A;
    Eigen::Vector3d b;

    const double n0 = t.row(0).squaredNorm();

    for (int k = 0; k < 3; ++k) {
        A.row(k) = t.row(k + 1) - t.row(0);
        b(k) = t.row(k + 1).squaredNorm() - n0;
    }

    c = 0.5 * A.fullPivHouseholderQr().solve(b);
}

/*
 * Compute the circumcenter of a face, given its vertex positions.
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
void faceCircumcenter(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                      Eigen::Vector3d& cc) {

    const double l[3]{(b - c).squaredNorm(), (a - c).squaredNorm(), (a - b).squaredNorm()};

    const double ba[3]{l[0] * (l[1] + l[2] - l[0]), l[1] * (l[2] + l[0] - l[1]), l[2] * (l[0] + l[1] - l[2])};
    const double sum = ba[0] + ba[1] + ba[2];

    cc = (ba[0] / sum) * a + (ba[1] / sum) * b + (ba[2] / sum) * c;
}

/*
 * Build the dual Laplacian for the tet mesh from Alexa et al. 2020 (https://igl.ethz.ch/projects/LB3D/LB3D.pdf).
 * Code from [https://igl.ethz.ch/projects/LB3D/dualLaplace.cpp]
 */
SparseMatrix<double> SignedHeatTetSolver::dualLaplacian() const {

    SparseMatrix<double> L(nVertices, nVertices);

    const int turn[4][4]{{-1, 2, 3, 1}, {3, -1, 0, 2}, {1, 3, -1, 0}, {2, 0, 1, -1}};

    auto getTet = [&](const int i, Eigen::Matrix<double, 4, 3>& t) {
        for (int k = 0; k < 4; ++k) {
            t.row(k) = vertices.row(tets(i, k));
        }
    };

    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Vector3d cc;
    Eigen::Matrix<double, 4, 3> t;

    for (size_t k = 0; k < nTets; k++) {
        // Compute the circumcenter of the tet.
        getTet(k, t);
        tetCircumcenter(t, cc);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    Eigen::Vector3d cf;
                    faceCircumcenter(t.row(i), t.row(j), t.row(turn[i][j]), cf);

                    const Eigen::Vector3d ce = 0.5 * (t.row(i) + t.row(j));

                    const double vol = tetVolume(t.row(i), ce, cf, cc);
                    const double wij = 6. * vol / (t.row(i) - t.row(j)).squaredNorm();

                    triplets.emplace_back(tets(k, i), tets(k, j), wij);
                    triplets.emplace_back(tets(k, j), tets(k, i), wij);
                    triplets.emplace_back(tets(k, i), tets(k, i), -wij);
                    triplets.emplace_back(tets(k, j), tets(k, j), -wij);
                }
            }
        }
    }
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

Vector<double> SignedHeatTetSolver::vertexDivergence(const Eigen::MatrixXd& X) const {

    const int turn[4][4]{{-1, 2, 3, 1}, {3, -1, 0, 2}, {1, 3, -1, 0}, {2, 0, 1, -1}};
    auto getTet = [&](const int i, Eigen::Matrix<double, 4, 3>& t) {
        for (int k = 0; k < 4; ++k) {
            t.row(k) = vertices.row(tets(i, k));
        }
    };
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Vector3d cc;
    Eigen::Matrix<double, 4, 3> t;
    Vector<double> div = Vector<double>::Zero(nVertices);
    for (size_t k = 0; k < nTets; k++) {
        getTet(k, t);
        tetCircumcenter(t, cc);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    Eigen::Vector3d cf;
                    faceCircumcenter(t.row(i), t.row(j), t.row(turn[i][j]), cf);
                    int vA = tets(k, i);
                    int vB = tets(k, j);
                    Eigen::Vector3d a = vertices.row(vA);
                    Eigen::Vector3d b = vertices.row(vB);
                    Eigen::Vector3d e = b - a;
                    const Eigen::Vector3d ce = 0.5 * (t.row(i) + t.row(j));
                    const double vol = tetVolume(t.row(i), ce, cf, cc);
                    const double wij = 6. * vol / (t.row(i) - t.row(j)).squaredNorm();
                    div[vA] += e.dot(X.row(k)) * wij;
                    div[vB] -= e.dot(X.row(k)) * wij;
                }
            }
        }
    }
    return div;
}

Vector<double> SignedHeatTetSolver::projectOntoVertices(const Vector<double>& u) const {

    SparseMatrix<double> At = avgMat.transpose();
    Vector<double> RHS = At * massMat * u;
    Vector<double> w = projectionSolver->solve(RHS);
    return w;
}

SparseMatrix<double> SignedHeatTetSolver::buildAveragingMatrix() const {

    SparseMatrix<double> A(nFaces, nVertices);
    std::vector<Eigen::Triplet<double>> triplets;
    double w = 1. / 3.;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            triplets.emplace_back(i, faces(i, j), w);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

void SignedHeatTetSolver::isosurface(std::unique_ptr<SurfaceMesh>& isoMesh,
                                     std::unique_ptr<VertexPositionGeometry>& isoGeom, const Vector<double>& phi,
                                     double isoval, double isoval_eps) const {

    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    Eigen::VectorXi J;
    Eigen::SparseMatrix<double> BC;
    std::cout << "ygl::marching_tets( " << isoval << ", " << isoval_eps << " )\n";
    ygl::marching_tets(vertices, tets, phi, isoval, isoval_eps, SV, SF, J, BC);
//    clean_zero_edges_simple(SV, SF);
    std::tie(isoMesh, isoGeom) = makeSurfaceMeshAndGeometry(SV, SF);
}





void SignedHeatTetSolver::clean_zero_edges_simple(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const {
    double epsilon = 1e-8;
    bool has_zero_edges = true;
    
    while (has_zero_edges) {
        has_zero_edges = false;
        
        // 正确的变量声明
        Eigen::MatrixXi uE;
        Eigen::VectorXi EMAP;
        Eigen::MatrixXi EF, EI;
        
        // 正确的 edge_flaps 调用
        igl::edge_flaps(F, uE, EMAP, EF, EI);
        
        for (int i = 0; i < uE.rows(); i++) {
            double len = (V.row(uE(i,0)) - V.row(uE(i,1))).norm();
            
            if (len < epsilon) {
                Eigen::RowVectorXd pos = V.row(uE(i,0));
                
                // 尝试 collapse_edge
                if (igl::collapse_edge(i, pos, V, F, uE, EMAP, EF, EI)) {
                    has_zero_edges = true;
                    break;
                }
            }
        }
    }
}



// =============== TET UTILITIES

Eigen::VectorXd SignedHeatTetSolver::computeTetVolumes() const {

    Eigen::VectorXd volumes(nTets);
    for (size_t i = 0; i < nTets; i++) {
        volumes(i) = computeTetVolume(i);
    }
    return volumes;
}

double SignedHeatTetSolver::computeTetVolume(size_t tIdx) const {

    return tetVolume(vertices.row(tets(tIdx, 0)), vertices.row(tets(tIdx, 1)), vertices.row(tets(tIdx, 2)),
                     vertices.row(tets(tIdx, 3)));
}

double SignedHeatTetSolver::tetVolume(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                                      const Eigen::Vector3d& d) const {
    Eigen::Matrix3d A;
    A.col(0) = b - a;
    A.col(1) = c - a;
    A.col(2) = d - a;
    return A.determinant() / 6.;
}

/*
 * Return the area-weighted normal vector of the face with index abs(fIdx). The sign of `fIdx` gives the orientation of
 * the face relative to its (arbitrary but fixed) global orientation.
 */
Eigen::Vector3d SignedHeatTetSolver::areaWeightedNormalVector(int fIdx) const {

    int idx = abs(fIdx);
    Eigen::Vector3d a = vertices.row(faces(idx, 0));
    Eigen::Vector3d b = vertices.row(faces(idx, 1));
    Eigen::Vector3d c = vertices.row(faces(idx, 2));
    Eigen::Vector3d n = 0.5 * (a - c).cross(b - c);
    if (fIdx < 0) n *= -1;
    return n;
}

Eigen::Vector3d SignedHeatTetSolver::faceBarycenter(size_t fIdx) const {

    return (vertices.row(faces(fIdx, 0)) + vertices.row(faces(fIdx, 1)) + vertices.row(faces(fIdx, 2))) / 3.;
}

// =============== TET-MESHING

/*
 * Tetmesh the interior and exterior of the given surface inside a bounding box, s.t. the vertices of the surface are
 * preserved.
 *
 * TetGen allows you to tetmesh while preserving the input faces; this allows us to construct a correspondence between
 * vertices in the original surface, and vertices in the tetmesh. However, there's no way to preserve only some faces
 * and not others. This is a problem if we want to generate a tetmesh within a particular bounding cube. (Without
 * specifying a bounding box, TetGen will just tetmesh a convex hull.) The faces of the cube are incredibly large,
 * leading to a terribly coarse tetrahedralization. So first we triangulate the surface of the bounding cube. Then we
 * generate a tetmesh, with the faces of the bounding cube and the surface constrained, with the command that they
 * should all be preserved. However, the faces of the bounding cube should be sufficiently refined from the first step
 * that the resulting tets are small enough and of similar size to the ones everywhere else.
 */
bool SignedHeatTetSolver::tetmeshDomain(VertexPositionGeometry& geometry, const Vector3& bboxMin,
                                        const Vector3 bboxMax) {

    SurfaceMesh& mesh = geometry.mesh;

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    triangulateCube(cubeSurface, bboxMin, bboxMax);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    // Create a constrained tetmesh of the surface, without changing any of the input faces itself.
    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    in.firstnumber = 0;
    in.numberofpoints = mesh.nVertices() + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < mesh.nVertices(); i++) {
        Vector3 pos = geometry.inputVertexPositions[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[mesh.nVertices() + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * mesh.nVertices() + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = mesh.nFaces() + cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy faces from input surface mesh.
    geometry.requireVertexIndices();
    for (size_t i = 0; i < mesh.nFaces(); i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        int j = 0;
        for (Vertex v : mesh.face(i).adjacentVertices()) {
            p->vertexlist[j] = geometry.vertexIndices[v];
            in.trifacelist[3 * i + j] = geometry.vertexIndices[v];
            j++;
        }
    }
    geometry.unrequireVertexIndices();
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[mesh.nFaces() + i] = 0;
        f = &in.facetlist[mesh.nFaces() + i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * mesh.nFaces() + 3 * i + j] = mesh.nVertices() + cubeSurface.trifacelist[3 * i + j];
        }
    }

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
        return false;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        return false;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
        return false;
    }
    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);

    // Determine the face ids in the tetmesh corresponding to the original input surface.
    // The indices of marked faces are not preserved in the final tet mesh. However, indices of marked points
    // (vertices) are. So we can match faces in the tetmesh to faces in the input surface mesh by comparing their
    // vertex indices.
    surfaceFaces.clear();
    int nConstraints = 0;
    geometry.requireVertexIndices();
    for (size_t i = 0; i < nFaces; i++) {
        if (out.trifacemarkerlist[i]) {
            // Determine orientation.
            int sign = 1;
            Vertex vA = mesh.vertex(faces(i, 0));
            for (Halfedge he : vA.outgoingHalfedges()) {
                size_t vBIdx = geometry.vertexIndices[he.tipVertex()];
                size_t vCIdx = geometry.vertexIndices[he.next().tipVertex()];
                if (vBIdx == faces(i, 1) && vCIdx == faces(i, 2)) {
                    sign = 1;
                    break;
                }
                if (vBIdx == faces(i, 2) && vCIdx == faces(i, 1)) {
                    sign = -1;
                    break;
                }
            }
            surfaceFaces.push_back(sign * i);
            nConstraints++;
        }
    }
    geometry.unrequireVertexIndices();

    return true;
}

void SignedHeatTetSolver::tetmeshPointCloud(pointcloud::PointPositionGeometry& pointGeom, const Vector3& bboxMin,
                                            const Vector3 bboxMax) {

    // First Delaunay triangulate the surface of the bounding cube.
    tetgenio cubeSurface;
    Vector3 geomCentroid = centroid(pointGeom);
    double geomRadius = radius(pointGeom, geomCentroid);
    triangulateCube(cubeSurface, bboxMax, bboxMin);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    // Define nodes.
    size_t P = pointGeom.cloud.nPoints();
    in.firstnumber = 0;
    in.numberofpoints = P + cubeSurface.numberofpoints;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    // Copy nodes from the input surface mesh.
    for (size_t i = 0; i < pointGeom.cloud.nPoints(); i++) {
        Vector3 pos = pointGeom.positions[i];
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    // Copy nodes from the triangulation of the cube surface.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[P + i] = 0;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * P + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }

    // Define facets.
    in.numberoffacets = cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    // Copy tri faces from triangulation of cube surface.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[i] = 0;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = P + cubeSurface.trifacelist[3 * i + j];
            in.trifacelist[3 * i + j] = P + cubeSurface.trifacelist[3 * i + j];
        }
    }

    // Tet mesh!
    try {
        tetrahedralize(const_cast<char*>(TETFLAGS_PRESERVE.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
    }

    if (VERBOSE) std::cerr << "domain tet-meshed" << std::endl;

    // Get tet mesh info.
    getTetmeshData(out);
}

/*
 * Generate a constrained Delaunay tetrahedralization of a cube surrounding the input surface mesh.
 * Return only the boundary of the cube.
 */
void SignedHeatTetSolver::triangulateCube(tetgenio& cubeSurface, const Vector3& bboxMin, const Vector3 bboxMax) const {

    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;

    tetmeshCube(in, out, bboxMin, bboxMax);

    // Determine which faces/vertices lie on the boundary.
    std::vector<int> fIdx; // indices of boundary faces in tetmesh
    Eigen::VectorXi vMap =
        -1 * Eigen::VectorXi::Ones(out.numberofpoints); // Map tet mesh vertex indices to new indexing.
    std::set<int> vSet;                                 // Map surface mesh vertex indices to tetmesh indices
    for (int i = 0; i < out.numberoftrifaces; i++) {
        if (out.trifacemarkerlist[i] == 1) {
            fIdx.push_back(i);
            for (int j = 0; j < 3; j++) {
                // have to do this way, because vertices added along edges don't inherit the boundary marker... argh
                vSet.insert(out.trifacelist[3 * i + j]);
            }
        }
    }
    std::vector<int> vIdx;
    for (int i : vSet) {
        vMap(i) = vIdx.size();
        vIdx.push_back(i);
    }

    cubeSurface.firstnumber = 0;
    cubeSurface.numberofpoints = vIdx.size();
    cubeSurface.pointlist = new REAL[cubeSurface.numberofpoints * 3];
    cubeSurface.pointmarkerlist = new int[cubeSurface.numberofpoints];
    cubeSurface.numberoffacets = fIdx.size();
    cubeSurface.facetlist = new tetgenio::facet[cubeSurface.numberoffacets];
    cubeSurface.facetmarkerlist = new int[cubeSurface.numberoffacets];
    cubeSurface.numberoftrifaces = fIdx.size();
    cubeSurface.trifacelist = new int[cubeSurface.numberoftrifaces * 3];
    // Define nodes.
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        for (int j = 0; j < 3; j++) {
            cubeSurface.pointlist[3 * i + j] = out.pointlist[3 * vIdx[i] + j];
        }
    }

    // Define faces.
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        f = &cubeSurface.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
            cubeSurface.trifacelist[3 * i + j] = vMap(out.trifacelist[3 * fIdx[i] + j]);
        }
    }
}

void SignedHeatTetSolver::tetmeshCube(tetgenio& in, tetgenio& out, const Vector3& bboxMin,
                                      const Vector3 bboxMax) const {

    tetgenio::facet* f;
    tetgenio::polygon* p;

    // All indices start from 0.
    in.firstnumber = 0;
    in.numberofpoints = 8; // there are 8 vertices of a cube
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];

    // Define nodes.
    std::vector<Vector3> cubeCorners = buildCubeAroundSurface(bboxMin, bboxMax);

    for (int i = 0; i < in.numberofpoints; i++) {
        in.pointmarkerlist[i] = 1;
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = cubeCorners[i][j];
        }
    }

    // Define facets.
    in.numberoffacets = 6;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];

    int cubeIndices[6][4] = {
        {0, 1, 2, 3}, // bottom face
        {4, 5, 6, 7}, // top face
        {0, 1, 5, 4}, // left face
        {3, 2, 6, 7}, // right face
        {0, 3, 7, 4}, // front face
        {1, 2, 6, 5}  // back face
    };

    for (int i = 0; i < in.numberoffacets; i++) {
        in.facetmarkerlist[i] = 1;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 4;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 4; j++) {
            p->vertexlist[j] = cubeIndices[i][j];
        }
    }

    tetrahedralize(const_cast<char*>(TETFLAGS.c_str()), &in, &out);
}

/*
 * Construct a cube around the input surface mesh.
 * Returns the 3D positions of the 8 corners of the cube.
 */
std::vector<Vector3> SignedHeatTetSolver::buildCubeAroundSurface(const Vector3& bboxMin, const Vector3 bboxMax) const {

    Vector3 s = bboxMax - bboxMin;
    std::vector<Vector3> cubeCorners = {
        bboxMin,                           // bottom lower left corner
        bboxMin + Vector3{0., 0., s[2]},   // bottom upper left
        bboxMin + Vector3{s[0], 0., s[2]}, // bottom upper right
        bboxMin + Vector3{s[0], 0., 0.},   // bottom lower right
        bboxMin + Vector3{0., s[1], 0.},   // upper lower left corner
        bboxMin + Vector3{0., s[1], s[2]}, // upper upper left
        bboxMax,                           // upper upper right
        bboxMin + Vector3{s[0], s[1], 0.}  // upper lower right
    };

    return cubeCorners;
}

void SignedHeatTetSolver::getTetmeshData(tetgenio& out) {

    nVertices = out.numberofpoints;
    nTets = out.numberoftetrahedra;
    nFaces = out.numberoftrifaces;
    nEdges = out.numberofedges;
    // out.numberofcorners is 4
    if (VERBOSE) std::cerr << "# of vertices: " << nVertices << std::endl;
    if (VERBOSE) std::cerr << "# of tets: " << nTets << std::endl;
    if (VERBOSE) std::cerr << "# of facets: " << out.numberoffacets << std::endl;
    if (VERBOSE) std::cerr << "# of tri-faces: " << out.numberoftrifaces << std::endl; // # of constrained faces
    if (VERBOSE) std::cerr << "# of edges: " << nEdges << std::endl;
    vertices.resize(nVertices, 3);
    tets.resize(nTets, 4);
    faces.resize(nFaces, 3);

    // Determine element-vertex matrices.
    for (size_t i = 0; i < nVertices; i++) {
        for (int j = 0; j < 3; j++) {
            vertices(i, j) = out.pointlist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`vertices` constructed" << std::endl;
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            tets(i, j) = out.tetrahedronlist[4 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`tets` constructed" << std::endl;
    for (size_t i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++) {
            faces(i, j) = out.trifacelist[3 * i + j];
        }
    }
    if (VERBOSE) std::cerr << "`faces` constructed" << std::endl;

    // Determine adjacency info.
    tetFace.resize(nTets, 4);
    for (size_t i = 0; i < nTets; i++) {
        // All tets should already be positively oriented.
        Eigen::MatrixXi tetFaces(4, 3); // oriented faces in the tet
        tetFaces.row(0) << tets(i, 0), tets(i, 1), tets(i, 2);
        tetFaces.row(1) << tets(i, 0), tets(i, 3), tets(i, 1);
        tetFaces.row(2) << tets(i, 0), tets(i, 2), tets(i, 3);
        tetFaces.row(3) << tets(i, 1), tets(i, 3), tets(i, 2);
        for (int j = 0; j < 4; j++) {
            int fIdx = out.tet2facelist[4 * i + j];
            // Determine orientation (slow way)
            int s = -1;
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 3; l++) {
                    if (faces(fIdx, 0) == tetFaces(k, (0 + l) % 3) && faces(fIdx, 1) == tetFaces(k, (1 + l) % 3) &&
                        faces(fIdx, 2) == tetFaces(k, (2 + l) % 3)) {
                        s = 1;
                        break;
                    }
                }
            }
            tetFace(i, j) = s * fIdx;
        }
    }
    vertexTet.clear();
    vertexTet.resize(nVertices);
    for (size_t i = 0; i < nTets; i++) {
        for (int j = 0; j < 4; j++) {
            vertexTet[tets(i, j)].insert(i);
        }
    }
    if (VERBOSE) std::cerr << "Adjacency structures constructed" << std::endl;
}

double SignedHeatTetSolver::computeMeanNodeSpacing() const {

    double h = 0.;
    for (size_t i = 0; i < nTets; i++) {
        Eigen::MatrixXd faceBarycenters(4, 3);
        for (int j = 0; j < 4; j++) {
            faceBarycenters.row(j) = faceBarycenter(abs(tetFace(i, j)));
        }
        for (int j = 0; j < 4; j++) {
            for (int k = j + 1; k < 4; k++) {
                h += (faceBarycenters.row(j) - faceBarycenters.row(k)).norm();
            }
        }
    }
    h /= 6 * nTets;
    return h;
}


// Modified computeDistance function for EdgeDualNormalGeometry
// 这是用来处理 dual normal per edge 的函数
Vector<double> SignedHeatTetSolver::computeDistance(EdgeDualNormalGeometry& edgeGeom,
                                                    const SignedHeat3DOptions& options) {
    
    bool VERBOSE = true;
    
    std::cout << "SignedHeatTetSolver with dual normals per edge" << std::endl;

    // 初始网格构建（如果需要）
    if (options.rebuild || vertices.size() == 0) {
        std::chrono::time_point<high_resolution_clock> t1, t2;
        std::chrono::duration<double, std::milli> ms_fp;
        t1 = high_resolution_clock::now();
        if (VERBOSE) std::cerr << "Building initial tet mesh..." << std::endl;
        
        // Calculate mean edge length for area scaling
        double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
        double targetArea = meanEdgeLength * meanEdgeLength;
        TETFLAGS = TET_PREFIX + std::to_string(targetArea);
        TETFLAGS_PRESERVE = TET_PREFIX + std::to_string(targetArea) + "Y";
        
        // Create point cloud from edge vertices for tetmesh generation
        const auto& vertices_data = edgeGeom.getVertices();
        size_t nPts = vertices_data.size();
        cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
        pointcloud::PointData<Vector3> pointPositions = pointcloud::PointData<Vector3>(*cloud);
        for (size_t i = 0; i < nPts; i++) {
            pointPositions[i] = vertices_data[i];
        }
        pointPolyGeom = std::unique_ptr<pointcloud::PointPositionGeometry>(
            new pointcloud::PointPositionGeometry(*cloud, pointPositions));
        
        // 初始四面体化（不包含额外点）
        std::vector<Vector3> emptyExtraPoints;
        tetmeshEdgeGeo(edgeGeom, options, emptyExtraPoints);
        
        if (VERBOSE) std::cerr << "Computing initial tet mesh data..." << std::endl;
        meanNodeSpacing = computeMeanNodeSpacing();
        shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
        tetVolumes = computeTetVolumes();
        laplaceMat = dualLaplacian();
        
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) std::cerr << "Initial mesh build time (s): " << ms_fp.count() / 1000. << std::endl;
    }

    // 主计算循环
    int maxIterations = 1;  // 你可以随意修改这个值
    int iteration = 0;
    Vector<double> phi;
    bool meshWasRefined = false;  // 跟踪是否进行了网格细分
    
    while (iteration < maxIterations) {
        if (iteration > 0 && VERBOSE) {
            std::cerr << "Refinement iteration " << iteration << std::endl;
        }
        
        // Steps 1 & 2: 计算四面体中心的向量场
        if (VERBOSE) std::cerr << "Steps 1 & 2..." << std::endl;
        
        Eigen::MatrixXd Yt = Eigen::MatrixXd::Zero(nTets, 3);
        double lambda = std::sqrt(1. / shortTime);
        
        const auto& edges = edgeGeom.getEdges();
        const auto& vertices_data = edgeGeom.getVertices();
        const auto& normals1 = edgeGeom.getNormals1();
        const auto& normals2 = edgeGeom.getNormals2();
        size_t numEdges = edges.size();

        for (size_t i = 0; i < nTets; i++) {
            // Compute tet barycenter (query point)
            Vector3 q = {0, 0, 0};
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 3; k++) q[k] += vertices(tets(i, j), k);
            }
            q /= 4.;
            
            Vector3 X = estimateNormalAtPoint( edgeGeom, lambda, q );
            
            for (int j = 0; j < 3; j++) Yt(i, j) = X[j];
        }
        if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

        // Step 3: 积分得到距离函数
        if (VERBOSE) std::cerr << "Step 3..." << std::endl;
        
        pointPolyGeom->requireTuftedTriangulation();
        pointPolyGeom->tuftedGeom->requireVertexDualAreas();
        
        phi = options.fastIntegration ? integrateVectorFieldGreedily(*pointPolyGeom, Yt, options)
                                      : integrateVectorField(*pointPolyGeom, Yt, options);
        
        pointPolyGeom->tuftedGeom->unrequireVertexDualAreas();
        pointPolyGeom->unrequireTuftedTriangulation();
        
        if (VERBOSE) std::cerr << "\tCompleted." << std::endl;

        // 检查是否需要边细分（但要保证至少有一次完整的计算）
        // 关键改进：只有在还有剩余迭代次数时才进行细分
        bool shouldCheckRefinement = (iteration < maxIterations - 1) || (maxIterations == 1);
        
        if (shouldCheckRefinement) {
            std::vector<EdgeRefinementInfo> edgesToRefine;
            double isovalue = 0.0;

            if (checkEdgesNeedRefinement(edgeGeom, phi, lambda, isovalue, edgesToRefine)) {
                
                visualizeWithProblematicEdges(edgesToRefine);

                // 只有在还有剩余迭代次数时才实际进行细分
                if (iteration < maxIterations - 1) {
                    if (VERBOSE) {
                        std::cout << "Found " << edgesToRefine.size() << " edges that need refinement" << std::endl;
                        std::cout << "Rebuilding mesh with additional constraint points..." << std::endl;
                    }
                    
                    
                    // 提取新的约束点
                    std::vector<Vector3> extraPoints;
                    for (const auto& info : edgesToRefine) {
                        extraPoints.push_back(info.newVertexPosition);
                    }

                    
                    std::cout << "refine again" << std::endl;
                    
                    // 重建包含新约束点的四面体网格
                    std::chrono::time_point<high_resolution_clock> refine_start = high_resolution_clock::now();
                                        
                    
                    tetmeshEdgeGeo(edgeGeom, options, extraPoints);
                    
                    // 重新计算网格数据
                    meanNodeSpacing = computeMeanNodeSpacing();
                    shortTime = options.tCoef * meanNodeSpacing * meanNodeSpacing;
                    tetVolumes = computeTetVolumes();
                    laplaceMat = dualLaplacian();
                    
                    std::chrono::time_point<high_resolution_clock> refine_end = high_resolution_clock::now();
                    auto refine_time = std::chrono::duration_cast<std::chrono::milliseconds>(refine_end - refine_start);
                    
                    if (VERBOSE) {
                        std::cout << "Mesh refinement completed in " << refine_time.count() << " ms" << std::endl;
                        std::cout << "New mesh has " << vertices.rows() << " vertices and " << nTets << " tetrahedra" << std::endl;
                    }
                    
                    meshWasRefined = true;
                    // 继续下一次迭代，重新计算距离函数
                    iteration++;
                    continue;
                } else {
                    // 没有足够的迭代次数进行细分，但已经有了有效的 phi
                    if (VERBOSE) {
                        std::cout << "Found " << edgesToRefine.size() << " edges that need refinement, "
                                  << "but no remaining iterations. Using current result." << std::endl;
                    }
                }
            } else {
                if (VERBOSE) std::cout << "No edges need refinement, mesh is adequate" << std::endl;
            }
        }
        
        // 无论如何都要退出循环，因为我们已经有了有效的 phi
        break;
    }
    
    // 安全检查：确保 phi 不为空
    if (phi.size() == 0) {
        if (VERBOSE) std::cerr << "Warning: phi is empty, this should not happen!" << std::endl;
        // 创建一个默认的 phi（全零或基于网格大小）
        phi = Vector<double>::Zero(vertices.rows());
        if (VERBOSE) std::cerr << "Created fallback phi with size: " << phi.size() << std::endl;
    }
    
    if (VERBOSE) {
        std::cout << "Distance computation completed";
        if (meshWasRefined) {
            std::cout << " with mesh refinement";
        }
        std::cout << std::endl;
        std::cout << "Final phi size: " << phi.size() << ", expected vertices: " << vertices.rows() << std::endl;
    }
    
//    exportDataAndMesh(phi, options);
    return phi;
}




// helper for EdgeDualNormalGeometry

// maybe I don't need this calculateAverageEdgeLength becasue
// I have sample edge rate.
// I may want to set the edgeResampleRate as one of the options so
//  1. can adjust it
//  2. I don't need this function.
double SignedHeatTetSolver::calculateAverageEdgeLength(const EdgeDualNormalGeometry& edgeGeom) {
    const auto& edges = edgeGeom.getEdges();
    const auto& vertices = edgeGeom.getVertices();
    double totalLength = 0;

    for (const auto& edge : edges) {
        Vector3 v0 = vertices[edge.first];
        Vector3 v1 = vertices[edge.second];
        totalLength += (v1 - v0).norm();
    }

    return totalLength / edges.size();
}




void SignedHeatTetSolver::tetmeshEdgeGeo(EdgeDualNormalGeometry& edgeGeom,
                                         const SignedHeat3DOptions& options,
                                         const std::vector<Vector3>& extraPoints) {
    
    // First Delaunay triangulate the surface of the bounding cube
    tetgenio cubeSurface;
    
    // Use provided bounding box
    Vector3 bboxMin = options.bboxMin;
    Vector3 bboxMax = options.bboxMax;
    
    triangulateCube(cubeSurface, bboxMax, bboxMin);
    if (VERBOSE) std::cerr << "bounding box triangulated" << std::endl;
    
    tetgenio in, out;
    tetgenio::facet* f;
    tetgenio::polygon* p;
    
    // Get edge geometry data
    const auto& vertices_data = edgeGeom.getVertices();
    const auto& edges = edgeGeom.getEdges();
    size_t P = vertices_data.size();
    size_t E = edges.size();
    size_t EP = extraPoints.size();  // Number of extra points
    
    if (VERBOSE && EP > 0) {
        std::cerr << "Adding " << EP << " extra constraint points" << std::endl;
    }
    
    // Define nodes
    in.firstnumber = 0;
    in.numberofpoints = P + EP + cubeSurface.numberofpoints;  // Include extra points
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointmarkerlist = new int[in.numberofpoints];
    
    // Copy nodes from edge geometry vertices
    for (size_t i = 0; i < P; i++) {
        Vector3 pos = vertices_data[i];
        in.pointmarkerlist[i] = 1; // Mark as constraint points
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * i + j] = pos[j];
        }
    }
    
    // Copy extra constraint points
    for (size_t i = 0; i < EP; i++) {
        Vector3 pos = extraPoints[i];
        in.pointmarkerlist[P + i] = 2; // Mark as extra constraint points (different marker)
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * (P + i) + j] = pos[j];
        }
    }
    
    // Copy nodes from the triangulation of the cube surface
    for (int i = 0; i < cubeSurface.numberofpoints; i++) {
        in.pointmarkerlist[P + EP + i] = 0; // Mark as boundary points
        for (int j = 0; j < 3; j++) {
            in.pointlist[3 * (P + EP) + 3 * i + j] = cubeSurface.pointlist[3 * i + j];
        }
    }
    
    // Define facets (cube surface only)
    in.numberoffacets = cubeSurface.numberoftrifaces;
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    in.numberoftrifaces = in.numberoffacets;
    in.trifacelist = new int[3 * in.numberoffacets];
    
    // Copy tri faces from triangulation of cube surface
    for (int i = 0; i < cubeSurface.numberoftrifaces; i++) {
        in.facetmarkerlist[i] = 0;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = 3;
        p->vertexlist = new int[p->numberofvertices];
        for (int j = 0; j < 3; j++) {
            p->vertexlist[j] = P + EP + cubeSurface.trifacelist[3 * i + j];  // Adjust indices
            in.trifacelist[3 * i + j] = P + EP + cubeSurface.trifacelist[3 * i + j];
        }
    }
    
    // *** KEY PART: Define edge constraints for floating edges ***
    in.numberofedges = E;
    in.edgelist = new int[in.numberofedges * 2];
    in.edgemarkerlist = new int[in.numberofedges];
    
    // Copy edges as constraints
    for (size_t i = 0; i < E; i++) {
        in.edgelist[2 * i] = edges[i].first;      // First vertex of edge
        in.edgelist[2 * i + 1] = edges[i].second; // Second vertex of edge
        in.edgemarkerlist[i] = 1; // Mark as constraint edge
    }
    
    if (VERBOSE) {
        std::cerr << "Input summary:" << std::endl;
        std::cerr << "  Points: " << in.numberofpoints << " (" << P << " edge vertices + "
                  << EP << " extra points + " << cubeSurface.numberofpoints << " boundary)" << std::endl;
        std::cerr << "  Constraint edges: " << in.numberofedges << std::endl;
        std::cerr << "  Boundary facets: " << in.numberoffacets << std::endl;
    }
    
    // Tetrahedralize with edge constraints
    // 'p' = use PLC (piecewise linear complex) with edge constraints
    // 'Y' = preserve input surface mesh
    // 'q' = quality mesh generation
    
    // 宽松质量约束 + 体积控制
    std::string EDGE_TET_PREFIX = "pq2.0Yfennna"; // q2.0 很宽松，不会破坏边

    double meanEdgeLength = calculateAverageEdgeLength(edgeGeom);
//    double targetArea = meanEdgeLength * meanEdgeLength;
    double targetArea = meanEdgeLength * meanEdgeLength * meanEdgeLength;

    std::string tetFlags = EDGE_TET_PREFIX + std::to_string(targetArea);

    try {
        tetrahedralize(const_cast<char*>(tetFlags.c_str()), &in, &out);
    } catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
    } catch (const int& x) {
        std::cerr << "TetGen error code: " << x << std::endl;
    }
    
    if (VERBOSE) std::cerr << "domain tet-meshed with edge constraints" << std::endl;
    
    // Get tet mesh info
    getTetmeshData(out);
    
    // Display the tetmesh in the GUI.
//    polyscope::VolumeMesh* psVolumeMesh = polyscope::registerTetMesh("domain", vertices, tets);
//    polyscope::show();

}

Vector3 SignedHeatTetSolver::estimateNormalAtPoint( const EdgeDualNormalGeometry& edgeGeom, double lambda, const Vector3& q ) {
    const auto& edges = edgeGeom.getEdges();
    const auto& vertices_data = edgeGeom.getVertices();
    const auto& normals1 = edgeGeom.getNormals1();
    const auto& normals2 = edgeGeom.getNormals2();
    size_t numEdges = edges.size();

    // Integrate contributions from all edges
    Vector3 X = {0, 0, 0};
    for (size_t edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
        // Get edge endpoints
        size_t v0Idx = edges[edgeIdx].first;
        size_t v1Idx = edges[edgeIdx].second;
        Vector3 v0 = vertices_data[v0Idx];
        Vector3 v1 = vertices_data[v1Idx];
        
        // Calculate edge midpoint (sample point on edge)
        Vector3 p = (v0 + v1) * 0.5;
        
        // Get dual normals for this edge
        Vector3 n = normals1[edgeIdx];
        Vector3 n_prime = normals2[edgeIdx];
        
        // Calculate edge length as area weight
        double edgeLength = (v1 - v0).norm();
        double A = edgeLength;
        
        // Direction from edge midpoint to query point
        Vector3 direction = q - p;
        
        // Calculate dot products to determine which side of each plane the query point is on
        double dot1 = dot(direction, n);
        double dot2 = dot(direction, n_prime);
        
        Vector3 normalToUse;
        
        // Logic for choosing which normal to use
        if (dot1 > 0 && dot2 < 0) {
            normalToUse = n;
        } else if (dot1 < 0 && dot2 > 0) {
            normalToUse = n_prime;
        } else if (dot1 > 0 && dot2 > 0) {
            // Outside both n1 and n2
            Vector3 bisector = (n_prime + n) / 2;
            bisector = bisector.normalize();
            
            double dot_bisector = dot(direction, bisector);
            assert(dot_bisector > 0);
            
            if (dot_bisector > dot1 && dot_bisector > dot2) {
                normalToUse = direction.normalize();
            } else if (dot1 < dot2) {
                normalToUse = n_prime;
            } else {
                normalToUse = n;
            }
        } else {
            if (dot1 > dot2) {
                normalToUse = n;
            } else {
                normalToUse = n_prime;
            }
        }
        
        X += yukawaPotential(p, q, lambda) * normalToUse * A;
    }
    
    X /= X.norm();
    return X;
}

// 简单启发式：检查四面体网格边是否需要在中点插入顶点
bool SignedHeatTetSolver::checkEdgesNeedRefinement(
    const EdgeDualNormalGeometry& edgeGeom,
    const Vector<double>& phi,
    double lambda,
    double isovalue,
    std::vector<EdgeRefinementInfo>& edgesToRefine) {
    
    // 首先收集 edgeGeom 中已存在的边，这些边不需要检查
    std::set<std::pair<size_t, size_t>> existingEdges;
    const auto& vertices_data = edgeGeom.getVertices();
    const auto& edges = edgeGeom.getEdges();
    
    // 需要建立四面体顶点索引到 edgeGeom 顶点索引的映射
    std::unordered_map<size_t, size_t> tetVertToEdgeVertMap;

    const double EPS = 1e-6;
    
    // 遍历四面体顶点，找到对应的 edgeGeom 顶点
    for (size_t tetVertIdx = 0; tetVertIdx < vertices.rows(); tetVertIdx++) {
        Vector3 tetVertPos{vertices(tetVertIdx, 0), vertices(tetVertIdx, 1), vertices(tetVertIdx, 2)};
        
        // 在 edgeGeom 顶点中找到匹配的点
        for (size_t edgeVertIdx = 0; edgeVertIdx < vertices_data.size(); edgeVertIdx++) {
            Vector3 edgeVertPos = vertices_data[edgeVertIdx];
            if ((tetVertPos - edgeVertPos).norm() < EPS) {
                tetVertToEdgeVertMap[tetVertIdx] = edgeVertIdx;
                break;
            }
        }
    }
    
    // 将 edgeGeom 中的边转换为四面体顶点索引
    for (const auto& edge : edges) {
        size_t edgeV0 = edge.first;
        size_t edgeV1 = edge.second;
        
        // 找到对应的四面体顶点索引
        size_t tetV0 = SIZE_MAX, tetV1 = SIZE_MAX;
        for (const auto& mapping : tetVertToEdgeVertMap) {
            if (mapping.second == edgeV0) tetV0 = mapping.first;
            if (mapping.second == edgeV1) tetV1 = mapping.first;
        }
        
        if (tetV0 != SIZE_MAX && tetV1 != SIZE_MAX) {
            if (tetV0 > tetV1) std::swap(tetV0, tetV1);
            existingEdges.insert({tetV0, tetV1});
        }
    }
    
    if (VERBOSE) {
        std::cerr << "Found " << existingEdges.size() << " existing edges from edgeGeom to skip" << std::endl;
    }
    
    // 收集所有四面体边（避免重复）
    std::set<std::pair<size_t, size_t>> uniqueEdges;
    
    // 遍历所有四面体，收集边
    for (size_t tetIdx = 0; tetIdx < nTets; tetIdx++) {
        // 每个四面体有6条边：(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        std::array<size_t, 4> tetVerts;
        for (int i = 0; i < 4; i++) {
            tetVerts[i] = tets(tetIdx, i);
        }
        
        // 添加6条边（确保小索引在前，避免重复）
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                size_t v0 = tetVerts[i];
                size_t v1 = tetVerts[j];
                if (v0 > v1) std::swap(v0, v1);  // 确保v0 < v1
                uniqueEdges.insert({v0, v1});
            }
        }
    }
    
    std::cout << "Checking " << uniqueEdges.size() << " unique tetrahedral edges" << std::endl;
    std::cout << "Skipping " << existingEdges.size() << " edges that already exist in edgeGeom" << std::endl;
    
    bool foundEdgesToRefine = false;
    double threshold = 0.1 * meanNodeSpacing;  // 可调参数：接近等值线的阈值
    
    // 检查每条四面体边
    for (const auto& edge : uniqueEdges) {
        size_t v0Idx = edge.first;
        size_t v1Idx = edge.second;
        
        // 跳过 edgeGeom 中已存在的边
        if (existingEdges.count(edge) > 0) {
            continue;
        }
        
        // if (( v0Idx >= edgeGeom.getVertices().size() ) && ( v1Idx >= edgeGeom.getVertices().size() )) continue;
        // Check if both vertices are on the bounding box.
        if(
            // Is v0 on the bounding box?
            ( ( vertices.row(v0Idx).minCoeff() < -1 + EPS ) || ( vertices.row(v0Idx).maxCoeff() > 1 - EPS ) )
            // and
            &&
            // Is v1 on the bounding box?
            ( ( vertices.row(v1Idx).minCoeff() < -1 + EPS ) || ( vertices.row(v1Idx).maxCoeff() > 1 - EPS ) )
        ) {
            continue;
        }
        
        double phi0 = phi[v0Idx];
        double phi1 = phi[v1Idx];
        
        // 如果端点在等值线异侧，传统方法已经能处理，跳过
        // If they are on different sides of the isovalue, skip the edge.
        if ((phi0 - isovalue) * (phi1 - isovalue) < 0) {
            continue;
        }
        
        // 简单启发式：两端点都在同侧，但都接近等值线
        double dist0 = std::abs(phi0 - isovalue);
        double dist1 = std::abs(phi1 - isovalue);
        
        bool shouldSplit = false;
        double splitAt = 0.5; // 默认在中点分割
        
//        if (dist0 <= threshold && dist1 <= threshold) shouldSplit = true;

        /*
        TODO:
        We know values at the endpoints.
        If both endpoints are above (resp. below) the isovalue,
        check if the function as a higher-order polynomial would dip down (resp. up) and cross the isovalue.
        If we have the gradients at the endpoints, we can project them onto the line segment,
        fit a cubic polynomial, and see if it has a minima (resp. maxima) by taking its derivative
        and solving the resulting the quadratic formula.
        */
        const double x0 = 0;
        Eigen::Vector3d along = vertices.row(v1Idx) - vertices.row(v0Idx);
        const double x1 = ( along ).norm();
        // Normalize along so we can project onto it.
        along /= x1;

        const double f0 = phi0;
        const double f1 = phi1;

        const double dfdx0 = -dot( estimateNormalAtPoint( edgeGeom, lambda, eigenToGC( vertices.row(v0Idx)) ), eigenToGC(along ));
        const double dfdx1 = -dot( estimateNormalAtPoint( edgeGeom, lambda, eigenToGC( vertices.row(v1Idx)) ), eigenToGC( along ));

        /// Solve for a cubic polynomial for the line segment.
        // f(x) = ax^3 + bx^2 + cx + d
        // f(x0) = f(0) = d = f0;
        // df/dx(x0) = df/dx(0) = c = dfdx0
        // f(x1) = a x1^3 + b x1^2 + dfdx0 x1 + f0 = phi1
        // df/dx(x1) = 3a x1^2 + 2b x1 + dfdx0 = dfdx1
        Eigen::Matrix4d A(4,4);
        Eigen::Vector4d rhs(4);
        // Fill the matrix and right-hand-side with our equations.
        A.row(0) << 0, 0, 0, 1;
        rhs(0) = f0;
        A.row(1) << 0, 0, 1, 0;
        rhs(1) = dfdx0;
        A.row(2) << x1*x1*x1, x1*x1, x1, 1;
        rhs(2) = f1;
        A.row(3) << 3*x1*x1, 2*x1, 1, 0;
        rhs(3) = dfdx1;
        Eigen::Vector4d abcd = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
        

        /// Find the minimum and maximum of the cubic by solving the quadratic formula.
        // Discard solutions outside of (x0,x1).
        const double a = abcd(0);
        const double b = abcd(1);
        const double c = abcd(2);
        const double d = abcd(3);
        // df/dx = 3ax^2 + 2bx + c
        const double discriminant = 4*b*b - 12*a*c;
        double x_solution0 = -1;
        double x_solution1 = -1;
        
        
        if( discriminant >= 0 && std::abs(a) > 1e-12 ) {
            double temp0 = ( -2*b + std::sqrt(discriminant) ) / ( 6*a );
            double temp1 = ( -2*b - std::sqrt(discriminant) ) / ( 6*a );
            
            // 只保留在边界内的解
            if (temp0 > 0 && temp0 < x1) {
                x_solution0 = temp0;
            }
            if (temp1 > 0 && temp1 < x1) {
                x_solution1 = temp1;
            }
        }
        
        
        if (x_solution0 >= 0 && x_solution1 >= 0) {
            
            const double f_solution0 = a*x_solution0*x_solution0*x_solution0 + b*x_solution0*x_solution0 + c*x_solution0 + d;
            const double f_solution1 = a*x_solution1*x_solution1*x_solution1 + b*x_solution1*x_solution1 + c*x_solution1 + d;
            
            double f_solution_max = f_solution0;
            double f_solution_min = f_solution1;
            double x_solution_max = x_solution0;
            double x_solution_min = x_solution1;
            
            if( f_solution0 < f_solution1 ) {
                std::swap( f_solution_max, f_solution_min );
                std::swap( x_solution_max, x_solution_min );
            }
        
//
//            std::cout << "Valid solutions found:" << std::endl;
//            std::cout << "x_solution_max " << x_solution_max << std::endl;
//            std::cout << "f_solution_max " << f_solution_max << std::endl;
//            std::cout << "x_solution_min " << x_solution_min << std::endl;
//            std::cout << "f_solution_min " << f_solution_min << std::endl;
//            std::cout << "phi0 " << phi0 << std::endl;
//            std::cout << "phi1 " << phi1 << std::endl;
//            std::cout << "x1 " << x1 << std::endl;
//            std::cout << "isovalue " << isovalue << std::endl;

            
            
            
            /// If phi0 and phi1 are both greater than the isovalue,
            /// we split at the minimum quadratic formula solution
            /// if it is within (x0,x1) and evaluates to less than the isovalue.
            if( phi0 > isovalue && phi1 > isovalue  && f_solution_min < isovalue ){
                shouldSplit = true;
                splitAt = x_solution_min/x1;
            }

            /// If phi0 and phi1 are both less than the isovalue,
            /// we split at the maximum quadratic formula solution
            /// if it is within (x0,x1) and evaluates to greater than the isovalue.
            if( phi0 < isovalue && phi1 < isovalue  && f_solution_max > isovalue ){
                shouldSplit = true;
                splitAt = x_solution_max/x1;
            }
            
            
            if (shouldSplit) {
                
                EdgeRefinementInfo info;
                info.v0Idx = v0Idx;
                info.v1Idx = v1Idx;
                
                Vector3 v0{vertices(v0Idx, 0), vertices(v0Idx, 1), vertices(v0Idx, 2)};
                Vector3 v1{vertices(v1Idx, 0), vertices(v1Idx, 1), vertices(v1Idx, 2)};
                
                info.newVertexPosition = v0 + (v1-v0) * splitAt;
                edgesToRefine.push_back(info);
                
                foundEdgesToRefine = true;
                
            }

        }
        
        
        
        
      

    }
    
    std::cout << "Found " << edgesToRefine.size() << " edges that need refinement" << std::endl;
    

    return foundEdgesToRefine;
}


/*
 * Write CSV file, where each row is a vertex of the tetrahedral mesh.
 * The vertex positions are stored in the vertices matrix.
 * Columns: xCoord, yCoord, zCoord, SDF
 * The first three columns record the (x,y,z) position of the vertex.
 * "SDF" records the SDF value at the vertex.
 */
void SignedHeatTetSolver::exportData(const Vector<double>& phi, const SignedHeat3DOptions& options) const {
    

    std::string filename = "../export/" + options.meshname + "_tet.csv";
    std::fstream f;
    f.open(filename, std::ios::out | std::ios::trunc);
    
    if (f.is_open()) {
        f << "xCoord,yCoord,zCoord,SDF" << "\n";
        
        // 遍历所有四面体网格顶点
        for (size_t i = 0; i < vertices.rows(); i++) {
            // vertices 是存储顶点坐标的矩阵 (nVertices × 3)
            double x = vertices(i, 0);
            double y = vertices(i, 1);
            double z = vertices(i, 2);
            double sdf = phi[i];
            
            f << x << "," << y << "," << z << "," << sdf << "\n";
        }
        
        f.close();
        if (VERBOSE) std::cerr << "File " << filename << " written successfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export '" << filename << "'!" << std::endl;
    }
}

/*
 * Export tetrahedral mesh vertices and tetrahedra to CSV files
 * This allows exact reconstruction of the mesh in Python for visualization
 */
void SignedHeatTetSolver::exportMesh(const SignedHeat3DOptions& options) const {
    
    // Export vertices
    std::string vertices_filename = "../export/" + options.meshname + "_vertices.csv";
    std::fstream vertices_file;
    vertices_file.open(vertices_filename, std::ios::out | std::ios::trunc);
    
    if (vertices_file.is_open()) {
        vertices_file << "x,y,z" << "\n";
        for (size_t i = 0; i < vertices.rows(); i++) {
            vertices_file << vertices(i, 0) << ","
                         << vertices(i, 1) << ","
                         << vertices(i, 2) << "\n";
        }
        vertices_file.close();
        if (VERBOSE) std::cerr << "Vertices file " << vertices_filename << " written successfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export vertices file '" << vertices_filename << "'!" << std::endl;
    }
    
    // Export tetrahedra
    std::string tets_filename = "../export/" + options.meshname + "_tets.csv";
    std::fstream tets_file;
    tets_file.open(tets_filename, std::ios::out | std::ios::trunc);
    
    if (tets_file.is_open()) {
        tets_file << "v0,v1,v2,v3" << "\n";
        for (size_t i = 0; i < nTets; i++) {
            tets_file << tets(i, 0) << ","
                     << tets(i, 1) << ","
                     << tets(i, 2) << ","
                     << tets(i, 3) << "\n";
        }
        tets_file.close();
        if (VERBOSE) std::cerr << "Tetrahedra file " << tets_filename << " written successfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export tetrahedra file '" << tets_filename << "'!" << std::endl;
    }
    
    if (VERBOSE) {
        std::cout << "Mesh export summary:" << std::endl;
        std::cout << "  Vertices: " << vertices.rows() << std::endl;
        std::cout << "  Tetrahedra: " << nTets << std::endl;
        std::cout << "  Files: " << vertices_filename << ", " << tets_filename << std::endl;
    }
}

/*
 * Combined export function - exports both SDF data and mesh structure
 * This gives you everything needed for exact visualization in Python
 */
void SignedHeatTetSolver::exportDataAndMesh(const Vector<double>& phi, const SignedHeat3DOptions& options) const {
    
    // Export SDF data (original function)
    std::string sdf_filename = "../export/" + options.meshname + "_sdf.csv";
    std::fstream sdf_file;
    sdf_file.open(sdf_filename, std::ios::out | std::ios::trunc);
    
    if (sdf_file.is_open()) {
        sdf_file << "xCoord,yCoord,zCoord,SDF" << "\n";
        for (size_t i = 0; i < vertices.rows(); i++) {
            sdf_file << vertices(i, 0) << ","
                    << vertices(i, 1) << ","
                    << vertices(i, 2) << ","
                    << phi[i] << "\n";
        }
        sdf_file.close();
        if (VERBOSE) std::cerr << "SDF file " << sdf_filename << " written successfully." << std::endl;
    } else {
        if (VERBOSE) std::cerr << "Could not export SDF file '" << sdf_filename << "'!" << std::endl;
    }
    
    // Export mesh structure
    exportMesh(options);
    
    if (VERBOSE) {
        std::cout << "Complete export finished. Use these files in Python:" << std::endl;
        std::cout << "  - SDF data: " << sdf_filename << std::endl;
        std::cout << "  - Vertices: ../export/" + options.meshname + "_vertices.csv" << std::endl;
        std::cout << "  - Tetrahedra: ../export/" + options.meshname + "_tets.csv" << std::endl;
    }
}


void SignedHeatTetSolver::visualizeWithProblematicEdges(const std::vector<EdgeRefinementInfo>& edgesToRefine) {
    
    
//    polyscope::registerTetMesh("tets", vertices, tets);
    
    // show problematic edges
    
    std::vector<std::array<double, 3>> edgeVertices;
    std::vector<std::array<int, 2>> edgeIndices;


    
    for (size_t i = 0; i < edgesToRefine.size(); i++) {
        const auto& edge = edgesToRefine[i];

        Eigen::Vector3d v0 = vertices.row(edge.v0Idx);
        Eigen::Vector3d v1 = vertices.row(edge.v1Idx);

        edgeVertices.push_back({v0[0], v0[1], v0[2]});
        edgeVertices.push_back({v1[0], v1[1], v1[2]});
        
    
        

        edgeIndices.push_back({static_cast<int>(2*i), static_cast<int>(2*i+1)});
    }

    auto* curves = polyscope::registerCurveNetwork("problematic edges",
                                                   edgeVertices, edgeIndices);

    curves->setColor({1.0, 0.0, 0.0});
    curves->setRadius(0.01);
    

    std::vector<Vector3> extraPoints;
    for (const auto& info : edgesToRefine) {
        extraPoints.push_back(info.newVertexPosition);
    }


    // 注册point cloud
    auto* pointCloud = polyscope::registerPointCloud("split points", extraPoints);
    pointCloud->setPointRadius(0.02);  // 比边粗一点，更显眼
    pointCloud->setPointColor({0.0, 1.0, 0.0});  // 绿色
    
    polyscope::show();
    
    
        

    
}


