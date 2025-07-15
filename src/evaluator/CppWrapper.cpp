#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip> 

// Forward declarations
struct MispredictionRange {
    size_t start_idx;
    size_t end_idx;
    size_t length() const { return end_idx - start_idx + 1; }
};

struct EvaluationResults {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
    
    // Overall volume metrics
    double mae_volume;
    double mse_volume;
    double rmse_volume;
    
    // Per-class volume metrics
    struct ClassVolumeMetrics {
        double mae_volume;
        double mse_volume;
        double rmse_volume;
        size_t sample_count;
        
        ClassVolumeMetrics() : mae_volume(0.0), mse_volume(0.0), rmse_volume(0.0), sample_count(0) {}
    };
    
    ClassVolumeMetrics class_0_volume_metrics;  // When true label = 0 (no intersection)
    ClassVolumeMetrics class_1_volume_metrics;  // When true label = 1 (intersection)
    
    double inference_time_ms;
    double preprocessing_time_ms;
    double total_time_ms;
    double time_per_sample_ms;
    double samples_per_second;
    size_t total_samples;
    size_t correct_predictions;
    std::vector<MispredictionRange> misprediction_ranges;
    
    // Constructor
    EvaluationResults() : accuracy(0.0), precision(0.0), recall(0.0), f1_score(0.0),
                         mae_volume(0.0), mse_volume(0.0), rmse_volume(0.0),
                         inference_time_ms(0.0), preprocessing_time_ms(0.0),
                         total_time_ms(0.0), time_per_sample_ms(0.0), samples_per_second(0.0),
                         total_samples(0), correct_predictions(0) {}
};

// File utilities
namespace FileUtils {
    bool file_exists_and_readable(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }
    
    void validate_file(const std::string& path, const std::string& description) {
        if (!file_exists_and_readable(path)) {
            throw std::runtime_error("Error: " + description + " does not exist or is not readable: " + path);
        }
    }
}

// Vectorized Unitary Tetrahedron Transformation
namespace UnitaryTetrahedronTransform {
    torch::Tensor apply_transformation(const torch::Tensor& input_tensor) {
        // Handle both single sample [24] and batch [N, 24] inputs
        torch::Tensor data = input_tensor.to(torch::kFloat64);
        bool is_batch = data.dim() == 2;
        
        if (!is_batch) {
            data = data.view({1, 24});  // Add batch dimension
        }
        
        int batch_size = data.size(0);
        
        // Reshape into two tetrahedra (batch_size, 4 vertices each, 3 coordinates)
        torch::Tensor tetra1 = data.slice(1, 0, 12).view({batch_size, 4, 3});   // [N, 4, 3]
        torch::Tensor tetra2 = data.slice(1, 12, 24).view({batch_size, 4, 3});  // [N, 4, 3]
        
        // Get first tetrahedron's basis vectors for each sample in batch
        torch::Tensor v0 = tetra1.slice(1, 0, 1);  // [N, 1, 3] - first vertex of each tetrahedron
        torch::Tensor edge_vectors = tetra1.slice(1, 1, 4) - v0;  // [N, 3, 3] - edge vectors
        
        // Calculate inverse transformation matrix for each sample
        torch::Tensor inv_transform;
        try {
            inv_transform = torch::inverse(edge_vectors);  // [N, 3, 3]
        } catch (const std::exception& e) {
            throw std::runtime_error("One or more tetrahedron basis matrices are singular and cannot be inverted");
        }
        
        // Vectorized transformation - translate and transform all vertices at once
        torch::Tensor translated = tetra2 - v0;  // [N, 4, 3]
        torch::Tensor transformed_verts = torch::bmm(translated, inv_transform);  // [N, 4, 3]
        
        // Return flattened transformed vertices
        torch::Tensor result = transformed_verts.view({batch_size, 12});  // [N, 12]
        
        if (!is_batch) {
            result = result.squeeze(0);  // Remove batch dimension for single sample
        }
        
        return result;
    }
}

// Vectorized Principal Axis Transformation
namespace PrincipalAxisTransform {
    torch::Tensor apply_transformation(const torch::Tensor& input_tensor) {
        // Handle both single sample [24] and batch [N, 24] inputs
        torch::Tensor data = input_tensor.to(torch::kFloat64);
        bool is_batch = data.dim() == 2;
        
        if (!is_batch) {
            data = data.view({1, 24});  // Add batch dimension
        }
        
        int batch_size = data.size(0);
        
        // Reshape into two tetrahedra (batch_size, 4 vertices each, 3 coordinates)
        torch::Tensor tetra1 = data.slice(1, 0, 12).view({batch_size, 4, 3});   // [N, 4, 3]
        torch::Tensor tetra2 = data.slice(1, 12, 24).view({batch_size, 4, 3});  // [N, 4, 3]
        
        // Compute tetra1's centroid for each sample
        torch::Tensor centroid1 = tetra1.mean(1, true);  // [N, 1, 3]
        torch::Tensor tetra1_centered = tetra1 - centroid1;  // [N, 4, 3]
        
        // Compute covariance matrix of tetra1's centered vertices for each sample
        torch::Tensor cov_matrix = torch::bmm(tetra1_centered.transpose(1, 2), tetra1_centered);  // [N, 3, 3]
        
        // Get eigenvalues and eigenvectors for each sample
        auto eigen_result = torch::linalg_eigh(cov_matrix);
        torch::Tensor eigenvalues = std::get<0>(eigen_result);   // [N, 3]
        torch::Tensor eigenvectors = std::get<1>(eigen_result);  // [N, 3, 3]
        
        // Sort eigenvectors by eigenvalues in descending order for each sample
        torch::Tensor sorted_indices = torch::argsort(eigenvalues, -1, true);  // [N, 3] descending
        
        // Use advanced indexing to get sorted eigenvectors
        torch::Tensor batch_indices = torch::arange(batch_size, torch::kLong).view({batch_size, 1, 1}).expand({batch_size, 3, 3});
        torch::Tensor col_indices = torch::arange(3, torch::kLong).view({1, 3, 1}).expand({batch_size, 3, 3});
        torch::Tensor sorted_indices_expanded = sorted_indices.view({batch_size, 1, 3}).expand({batch_size, 3, 3});
        
        torch::Tensor R_canonical = eigenvectors.gather(2, sorted_indices_expanded);  // [N, 3, 3]
        
        // Apply transformation to both tetrahedra
        torch::Tensor tetra1_centered_all = tetra1 - centroid1;  // [N, 4, 3]
        torch::Tensor tetra2_centered_all = tetra2 - centroid1;  // [N, 4, 3]
        
        torch::Tensor tetra1_transformed = torch::bmm(tetra1_centered_all, R_canonical);  // [N, 4, 3]
        torch::Tensor tetra2_transformed = torch::bmm(tetra2_centered_all, R_canonical);  // [N, 4, 3]
        
        // Flatten and concatenate results
        torch::Tensor result = torch::cat({
            tetra1_transformed.view({batch_size, 12}),
            tetra2_transformed.view({batch_size, 12})
        }, 1);  // [N, 24]
        
        if (!is_batch) {
            result = result.squeeze(0);  // Remove batch dimension for single sample
        }
        
        return result;
    }
}

// Vectorized Unit Cube Normalization
namespace UnitCubeNormalization {
    torch::Tensor apply_transformation(const torch::Tensor& input_tensor) {
        // Handle both single sample [24] and batch [N, 24] inputs
        torch::Tensor data = input_tensor.to(torch::kFloat64);
        bool is_batch = data.dim() == 2;
        
        if (!is_batch) {
            data = data.view({1, 24});  // Add batch dimension
        }
        
        int batch_size = data.size(0);
        
        // Reshape into two tetrahedra (batch_size, 4 vertices each, 3 coordinates)
        torch::Tensor tetra1 = data.slice(1, 0, 12).view({batch_size, 4, 3});   // [N, 4, 3]
        torch::Tensor tetra2 = data.slice(1, 12, 24).view({batch_size, 4, 3});  // [N, 4, 3]
        
        // Combine all vertices to find global bounding box for each sample
        torch::Tensor all_vertices = torch::cat({tetra1, tetra2}, 1);  // [N, 8, 3]
        
        // Find min and max coordinates across all dimensions for each sample
        auto min_result = torch::min(all_vertices, 1);
        auto max_result = torch::max(all_vertices, 1);
        torch::Tensor min_coords = std::get<0>(min_result);  // [N, 3]
        torch::Tensor max_coords = std::get<0>(max_result);  // [N, 3]
        
        // Calculate the range in each dimension for each sample
        torch::Tensor range_coords = max_coords - min_coords;  // [N, 3]
        
        // Find the maximum range to maintain aspect ratio for each sample
        torch::Tensor max_range = std::get<0>(torch::max(range_coords, 1));  // [N]
        
        // Avoid division by zero for degenerate cases
        torch::Tensor epsilon = torch::full_like(max_range, 1e-10f);
        torch::Tensor safe_max_range = torch::maximum(max_range, epsilon);
        
        // Normalize: translate to origin, then scale to fit in unit cube
        torch::Tensor translated = all_vertices - min_coords.unsqueeze(1);  // [N, 8, 3]
        torch::Tensor normalized = translated / safe_max_range.unsqueeze(1).unsqueeze(2);  // [N, 8, 3]
        
        // Split back into two tetrahedra and flatten
        torch::Tensor normalized_tetra1 = normalized.slice(1, 0, 4).view({batch_size, 12});  // [N, 12]
        torch::Tensor normalized_tetra2 = normalized.slice(1, 4, 8).view({batch_size, 12});  // [N, 12]
        
        // Concatenate and return
        torch::Tensor result = torch::cat({normalized_tetra1, normalized_tetra2}, 1);  // [N, 24]
        
        if (!is_batch) {
            result = result.squeeze(0);  // Remove batch dimension for single sample
        }
        
        return result;
    }
}


namespace VolumeNormalization {
    torch::Tensor apply_transformation(const torch::Tensor& input_tensor) {
        // Handle both single sample [24] and batch [N, 24] inputs
        torch::Tensor data = input_tensor.to(torch::kFloat64);
        bool is_batch = data.dim() == 2;
        
        if (!is_batch) {
            data = data.view({1, 24});  // Add batch dimension
        }
        
        int batch_size = data.size(0);
        
        // Reshape into two tetrahedra (batch_size, 4 vertices each, 3 coordinates)
        torch::Tensor tetra1 = data.slice(1, 0, 12).view({batch_size, 4, 3});   // [N, 4, 3]
        torch::Tensor tetra2 = data.slice(1, 12, 24).view({batch_size, 4, 3});  // [N, 4, 3]
        
        // Calculate volumes for BOTH T1 and T2 using the determinant formula
        // Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
        
        // T1 volumes
        torch::Tensor v0_t1 = tetra1.slice(1, 0, 1);  // [N, 1, 3] - first vertex of T1
        torch::Tensor edge_vectors_t1 = tetra1.slice(1, 1, 4) - v0_t1;  // [N, 3, 3] - edge vectors from v0
        torch::Tensor det_t1 = torch::det(edge_vectors_t1);  // [N]
        torch::Tensor volume_t1 = torch::abs(det_t1) / 6.0;  // [N]
        
        // T2 volumes
        torch::Tensor v0_t2 = tetra2.slice(1, 0, 1);  // [N, 1, 3] - first vertex of T2
        torch::Tensor edge_vectors_t2 = tetra2.slice(1, 1, 4) - v0_t2;  // [N, 3, 3] - edge vectors from v0
        torch::Tensor det_t2 = torch::det(edge_vectors_t2);  // [N]
        torch::Tensor volume_t2 = torch::abs(det_t2) / 6.0;  // [N]
        
        // Define target volume range [1e-6, 1e-1]
        double min_target_volume = 1e-6;
        double max_target_volume = 1e-1;
        
        // For each sample, determine the scaling factor needed to bring BOTH volumes into range
        torch::Tensor scale_factors = torch::ones({batch_size}, torch::kFloat64);
        
        for (int i = 0; i < batch_size; i++) {
            double vol1 = volume_t1[i].item<double>();
            double vol2 = volume_t2[i].item<double>();
            
            // Handle degenerate cases (volumes too small)
            if (vol1 < 1e-20 || vol2 < 1e-20) {
                scale_factors[i] = 1.0;  // Keep original scale for degenerate cases
                continue;
            }
            
            // Find the scaling factor that ensures BOTH volumes are within range
            // Since volume scales as scale^3, we need: min_target <= vol * scale^3 <= max_target
            
            // Calculate scale bounds for T1: (min_target/vol1)^(1/3) <= scale <= (max_target/vol1)^(1/3)
            double scale_min_t1 = std::pow(min_target_volume / vol1, 1.0/3.0);
            double scale_max_t1 = std::pow(max_target_volume / vol1, 1.0/3.0);
            
            // Calculate scale bounds for T2: (min_target/vol2)^(1/3) <= scale <= (max_target/vol2)^(1/3)
            double scale_min_t2 = std::pow(min_target_volume / vol2, 1.0/3.0);
            double scale_max_t2 = std::pow(max_target_volume / vol2, 1.0/3.0);
            
            // Find the intersection of valid scale ranges
            double scale_min_combined = std::max(scale_min_t1, scale_min_t2);
            double scale_max_combined = std::min(scale_max_t1, scale_max_t2);
            
            // Check if there's a valid scale that works for both
            if (scale_min_combined <= scale_max_combined) {
                // Choose a random scale within the valid range (log-uniform distribution)
                double log_min = std::log(scale_min_combined);
                double log_max = std::log(scale_max_combined);
                double random_factor = static_cast<double>(rand()) / RAND_MAX;  // Random [0,1]
                double log_scale = log_min + random_factor * (log_max - log_min);
                scale_factors[i] = std::exp(log_scale);
            } else {
                // No valid scale exists - choose the scale that minimizes the maximum violation
                // This is a compromise solution for edge cases
                
                // Try scale that brings the larger volume to max_target
                double larger_vol = std::max(vol1, vol2);
                double scale_for_larger = std::pow(max_target_volume / larger_vol, 1.0/3.0);
                
                // Try scale that brings the smaller volume to min_target
                double smaller_vol = std::min(vol1, vol2);
                double scale_for_smaller = std::pow(min_target_volume / smaller_vol, 1.0/3.0);
                
                // Choose the scale that keeps both volumes closer to the valid range
                double violation1 = std::max(0.0, std::min(vol1, vol2) * std::pow(scale_for_larger, 3) - max_target_volume) + 
                                   std::max(0.0, min_target_volume - std::max(vol1, vol2) * std::pow(scale_for_larger, 3));
                double violation2 = std::max(0.0, std::min(vol1, vol2) * std::pow(scale_for_smaller, 3) - max_target_volume) + 
                                   std::max(0.0, min_target_volume - std::max(vol1, vol2) * std::pow(scale_for_smaller, 3));
                
                scale_factors[i] = (violation1 <= violation2) ? scale_for_larger : scale_for_smaller;
            }
        }
        
        // Apply RIGID transformation: uniform scaling around a common reference point
        // Choose the centroid of both tetrahedra combined as the reference point
        torch::Tensor all_vertices = torch::cat({tetra1, tetra2}, 1);  // [N, 8, 3]
        torch::Tensor combined_centroid = all_vertices.mean(1, true);  // [N, 1, 3]
        
        // Apply uniform scaling to both tetrahedra around the combined centroid
        // This preserves the relative positioning and orientation between tetrahedra
        torch::Tensor centered_t1 = tetra1 - combined_centroid;  // [N, 4, 3]
        torch::Tensor centered_t2 = tetra2 - combined_centroid;  // [N, 4, 3]
        
        torch::Tensor scaled_centered_t1 = centered_t1 * scale_factors.unsqueeze(1).unsqueeze(2);  // [N, 4, 3]
        torch::Tensor scaled_centered_t2 = centered_t2 * scale_factors.unsqueeze(1).unsqueeze(2);  // [N, 4, 3]
        
        torch::Tensor scaled_t1 = scaled_centered_t1 + combined_centroid;  // [N, 4, 3]
        torch::Tensor scaled_t2 = scaled_centered_t2 + combined_centroid;  // [N, 4, 3]
        
        // Flatten and concatenate results
        torch::Tensor result = torch::cat({
            scaled_t1.view({batch_size, 12}),
            scaled_t2.view({batch_size, 12})
        }, 1);  // [N, 24]
        
        if (!is_batch) {
            result = result.squeeze(0);  // Remove batch dimension for single sample
        }
        
        return result;
    }
}


namespace TransformationManager {
    enum class TransformationType {
        NONE,
        UNIT_CUBE_NORMALIZATION,              
        UNITARY_TETRAHEDRON,
        PRINCIPAL_AXIS,
        VOLUME_NORMALIZATION,                 // Volume normalization
        PRINCIPAL_THEN_VOLUME,                // NEW: Principal axis + volume normalization
        UNIT_CUBE_THEN_UNITARY,              // Unit cube + unitary tetrahedron
        UNIT_CUBE_THEN_PRINCIPAL,            // Unit cube + principal axis  
        UNIT_CUBE_THEN_BOTH,                 // Unit cube + principal axis + unitary
        VOLUME_THEN_UNITARY,                 // Volume normalization + unitary tetrahedron
        VOLUME_THEN_PRINCIPAL,               // Volume normalization + principal axis
        VOLUME_THEN_BOTH,                    // Volume normalization + principal axis + unitary
        PRINCIPAL_THEN_VOLUME_THEN_UNITARY,  // NEW: Principal + volume + unitary
        BOTH                                 // Principal axis + unitary tetrahedron
    };
    
    TransformationType parse_transformation_type(const std::string& transform_str) {
        if (transform_str == "none" || transform_str.empty()) {
            return TransformationType::NONE;
        } else if (transform_str == "unit_cube_normalization") {
            return TransformationType::UNIT_CUBE_NORMALIZATION;
        } else if (transform_str == "unitary_tetrahedron_transformation") {
            return TransformationType::UNITARY_TETRAHEDRON;
        } else if (transform_str == "principal_axis_transformation") {
            return TransformationType::PRINCIPAL_AXIS;
        } else if (transform_str == "volume_normalization") {
            return TransformationType::VOLUME_NORMALIZATION;
        } else if (transform_str == "principal_then_volume") {
            return TransformationType::PRINCIPAL_THEN_VOLUME;
        } else if (transform_str == "unit_cube_then_unitary") {
            return TransformationType::UNIT_CUBE_THEN_UNITARY;
        } else if (transform_str == "unit_cube_then_principal") {
            return TransformationType::UNIT_CUBE_THEN_PRINCIPAL;
        } else if (transform_str == "unit_cube_then_both") {
            return TransformationType::UNIT_CUBE_THEN_BOTH;
        } else if (transform_str == "volume_then_unitary") {
            return TransformationType::VOLUME_THEN_UNITARY;
        } else if (transform_str == "volume_then_principal") {
            return TransformationType::VOLUME_THEN_PRINCIPAL;
        } else if (transform_str == "volume_then_both") {
            return TransformationType::VOLUME_THEN_BOTH;
        } else if (transform_str == "principal_then_volume_then_unitary") {
            return TransformationType::PRINCIPAL_THEN_VOLUME_THEN_UNITARY;
        } else if (transform_str == "both") {
            return TransformationType::BOTH;
        } else {
            throw std::runtime_error("Unknown transformation type: " + transform_str);
        }
    }
    
    // Updated vectorized batch transformation function
    torch::Tensor apply_transformations_batch(const torch::Tensor& input_batch, TransformationType transform_type) {
        torch::Tensor result = input_batch;
        
        switch (transform_type) {
            case TransformationType::NONE:
                // No transformation
                break;
                
            case TransformationType::UNIT_CUBE_NORMALIZATION:
                result = UnitCubeNormalization::apply_transformation(result);
                break;
                
            case TransformationType::UNITARY_TETRAHEDRON:
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::PRINCIPAL_AXIS:
                result = PrincipalAxisTransform::apply_transformation(result);
                break;
                
            case TransformationType::VOLUME_NORMALIZATION:
                result = VolumeNormalization::apply_transformation(result);
                break;
                
            case TransformationType::PRINCIPAL_THEN_VOLUME:
                result = PrincipalAxisTransform::apply_transformation(result);
                result = VolumeNormalization::apply_transformation(result);
                break;
                
            case TransformationType::UNIT_CUBE_THEN_UNITARY:
                result = UnitCubeNormalization::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::UNIT_CUBE_THEN_PRINCIPAL:
                result = UnitCubeNormalization::apply_transformation(result);
                result = PrincipalAxisTransform::apply_transformation(result);
                break;
                
            case TransformationType::UNIT_CUBE_THEN_BOTH:
                result = UnitCubeNormalization::apply_transformation(result);
                result = PrincipalAxisTransform::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::VOLUME_THEN_UNITARY:
                result = VolumeNormalization::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::VOLUME_THEN_PRINCIPAL:
                result = VolumeNormalization::apply_transformation(result);
                result = PrincipalAxisTransform::apply_transformation(result);
                break;
                
            case TransformationType::VOLUME_THEN_BOTH:
                result = VolumeNormalization::apply_transformation(result);
                result = PrincipalAxisTransform::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::PRINCIPAL_THEN_VOLUME_THEN_UNITARY:
                result = PrincipalAxisTransform::apply_transformation(result);
                result = VolumeNormalization::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
                
            case TransformationType::BOTH:
                result = PrincipalAxisTransform::apply_transformation(result);
                result = UnitaryTetrahedronTransform::apply_transformation(result);
                break;
        }
        
        return result;
    }
    
    // Single sample function for backward compatibility
    torch::Tensor apply_transformations(const torch::Tensor& input_tensor, TransformationType transform_type) {
        return apply_transformations_batch(input_tensor, transform_type);
    }
    
    int get_expected_output_dim(TransformationType transform_type) {
        switch (transform_type) {
            case TransformationType::NONE:
            case TransformationType::UNIT_CUBE_NORMALIZATION:
            case TransformationType::PRINCIPAL_AXIS:
            case TransformationType::UNIT_CUBE_THEN_PRINCIPAL:
            case TransformationType::VOLUME_NORMALIZATION:
            case TransformationType::VOLUME_THEN_PRINCIPAL:
            case TransformationType::PRINCIPAL_THEN_VOLUME:
                return 24;
            case TransformationType::UNITARY_TETRAHEDRON:
            case TransformationType::UNIT_CUBE_THEN_UNITARY:
            case TransformationType::UNIT_CUBE_THEN_BOTH:
            case TransformationType::VOLUME_THEN_UNITARY:
            case TransformationType::VOLUME_THEN_BOTH:
            case TransformationType::PRINCIPAL_THEN_VOLUME_THEN_UNITARY:
            case TransformationType::BOTH:
                return 12;  // Final output after unitary tetrahedron transformation
            default:
                return 24;
        }
    }
}

// Update the Sample struct to store raw data
struct Sample {
    torch::Tensor raw_input;  // [24] - original coordinates (no preprocessing)
    torch::Tensor volume;     // [1]
    torch::Tensor label;      // [1]
};

// Update load_csv 
std::vector<Sample> load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Sample> data;
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;

        // Parse each comma-separated value
        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stof(cell));
        }

        if (values.size() != 26) {
            std::cerr << "Warning: Expected 26 values, got " << values.size() << std::endl;
            continue;
        }

        // Split into input and targets
        std::vector<double> input_values(values.begin(), values.begin() + 24);
        float volume = values[24];
        float label = values[25];

        // Store RAW data (no preprocessing here for coordinates)
        torch::Tensor raw_input_tensor = torch::from_blob(input_values.data(), {24}, torch::kFloat64).clone();
        torch::Tensor label_tensor = torch::tensor({label}, torch::kFloat64);
        torch::Tensor volume_tensor = torch::tensor({volume}, torch::kFloat64);

        data.push_back({raw_input_tensor, volume_tensor, label_tensor});
    }

    return data;
}

namespace DataManager {
    void apply_transformations(std::vector<Sample>& dataset, TransformationManager::TransformationType transform_type) {
        if (transform_type == TransformationManager::TransformationType::NONE) {
            std::cout << "No transformations applied - using raw data" << std::endl;
            return;
        }
        
        std::string transform_name;
        switch (transform_type) {
            case TransformationManager::TransformationType::UNIT_CUBE_NORMALIZATION:
                transform_name = "unit cube normalization";
                break;
            case TransformationManager::TransformationType::UNITARY_TETRAHEDRON:
                transform_name = "unitary tetrahedron";
                break;
            case TransformationManager::TransformationType::PRINCIPAL_AXIS:
                transform_name = "principal axis";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_UNITARY:
                transform_name = "unit cube + unitary tetrahedron";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_PRINCIPAL:
                transform_name = "unit cube + principal axis";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_BOTH:
                transform_name = "unit cube + principal axis + unitary tetrahedron";
                break;
            case TransformationManager::TransformationType::BOTH:
                transform_name = "principal axis + unitary tetrahedron";
                break;
            default:
                transform_name = "unknown";
        }
        
        std::cout << "Applying " << transform_name << " transformations..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Vectorized batch processing
        const size_t batch_size = 1024;  // Process in batches for memory efficiency
        
        for (size_t batch_start = 0; batch_start < dataset.size(); batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, dataset.size());
            size_t current_batch_size = batch_end - batch_start;
            
            // Stack batch inputs
            std::vector<torch::Tensor> batch_inputs;
            for (size_t i = batch_start; i < batch_end; ++i) {
                batch_inputs.push_back(dataset[i].raw_input);
            }
            
            torch::Tensor batch_tensor = torch::stack(batch_inputs);  // [batch_size, 24]
            
            try {
                // Apply vectorized transformations
                torch::Tensor transformed_batch = TransformationManager::apply_transformations_batch(batch_tensor, transform_type);
                
                // Update dataset with transformed data
                for (size_t i = 0; i < current_batch_size; ++i) {
                    dataset[batch_start + i].raw_input = transformed_batch[i].clone();
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to transform batch starting at " << batch_start << ": " << e.what() << std::endl;
                std::cerr << "Falling back to individual sample processing for this batch..." << std::endl;
                
                // Fallback to individual processing for this batch
                for (size_t i = batch_start; i < batch_end; ++i) {
                    try {
                        dataset[i].raw_input = TransformationManager::apply_transformations(dataset[i].raw_input, transform_type);
                    } catch (const std::exception& e2) {
                        std::cerr << "Warning: Failed to transform sample " << i << ": " << e2.what() << std::endl;
                        // Keep original data if transformation fails
                    }
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Vectorized transformations completed in " << duration.count() << "ms" << std::endl;
        
        // Verify output dimensions
        if (!dataset.empty()) {
            int expected_dim = TransformationManager::get_expected_output_dim(transform_type);
            int actual_dim = dataset[0].raw_input.size(0);
            std::cout << "Output dimensions: " << actual_dim << " (expected: " << expected_dim << ")" << std::endl;
            
            if (actual_dim != expected_dim) {
                std::cerr << "Warning: Output dimension mismatch!" << std::endl;
            }
        }
    }

    std::vector<Sample> load_dataset(const std::string& csv_path, 
                                   TransformationManager::TransformationType transform_type = TransformationManager::TransformationType::NONE) {
        std::cout << "Loading dataset from: " << csv_path << std::endl;
        
        FileUtils::validate_file(csv_path, "CSV file");
        
        std::vector<Sample> dataset = load_csv(csv_path);
        
        if (dataset.empty()) {
            throw std::runtime_error("Dataset is empty!");
        }
        
        std::cout << "Dataset loaded successfully! Samples: " << dataset.size() << std::endl;
        
        // Apply transformations once during loading
        if (transform_type != TransformationManager::TransformationType::NONE) {
            apply_transformations(dataset, transform_type);
        } else {
            std::cout << "Dataset loaded with RAW data (no preprocessing applied)" << std::endl;
        }
        
        return dataset;
    }
}

// Updated Evaluator with vectorized preprocessing
namespace Evaluator {
    EvaluationResults evaluate_model(torch::jit::script::Module& model, 
                                   const std::vector<Sample>& dataset,
                                   int batch_size = 64,
                                   TransformationManager::TransformationType transform_type = TransformationManager::TransformationType::NONE,
                                   bool measure_preprocessing = false, double volume_scaling_factor = 1000.0) {
        
        EvaluationResults results;
        results.total_samples = dataset.size();
        
        std::vector<double> true_labels;
        std::vector<double> predicted_labels;
        std::vector<double> true_volumes;
        std::vector<double> predicted_volumes;
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        double total_preprocessing_time = 0.0;
        
        model.eval();
        
        // Process data in batches
        for (size_t batch_idx = 0; batch_idx < dataset.size(); batch_idx += batch_size) {
            size_t batch_end = std::min(batch_idx + batch_size, dataset.size());
            
            std::vector<torch::Tensor> batch_inputs;
            std::vector<torch::Tensor> batch_labels;
            std::vector<torch::Tensor> batch_volumes;
            
            // Prepare batch data
            for (size_t i = batch_idx; i < batch_end; ++i) {
                batch_inputs.push_back(dataset[i].raw_input);
                batch_labels.push_back(dataset[i].label);
                batch_volumes.push_back(dataset[i].volume);
            }
            
            // Stack batch tensors
            torch::Tensor batch_input_tensor = torch::stack(batch_inputs);
            torch::Tensor batch_label_tensor = torch::stack(batch_labels);
            torch::Tensor batch_volume_tensor = torch::stack(batch_volumes);
            
            // Apply vectorized transformations during evaluation if measuring preprocessing
            if (measure_preprocessing && transform_type != TransformationManager::TransformationType::NONE) {
                auto prep_start = std::chrono::high_resolution_clock::now();
                
                batch_input_tensor = TransformationManager::apply_transformations_batch(batch_input_tensor, transform_type);
                
                auto prep_end = std::chrono::high_resolution_clock::now();
                total_preprocessing_time += std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
            }
            
            // Forward pass
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_input_tensor);
            
            torch::Tensor output = model.forward(inputs).toTensor();
            
            // Handle different output formats
            torch::Tensor pred_labels, pred_volumes;
            
            if (output.size(-1) == 2) {
                // Combined task: [classification, regression]
                pred_labels = output.slice(-1, 0, 1).squeeze();
                pred_volumes = output.slice(-1, 1, 2).squeeze();
                pred_volumes = pred_volumes / volume_scaling_factor;
            } else if (output.size(-1) == 1) {
                // Single task - check if it's classification or regression based on data
                if (batch_label_tensor.max().item<double>() <= 1.0f && batch_label_tensor.min().item<double>() >= 0.0f) {
                    // Classification task
                    pred_labels = output.squeeze();
                    pred_volumes = torch::zeros_like(batch_volume_tensor.squeeze());
                } else {
                    // Regression task
                    pred_labels = torch::zeros_like(batch_label_tensor.squeeze());
                    pred_volumes = output.squeeze() / volume_scaling_factor;;
                }
            } else {
                throw std::runtime_error("Unexpected output dimension: " + std::to_string(output.size(-1)));
            }
            
            // Convert to binary predictions for classification
            torch::Tensor binary_pred_labels = (pred_labels > 0.5).to(torch::kFloat64);
            
            // Store results
            for (size_t i = 0; i < batch_end - batch_idx; ++i) {
                true_labels.push_back(batch_label_tensor[i].item<double>());
                predicted_labels.push_back(binary_pred_labels[i].item<double>());
                true_volumes.push_back(batch_volume_tensor[i].item<double>());
                predicted_volumes.push_back(pred_volumes[i].item<double>());
            }
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        results.inference_time_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        results.preprocessing_time_ms = total_preprocessing_time;
        results.total_time_ms = results.inference_time_ms + results.preprocessing_time_ms;
        
        // Calculate time per sample and samples per second
        if (results.total_samples > 0) {
            results.time_per_sample_ms = results.total_time_ms / results.total_samples;
            results.samples_per_second = 1000.0 / results.time_per_sample_ms;  // Convert ms to seconds
        }
        
        // Calculate classification metrics
        size_t tp = 0, tn = 0, fp = 0, fn = 0;
        results.correct_predictions = 0;
        
        for (size_t i = 0; i < true_labels.size(); ++i) {
            bool true_label = true_labels[i] > 0.5f;
            bool pred_label = predicted_labels[i] > 0.5f;
            
            if (true_label && pred_label) tp++;
            else if (!true_label && !pred_label) tn++;
            else if (!true_label && pred_label) fp++;
            else if (true_label && !pred_label) fn++;
            
            if (true_label == pred_label) {
                results.correct_predictions++;
            }
        }
        
        results.accuracy = static_cast<double>(results.correct_predictions) / results.total_samples;
        
        if (tp + fp > 0) {
            results.precision = static_cast<double>(tp) / (tp + fp);
        }
        if (tp + fn > 0) {
            results.recall = static_cast<double>(tp) / (tp + fn);
        }
        if (results.precision + results.recall > 0) {
            results.f1_score = 2.0 * (results.precision * results.recall) / (results.precision + results.recall);
        }
        
        // Calculate volume regression metrics - overall and per-class
        double sum_ae = 0.0, sum_se = 0.0;
        double sum_ae_class_0 = 0.0, sum_se_class_0 = 0.0;
        double sum_ae_class_1 = 0.0, sum_se_class_1 = 0.0;
        size_t count_class_0 = 0, count_class_1 = 0;
        
        for (size_t i = 0; i < true_volumes.size(); ++i) {
            double error = predicted_volumes[i] - true_volumes[i];
            double abs_error = std::abs(error);
            double squared_error = error * error;
            
            // Overall metrics
            sum_ae += abs_error;
            sum_se += squared_error;
            
            // Per-class metrics
            bool true_label = true_labels[i] > 0.5;
            if (true_label) {
                // Class 1 (intersection)
                sum_ae_class_1 += abs_error;
                sum_se_class_1 += squared_error;
                count_class_1++;
            } else {
                // Class 0 (no intersection)
                sum_ae_class_0 += abs_error;
                sum_se_class_0 += squared_error;
                count_class_0++;
            }
        }
        
        // Overall volume metrics
        results.mae_volume = sum_ae / true_volumes.size();
        results.mse_volume = sum_se / true_volumes.size();
        results.rmse_volume = std::sqrt(results.mse_volume);
        
        // Class 0 volume metrics (no intersection)
        results.class_0_volume_metrics.sample_count = count_class_0;
        if (count_class_0 > 0) {
            results.class_0_volume_metrics.mae_volume = sum_ae_class_0 / count_class_0;
            results.class_0_volume_metrics.mse_volume = sum_se_class_0 / count_class_0;
            results.class_0_volume_metrics.rmse_volume = std::sqrt(results.class_0_volume_metrics.mse_volume);
        }
        
        // Class 1 volume metrics (intersection)
        results.class_1_volume_metrics.sample_count = count_class_1;
        if (count_class_1 > 0) {
            results.class_1_volume_metrics.mae_volume = sum_ae_class_1 / count_class_1;
            results.class_1_volume_metrics.mse_volume = sum_se_class_1 / count_class_1;
            results.class_1_volume_metrics.rmse_volume = std::sqrt(results.class_1_volume_metrics.mse_volume);
        }
        
        // Find misprediction ranges
        std::vector<MispredictionRange> misprediction_ranges;
        bool in_misprediction = false;
        size_t start_idx = 0;
        
        for (size_t i = 0; i < true_labels.size(); ++i) {
            bool is_misprediction = (true_labels[i] > 0.5f) != (predicted_labels[i] > 0.5f);
            
            if (is_misprediction && !in_misprediction) {
                start_idx = i;
                in_misprediction = true;
            } else if (!is_misprediction && in_misprediction) {
                misprediction_ranges.push_back({start_idx, i - 1});
                in_misprediction = false;
            }
        }
        
        // Handle case where misprediction range extends to the end
        if (in_misprediction) {
            misprediction_ranges.push_back({start_idx, true_labels.size() - 1});
        }
        
        results.misprediction_ranges = misprediction_ranges;
        
        return results;
    }

    
    void print_results(const EvaluationResults& results) {
        std::cout << "\n=== EVALUATION RESULTS ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        
        std::cout << "\nDataset Statistics:" << std::endl;
        std::cout << "  Total samples: " << results.total_samples << std::endl;
        std::cout << "  Correct predictions: " << results.correct_predictions << std::endl;
        
        std::cout << "\nClassification Metrics:" << std::endl;
        std::cout << "  Accuracy: " << results.accuracy << std::endl;
        std::cout << "  Precision: " << results.precision << std::endl;
        std::cout << "  Recall: " << results.recall << std::endl;
        std::cout << "  F1-Score: " << results.f1_score << std::endl;
        
        std::cout << "\nOverall Volume Regression Metrics:" << std::endl;
        std::cout << std::scientific << std::setprecision(8);
        std::cout << "  MAE: " << results.mae_volume << std::endl;
        std::cout << "  MSE: " << results.mse_volume << std::endl;
        std::cout << "  RMSE: " << results.rmse_volume << std::endl;
        
        std::cout << "\nClass 0 (No Intersection) Volume Metrics:" << std::endl;
        std::cout << "  Sample Count: " << results.class_0_volume_metrics.sample_count << std::endl;
        if (results.class_0_volume_metrics.sample_count > 0) {
            std::cout << "  MAE: " << results.class_0_volume_metrics.mae_volume << std::endl;
            std::cout << "  MSE: " << results.class_0_volume_metrics.mse_volume << std::endl;
            std::cout << "  RMSE: " << results.class_0_volume_metrics.rmse_volume << std::endl;
        } else {
            std::cout << "  No samples in this class" << std::endl;
        }
        
        std::cout << "\nClass 1 (Intersection) Volume Metrics:" << std::endl;
        std::cout << "  Sample Count: " << results.class_1_volume_metrics.sample_count << std::endl;
        if (results.class_1_volume_metrics.sample_count > 0) {
            std::cout << "  MAE: " << results.class_1_volume_metrics.mae_volume << std::endl;
            std::cout << "  MSE: " << results.class_1_volume_metrics.mse_volume << std::endl;
            std::cout << "  RMSE: " << results.class_1_volume_metrics.rmse_volume << std::endl;
        } else {
            std::cout << "  No samples in this class" << std::endl;
        }
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\nPerformance Metrics:" << std::endl;
        std::cout << "  Inference Time: " << results.inference_time_ms << " ms" << std::endl;
        std::cout << "  Preprocessing Time: " << results.preprocessing_time_ms << " ms" << std::endl;
        std::cout << "  Total Time: " << results.total_time_ms << " ms" << std::endl;
        std::cout << std::setprecision(6);
        std::cout << "  Time per Sample: " << results.time_per_sample_ms << " ms" << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "  Samples per Second: " << results.samples_per_second << std::endl;
        
        if (!results.misprediction_ranges.empty()) {
            std::cout << "\nMisprediction Ranges:" << std::endl;
            for (const auto& range : results.misprediction_ranges) {
                std::cout << "  [" << range.start_idx << "-" << range.end_idx << "] (length: " << range.length() << ")" << std::endl;
            }
        }
        
        std::cout << "\n==========================" << std::endl;
    }


}

namespace ArgumentParser {
    struct Args {
        std::string model_path;
        std::string csv_path;
        int batch_size;
        TransformationManager::TransformationType transform_type;
        bool measure_preprocessing;
    };
    
    Args parse_arguments(int argc, char* argv[]) {
        if (argc < 3) {
            throw std::runtime_error(
                "Usage: " + std::string(argv[0]) + " <model_path> <csv_path> [batch_size] [--transform=TYPE] [--measure-preprocessing]\n"
                "  model_path: Path to the TorchScript model file\n"
                "  csv_path: Path to the CSV dataset file\n"
                "  batch_size: Batch size for evaluation (default: 32)\n"
                "  --transform=TYPE: Transformation type options:\n"
                "    none - No transformation (default)\n"
                "    unit_cube_normalization - Normalize both tetrahedra to unit cube\n"
                "    unitary_tetrahedron_transformation - Transform using first tetrahedron as basis\n"
                "    principal_axis_transformation - Align with principal axes\n"
                "    volume_normalization - Scale T2 volume to [1e-13, 1e-1], apply same scale to T1\n"
                "    principal_then_volume - Principal axis alignment + volume normalization\n"
                "    unit_cube_then_unitary - Unit cube normalization + unitary transformation\n"
                "    unit_cube_then_principal - Unit cube normalization + principal axis\n"
                "    unit_cube_then_both - Unit cube + principal axis + unitary\n"
                "    volume_then_unitary - Volume normalization + unitary transformation\n"
                "    volume_then_principal - Volume normalization + principal axis\n"
                "    volume_then_both - Volume normalization + principal axis + unitary\n"
                "    principal_then_volume_then_unitary - Principal axis + volume + unitary\n"
                "    both - Principal axis + unitary transformation\n"
                "  --measure-preprocessing: Apply transformations during evaluation to measure preprocessing time"
            );
        }
        
        Args args;
        args.model_path = argv[1];
        args.csv_path = argv[2];
        args.batch_size = 32;  // default
        args.transform_type = TransformationManager::TransformationType::NONE;  // default
        args.measure_preprocessing = false;  // default
        
        // Parse optional arguments
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            
            // Check if it's a batch size (numeric argument without -- prefix)
            if (arg.find_first_not_of("0123456789") == std::string::npos) {
                args.batch_size = std::stoi(arg);
            }
            // Check for transform argument
            else if (arg.substr(0, 12) == "--transform=") {
                std::string transform_str = arg.substr(12);
                args.transform_type = TransformationManager::parse_transformation_type(transform_str);
            }
            // Check for preprocessing measurement flag
            else if (arg == "--measure-preprocessing") {
                args.measure_preprocessing = true;
            }
            else {
                std::cerr << "Warning: Unknown argument: " << arg << std::endl;
            }
        }
        
        return args;
    }
}

// Model management
namespace ModelManager {
    torch::jit::script::Module load_model(const std::string& model_path) {
        std::cout << "Loading model from: " << model_path << std::endl;
        
        FileUtils::validate_file(model_path, "Model file");
        
        torch::jit::script::Module scripted_net = torch::jit::load(model_path, torch::kCPU);
        std::cout << "Model loaded successfully!" << std::endl;
        
        return scripted_net;
    }
}

int main(int argc, char* argv[]) {
    try {
        auto args = ArgumentParser::parse_arguments(argc, argv);
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Model: " << args.model_path << std::endl;
        std::cout << "  Dataset: " << args.csv_path << std::endl;
        std::cout << "  Batch size: " << args.batch_size << std::endl;
        std::cout << "  Measure preprocessing: " << (args.measure_preprocessing ? "Yes" : "No") << std::endl;
        
        std::string transform_name;
        switch (args.transform_type) {
            case TransformationManager::TransformationType::NONE:
                transform_name = "none";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_NORMALIZATION:
                transform_name = "unit_cube_normalization";
                break;
            case TransformationManager::TransformationType::UNITARY_TETRAHEDRON:
                transform_name = "unitary_tetrahedron_transformation";
                break;
            case TransformationManager::TransformationType::PRINCIPAL_AXIS:
                transform_name = "principal_axis_transformation";
                break;
            case TransformationManager::TransformationType::PRINCIPAL_THEN_VOLUME:
                transform_name = "principal_then_volume";
                break;
            case TransformationManager::TransformationType::PRINCIPAL_THEN_VOLUME_THEN_UNITARY:
                transform_name = "principal_then_volume_then_unitary";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_UNITARY:
                transform_name = "unit_cube_then_unitary";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_PRINCIPAL:
                transform_name = "unit_cube_then_principal";
                break;
            case TransformationManager::TransformationType::UNIT_CUBE_THEN_BOTH:
                transform_name = "unit_cube_then_both";
                break;
            case TransformationManager::TransformationType::BOTH:
                transform_name = "both (principal_axis + unitary_tetrahedron)";
                break;
        }
 
        std::cout << "  Transformation: " << transform_name << std::endl;
        std::cout << std::endl;
        
        // Load model
        auto model = ModelManager::load_model(args.model_path);
        
        // Load dataset - apply transformations during loading unless we want to measure preprocessing
        auto dataset_transform_type = args.measure_preprocessing ? 
            TransformationManager::TransformationType::NONE : args.transform_type;
        auto dataset = DataManager::load_dataset(args.csv_path, dataset_transform_type);
        
        // Evaluate model
        auto eval_transform_type = args.measure_preprocessing ? 
            args.transform_type : TransformationManager::TransformationType::NONE;
        auto results = Evaluator::evaluate_model(model, dataset, args.batch_size, 
                                                eval_transform_type, args.measure_preprocessing);
        
        // Print results
        Evaluator::print_results(results);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}