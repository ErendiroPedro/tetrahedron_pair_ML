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

struct Sample {
    torch::Tensor input;  // [24] - remove the batch dimension
    torch::Tensor volume; // [1]
    torch::Tensor label;  // [1]
};

std::vector<Sample> load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Sample> data;
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<float> values;
        std::stringstream ss(line);
        std::string cell;

        // Parse each comma-separated value
        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stof(cell));
        }

        assert (values.size() == 26); // Ensure correct shape

        // Split into input and targets
        std::vector<float> input_values(values.begin(), values.begin() + 24);
        float volume = values[24];
        float label = values[25];

        // Create tensors without batch dimension - will be batched later
        torch::Tensor input_tensor = torch::from_blob(input_values.data(), {24}).clone();
        torch::Tensor label_tensor = torch::tensor({label});
        torch::Tensor volume_tensor = torch::tensor({volume});

        data.push_back({input_tensor, volume_tensor, label_tensor});
    }

    return data;
}

struct CommonParams {
    size_t input_dim;
    std::string task;
    std::string activation;
    double dropout_rate;
    double volume_scale_factor;
};

struct MLPParams {
    std::vector<size_t> shared_layers;
    std::vector<size_t> classification_head;
    std::vector<size_t> regression_head;
};

struct TetrahedronPairNetParams {
    std::vector<size_t> per_vertex_layers;
    std::vector<size_t> per_tetrahedron_layers;
    std::vector<size_t> per_two_tetrahedra_layers;
    std::string vertices_aggregation_function;
    std::string tetrahedra_aggregation_function;
};

// Activation functions
namespace Activations {
    torch::Tensor relu(const torch::Tensor& x) { return torch::relu(x); }
    torch::Tensor sigmoid(const torch::Tensor& x) { return torch::sigmoid(x); }
    torch::Tensor tanh_act(const torch::Tensor& x) { return torch::tanh(x); }
    torch::Tensor gelu(const torch::Tensor& x) { return torch::gelu(x); }
    torch::Tensor leaky_relu(const torch::Tensor& x) { return torch::leaky_relu(x, 0.01); }
    torch::Tensor linear(const torch::Tensor& x) { return x; }
    
    std::function<torch::Tensor(const torch::Tensor&)> get_activation(const std::string& name) {
        static std::map<std::string, std::function<torch::Tensor(const torch::Tensor&)>> activations = {
            {"relu", relu},
            {"sigmoid", sigmoid},
            {"tanh", tanh_act},
            {"gelu", gelu},
            {"leaky_relu", leaky_relu},
            {"linear", linear},
            {"none", linear}
        };
        return activations.count(name) ? activations[name] : linear;
    }
}

// Aggregation functions
namespace Aggregations {
    enum class Type { MAX, MEAN, SUM, MIN, HADAMARD_PROD };
    
    Type from_string(const std::string& name) {
        if (name == "max") return Type::MAX;
        if (name == "mean") return Type::MEAN;
        if (name == "sum") return Type::SUM;
        if (name == "min") return Type::MIN;
        if (name == "hadamard_prod") return Type::HADAMARD_PROD;
        return Type::MAX; // default
    }
    
    torch::Tensor aggregate(const std::vector<torch::Tensor>& embeddings, Type type) {
        if (embeddings.empty()) return torch::Tensor();
        
        switch (type) {
            case Type::MAX:
                return torch::stack(embeddings).amax(0);
            case Type::MEAN:
                return torch::stack(embeddings).mean(0);
            case Type::SUM:
                return torch::stack(embeddings).sum(0);
            case Type::MIN:
                return torch::stack(embeddings).amin(0);
            case Type::HADAMARD_PROD:
                torch::Tensor result = torch::ones_like(embeddings[0]);
                for (const auto& emb : embeddings) {
                    result = result * torch::relu(emb);
                }
                return result;
        }
        return torch::Tensor();
    }
}

// File utilities with RAII
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

// Data management
namespace DataManager {
    std::vector<Sample> load_dataset(const std::string& csv_path) {
        std::cout << "Loading dataset from: " << csv_path << std::endl;
        
        FileUtils::validate_file(csv_path, "CSV file");
        
        std::vector<Sample> dataset = load_csv(csv_path);
        
        if (dataset.empty()) {
            throw std::runtime_error("Dataset is empty!");
        }
        
        std::cout << "Dataset loaded successfully! Samples: " << dataset.size() << std::endl;
        return dataset;
    }
    
    // Placeholder for data transformations
    void apply_transformations(std::vector<Sample>& dataset) {
        // TODO: Implement data transformations here
        // - Normalization
        // - Augmentation
        // - Preprocessing
        std::cout << "Applying data transformations... (placeholder)" << std::endl;
    }
}

// Evaluation management
namespace Evaluator {
    struct EvaluationResults {
        double accuracy;
        double avg_inference_time_ms;
        double samples_per_second;
        int correct_predictions;
        int total_samples;
    };

    std::vector<float> extract_predictions(torch::Tensor output, const std::string& task) {
        // // Convert to double if needed
        // if (output.dtype() != torch::kDouble) {
        //     output = output.to(torch::kDouble);
        // }
        
        auto accessor = output.accessor<float, 2>();
        
        if (task == "IntersectionStatus") {
            return {accessor[0][0] > 0.5 ? 1.0 : 0.0};
        }
        else if (task == "IntersectionVolume") {
            return {accessor[0][0] / 1e8};  // Use correct scale factor
        }
        else if (task == "IntersectionStatus_IntersectionVolume") {
            float cls_pred = accessor[0][0] > 0.5 ? 1.0 : 0.0;
            float reg_pred = accessor[0][1] / 1e8;  // Use correct scale factor
            return {cls_pred, reg_pred};
        }
        
        throw std::runtime_error("Unknown task: " + task);
    }

    EvaluationResults evaluate_model(torch::jit::script::Module& model, 
                                   const std::vector<Sample>& dataset,
                                   int batch_size = 64) {
        std::cout << "Starting evaluation with batch_size=" << batch_size << "..." << std::endl;
        
        int correct = 0;
        int total = 0;
        double total_inference_time_ms = 0.0;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Process in batches
        for (size_t batch_idx = 0; batch_idx < dataset.size(); batch_idx += batch_size) {
            size_t batch_end = std::min(batch_idx + batch_size, dataset.size());
            size_t current_batch_size = batch_end - batch_idx;
            
            // Prepare batch tensors
            std::vector<torch::Tensor> batch_inputs;
            std::vector<torch::Tensor> batch_labels;
            std::vector<torch::Tensor> batch_volumes;
            
            for (size_t i = batch_idx; i < batch_end; ++i) {
                batch_inputs.push_back(dataset[i].input);
                batch_labels.push_back(dataset[i].label);
                batch_volumes.push_back(dataset[i].volume);
            }
            
            // Stack into batch tensors
            torch::Tensor batched_inputs = torch::stack(batch_inputs).to(torch::kCPU);  // [batch_size, 24]
            torch::Tensor batched_labels = torch::stack(batch_labels);                   // [batch_size, 1]
            torch::Tensor batched_volumes = torch::stack(batch_volumes);                 // [batch_size, 1]
            
            std::vector<torch::jit::IValue> inputs{batched_inputs};
            
            // Inference timing
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor batch_output = model.forward(inputs).toTensor();  // [batch_size, output_dim]
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double, std::milli> inference_time = end - start;
            total_inference_time_ms += inference_time.count();
            
            // Process each sample in the batch
            for (size_t i = 0; i < current_batch_size; ++i) {
                // Extract single sample output [1, output_dim]
                torch::Tensor single_output = batch_output.index({static_cast<int64_t>(i)}).unsqueeze(0);
                
                // Extract predictions
                std::vector<float> preds = extract_predictions(single_output, "IntersectionStatus_IntersectionVolume");
                
                // Classification check
                int predicted_cls = static_cast<int>(preds[0]);
                int true_cls = batched_labels[i].item<int>();
                
                if (predicted_cls == true_cls) {
                    correct++;
                }
                
                total++;
                
                // Debug: Print first few predictions
                if (total <= 5) {
                    std::cout << "Sample " << total << ":" << std::endl;
                    std::cout << "  Batch input shape: " << batched_inputs.sizes() << std::endl;
                    std::cout << "  Single output: " << single_output << std::endl;
                    std::cout << "  Predictions: [" << preds[0] << ", " << preds[1] << "]" << std::endl;
                    std::cout << "  True label: " << true_cls << std::endl;
                }
            }
            
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration_sec = total_end - total_start;
        
        return {
            .accuracy = (static_cast<double>(correct) / total) * 100.0,
            .avg_inference_time_ms = total_inference_time_ms / total,
            .samples_per_second = total / total_duration_sec.count(),
            .correct_predictions = correct,
            .total_samples = total
        };
    }
    
    void print_results(const EvaluationResults& results) {
        std::cout << "\n=== EVALUATION RESULTS ===" << std::endl;
        std::cout << "Accuracy: " << results.accuracy << "%" << std::endl;
        std::cout << "Correct predictions: " << results.correct_predictions 
                  << "/" << results.total_samples << std::endl;
        std::cout << "Average inference time per sample: " 
                  << results.avg_inference_time_ms << " ms" << std::endl;
        std::cout << "Samples per second: " << results.samples_per_second << std::endl;
    }
}

// Command line argument parsing
namespace ArgumentParser {
    struct Args {
        std::string model_path;
        std::string csv_path;
        int batch_size;
    };
    
    Args parse_arguments(int argc, char* argv[]) {
        if (argc < 3) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <model_path> <csv_path> [batch_size]");
        }
        
        int batch_size = 32;  // Default batch size
        if (argc >= 4) {
            batch_size = std::atoi(argv[3]);
            if (batch_size <= 0) {
                throw std::runtime_error("Invalid batch size: " + std::string(argv[3]));
            }
        }
        
        return {
            .model_path = argv[1],
            .csv_path = argv[2],
            .batch_size = batch_size
        };
    }
}

// Clean main function
int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        auto args = ArgumentParser::parse_arguments(argc, argv);
        
        // Load model
        auto model = ModelManager::load_model(args.model_path);
        
        // Load and transform data
        auto dataset = DataManager::load_dataset(args.csv_path);
        DataManager::apply_transformations(dataset);
        
        // Evaluate model with configurable batch size
        auto results = Evaluator::evaluate_model(model, dataset, args.batch_size);
        
        // Print results
        Evaluator::print_results(results);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}