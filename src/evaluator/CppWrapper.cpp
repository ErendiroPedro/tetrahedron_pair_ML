// IN DEVELOPMENT
#include <torch/script.h>
#include <vector>
#include <iostream>

class TetrahedraPairIntersectionAnalysis {
private:
    torch::jit::script::Module model;

public:
    TetrahedraPairIntersectionAnalysis(const std::string& model_path) {
        try {
            model = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }

    void preprocessing(std::vector<float>& input1, std::vector<float>& input2) {

        // Transform Data

        // Convert vectors to tensors
        at::Tensor tensor1 = torch::from_blob((void*)input1.data(), {1, static_cast<long>(input1.size())}, torch::kFloat32);
        at::Tensor tensor2 = torch::from_blob((void*)input2.data(), {1, static_cast<long>(input2.size())}, torch::kFloat32);
        
        // Create vector of inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor1);
        inputs.push_back(tensor2);
    }

    std::pair<bool, float> predict(const std::vector<float>& input1, const std::vector<float>& input2) {
        try {
            // Run inference
            auto output = model.forward(inputs).toTuple();

            // Extract results
            at::Tensor intersect_tensor = output->elements()[0].toTensor();
            at::Tensor volume_tensor = output->elements()[1].toTensor();

            // Convert to native types
            bool intersects = intersect_tensor.item<float>() > 0.5f;  // Assuming sigmoid output
            float volume = volume_tensor.item<float>();

            return {intersects, volume};
        }
        catch (const c10::Error& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            throw;
        }
    }
};

// Example usage
int main() {
    TetrahedraPairIntersectionAnalysis model("model.pt");
    
    std::vector<float> tetra1 {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> tetra2 {0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f};
    
    auto [intersects, volume] = model.predict(tetra1, tetra2);
    
    std::cout << "Intersection: " << (intersects ? "Yes" : "No") << "\n";
    std::cout << "Volume: " << volume << std::endl;
    
    return 0;
}