#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>

// Constants for the network
constexpr int INPUT_SIZE = 24;  // 24 features from the tetrahedron pair
constexpr int HIDDEN_SIZE = 24; // Hidden layer size
constexpr int OUTPUT_SIZE = 1;  // Single boolean output

// Kernel to compute hidden layer activations
__global__ void computeHiddenLayer(
    const float *input,
    const float *weights,
    const float *bias,
    float *output,
    int batchSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batchSize)
    {
        for (int neuron = 0; neuron < HIDDEN_SIZE; neuron++)
        {
            float sum = bias[neuron];
            for (int i = 0; i < INPUT_SIZE; i++)
            {
                sum += input[row * INPUT_SIZE + i] * weights[neuron * INPUT_SIZE + i];
            }
            // ReLU activation
            output[row * HIDDEN_SIZE + neuron] = (sum > 0) ? sum : 0;
        }
    }
}

// Kernel to compute output layer
__global__ void computeOutputLayer(
    const float *hiddenOutput,
    const float *weights,
    const float *bias,
    float *output,
    int batchSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batchSize)
    {
        float sum = bias[0];
        for (int i = 0; i < HIDDEN_SIZE; i++)
        {
            sum += hiddenOutput[row * HIDDEN_SIZE + i] * weights[i];
        }
        // Model trained with binary cross-entropy loss with logits, no need to apply sigmoid
        output[row] = sum;
    }
}

class CUDAIntersectionPredictor
{
private:
    // Host arrays for weights and biases
    float h_hidden_weights[INPUT_SIZE * HIDDEN_SIZE];
    float h_hidden_bias[HIDDEN_SIZE];
    float h_output_weights[HIDDEN_SIZE];
    float h_output_bias[OUTPUT_SIZE];

    // Device pointers
    float *d_input, *d_hidden_weights, *d_hidden_bias;
    float *d_hidden_output, *d_output_weights, *d_output_bias, *d_output;

    bool initialized;
    int maxBatchSize;

public:
    CUDAIntersectionPredictor(int batchSize = 32) : initialized(false), maxBatchSize(batchSize)
    {
        // Initialize CUDA resources
        initialize();
    }

    ~CUDAIntersectionPredictor()
    {
        if (initialized)
        {
            cleanup();
        }
    }

    void initialize()
    {
        if (initialized)
            return;

        // Allocate device memory
        cudaMalloc(&d_input, maxBatchSize * INPUT_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_bias, HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_hidden_output, maxBatchSize * HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_output_weights, HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_output_bias, OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&d_output, maxBatchSize * OUTPUT_SIZE * sizeof(float));

        // Copy weights to device (would normally load from a trained model)
        cudaMemcpy(d_hidden_weights, h_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden_bias, h_hidden_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_weights, h_output_weights, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_bias, h_output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        initialized = true;
    }

    void loadModelWeights(torch::jit::Module &model)
    {
        // Extract weights from PyTorch model
        // This is a simplified example - you'll need to adjust for your model structure
        auto params = model.named_parameters();
        for (auto &param : params)
        {
            std::string name = param.name;
            torch::Tensor tensor = param.value;

            if (name == "hidden.weight")
            {
                // Copy weights to host array
                std::memcpy(h_hidden_weights, tensor.data_ptr<float>(),
                            INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            }
            else if (name == "hidden.bias")
            {
                std::memcpy(h_hidden_bias, tensor.data_ptr<float>(),
                            HIDDEN_SIZE * sizeof(float));
            }
            else if (name == "output.weight")
            {
                std::memcpy(h_output_weights, tensor.data_ptr<float>(),
                            HIDDEN_SIZE * sizeof(float));
            }
            else if (name == "output.bias")
            {
                std::memcpy(h_output_bias, tensor.data_ptr<float>(),
                            OUTPUT_SIZE * sizeof(float));
            }
        }

        // Update device memory
        cudaMemcpy(d_hidden_weights, h_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden_bias, h_hidden_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_weights, h_output_weights, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_bias, h_output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::vector<bool> predict(const std::vector<std::vector<float>> &inputs)
    {
        int batchSize = inputs.size();
        if (batchSize > maxBatchSize)
        {
            std::cerr << "Batch size exceeds maximum allowed" << std::endl;
            return std::vector<bool>(batchSize, false);
        }

        // Flatten input data
        std::vector<float> flatInputs;
        for (const auto &input : inputs)
        {
            flatInputs.insert(flatInputs.end(), input.begin(), input.end());
        }

        // Copy input to device
        cudaMemcpy(d_input, flatInputs.data(), batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernels
        dim3 hiddenBlockDim(1, 32);
        dim3 hiddenGridDim(1, (batchSize + hiddenBlockDim.y - 1) / hiddenBlockDim.y);
        computeHiddenLayer<<<hiddenGridDim, hiddenBlockDim>>>(
            d_input, d_hidden_weights, d_hidden_bias, d_hidden_output, batchSize);

        dim3 outputBlockDim(32);
        dim3 outputGridDim((batchSize + outputBlockDim.x - 1) / outputBlockDim.x);
        computeOutputLayer<<<outputGridDim, outputBlockDim>>>(
            d_hidden_output, d_output_weights, d_output_bias, d_output, batchSize);

        // Get results
        std::vector<float> outputs(batchSize);
        cudaMemcpy(outputs.data(), d_output, batchSize * sizeof(float), cudaMemcpyDeviceToHost);

        // Convert to boolean predictions
        std::vector<bool> predictions;
        for (float out : outputs)
        {
            predictions.push_back(out > 0.5f);
        }

        return predictions;
    }

    void cleanup()
    {
        if (!initialized)
            return;

        cudaFree(d_input);
        cudaFree(d_hidden_weights);
        cudaFree(d_hidden_bias);
        cudaFree(d_hidden_output);
        cudaFree(d_output_weights);
        cudaFree(d_output_bias);
        cudaFree(d_output);

        initialized = false;
    }
};

// Example usage
extern "C" bool predictIntersection(const float *vertices, int batchSize)
{
    static CUDAIntersectionPredictor predictor;

    // Format input data
    std::vector<std::vector<float>> inputs(1, std::vector<float>(vertices, vertices + 24));

    // Get prediction
    std::vector<bool> results = predictor.predict(inputs);

    return results[0];
}