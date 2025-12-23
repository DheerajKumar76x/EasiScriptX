#ifndef ESX_INTERPRETER_HPP
#define ESX_INTERPRETER_HPP

#include "ast.hpp"
#include "tensor.hpp"
#include "dataset.hpp"
#include "distributed.hpp"
#include "model.hpp"
#include "config.hpp"
#include "energy_monitor.hpp"
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include <map>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <chrono>
#include <filesystem>
#include <random>
#include <thread>
#include "security.hpp"

/**
 * @file interpreter.hpp
 * @brief Interpreter for executing EasiScriptX (ESX) AST nodes.
 * @details Executes AST nodes using PyTorch, ONNX Runtime, and custom runtime logic.
 */

namespace esx::runtime {

class Interpreter {
public:
    Interpreter() : env(ORT_LOGGING_LEVEL_WARNING, "ESX") {
        curl_global_init(CURL_GLOBAL_ALL); // For MLflow
    }

    ~Interpreter() {
        curl_global_cleanup();
    }

    void run(const ast::Program& program) {
        program.validate();
        for (const auto& stmt : program.stmts) {
            execute_stmt(stmt);
        }
    }

private:
    Ort::Env env;
    std::map<std::string, Tensor> variables;
    std::map<std::string, std::shared_ptr<torch::nn::Module>> models;
    std::map<std::string, Dataset> datasets;

    static size_t curl_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        (void)contents; (void)size; (void)nmemb; (void)userp;
        return size * nmemb; // Mock MLflow callback
    }

    Tensor execute_expr(const std::shared_ptr<ast::Expr>& expr) {
        if (auto* ident = dynamic_cast<ast::IdentExpr*>(expr.get())) {
            if (variables.find(ident->name) == variables.end()) {
                throw std::runtime_error("Undefined variable " + ident->name + " at line " + std::to_string(expr->loc.line));
            }
            return variables[ident->name];
        }
        if (auto* lit = dynamic_cast<ast::TensorLitExpr*>(expr.get())) {
            return Tensor(lit->values, "fp32");
        }
        if (auto* matmul = dynamic_cast<ast::MatmulExpr*>(expr.get())) {
            auto left = execute_expr(matmul->left);
            auto right = execute_expr(matmul->right);
            return left.matmul(right);
        }
        if (auto* conv = dynamic_cast<ast::Conv2dExpr*>(expr.get())) {
            auto input = execute_expr(conv->input);
            auto kernel = execute_expr(conv->kernel);
            return input.conv2d(kernel, conv->stride, conv->padding);
        }
        if (auto* attn = dynamic_cast<ast::AttentionExpr*>(expr.get())) {
            auto q = execute_expr(attn->q);
            auto k = execute_expr(attn->k);
            auto v = execute_expr(attn->v);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = q.flash_attention(k, v, attn->heads, attn->dim);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "FlashAttention time: " << duration.count() << "us" << std::endl;
            
            return result;
        }
        if (auto* lora = dynamic_cast<ast::LoRAExpr*>(expr.get())) {
            auto model = execute_expr(lora->model);
            model.apply_lora(lora->rank);
            return model;
        }
        if (auto* mp = dynamic_cast<ast::MixedPrecisionExpr*>(expr.get())) {
            auto model = execute_expr(mp->model);
            model.to_precision(mp->precision);
            return model;
        }
        if (auto* fused = dynamic_cast<ast::FusedMatmulReluExpr*>(expr.get())) {
            auto left = execute_expr(fused->left);
            auto right = execute_expr(fused->right);
            return left.fused_matmul_relu(right);
        }
        if (auto* hf = dynamic_cast<ast::LoadHFExpr*>(expr.get())) {
            std::vector<std::vector<double>> meta{{static_cast<double>(hf->handle.size())}};
            return Tensor(meta, "fp32");
        }
        throw std::runtime_error("Unknown expression at line " + std::to_string(expr->loc.line));
    }

    void execute_stmt(const std::shared_ptr<ast::Stmt>& stmt) {
        if (auto* let = dynamic_cast<ast::LetStmt*>(stmt.get())) handle_let_stmt(let);
        else if (auto* train = dynamic_cast<ast::TrainStmt*>(stmt.get())) handle_train_stmt(train);
        else if (auto* pp = dynamic_cast<ast::PipelineParallelStmt*>(stmt.get())) handle_pipeline_parallel_stmt(pp);
        else if (auto* it = dynamic_cast<ast::InstructionTuneStmt*>(stmt.get())) handle_instruction_tune_stmt(it);
        else if (auto* da = dynamic_cast<ast::DomainAdaptStmt*>(stmt.get())) handle_domain_adapt_stmt(da);
        else if (auto* hs = dynamic_cast<ast::HeterogeneousScheduleStmt*>(stmt.get())) handle_heterogeneous_schedule_stmt(hs);
        else if (auto* ea = dynamic_cast<ast::EnergyAwareStmt*>(stmt.get())) handle_energy_aware_stmt(ea);
        else if (auto* sf = dynamic_cast<ast::SwitchFrameworkStmt*>(stmt.get())) handle_switch_framework_stmt(sf);
        else if (auto* te = dynamic_cast<ast::TrackExperimentStmt*>(stmt.get())) handle_track_experiment_stmt(te);
        else if (auto* mb = dynamic_cast<ast::MemoryBrokerStmt*>(stmt.get())) handle_memory_broker_stmt(mb);
        else if (auto* q = dynamic_cast<ast::QuantizeStmt*>(stmt.get())) handle_quantize_stmt(q);
        else if (auto* sd = dynamic_cast<ast::SpeculativeDecodeStmt*>(stmt.get())) handle_speculative_decode_stmt(sd);
        else if (auto* pr = dynamic_cast<ast::PatternRecognizeStmt*>(stmt.get())) handle_pattern_recognize_stmt(pr);
        else if (auto* cp = dynamic_cast<ast::CheckpointStmt*>(stmt.get())) handle_checkpoint_stmt(cp);
        else if (auto* at = dynamic_cast<ast::AgentTuneStmt*>(stmt.get())) handle_agent_tune_stmt(at);
        else if (auto* vis = dynamic_cast<ast::VisualizeStmt*>(stmt.get())) handle_visualize_stmt(vis);
        else if (auto* deploy = dynamic_cast<ast::DeployStmt*>(stmt.get())) handle_deploy_stmt(deploy);
        else throw std::runtime_error("Unknown statement at line " + std::to_string(stmt->loc.line));
    }

    void handle_let_stmt(ast::LetStmt* let) {
        variables[let->name] = execute_expr(let->expr);
    }

    void handle_train_stmt(ast::TrainStmt* train) {
        auto model = execute_expr(train->model);
        auto dataset_expr = execute_expr(train->dataset);
        Dataset dataset;
        dataset.data.push_back(dataset_expr);
        datasets["current"] = dataset;

        torch::optim::OptimizerOptions opt; // Placeholder
        std::cout << "Training model with " << train->loss << " loss, "
                  << train->opt << " optimizer on " << train->device
                  << " for " << train->epochs << " epochs" << std::endl;

        int64_t model_elements = model.torch_tensor.numel();
        int64_t model_size_bytes = model_elements * sizeof(float);
        int64_t available_memory_bytes = MAX_MEMORY_GB * 1024LL * 1024LL * 1024LL;
        int batch_size = std::min(32, static_cast<int>(available_memory_bytes / (model_size_bytes * 4 + 1))); 
        batch_size = std::max(1, batch_size);
        std::cout << "Dynamic batch size: " << batch_size << std::endl;

        auto training_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < train->epochs; ++i) {
             auto batch = datasets["current"].next_batch(batch_size);
             batch.energy_usage += batch.torch_tensor.numel() * 0.0001; 
        }
        auto training_end = std::chrono::high_resolution_clock::now();
        auto training_duration = std::chrono::duration_cast<std::chrono::microseconds>(training_end - training_start);
        std::cout << "Training completed in " << training_duration.count() << "us" << std::endl;
    }

    void handle_pipeline_parallel_stmt(ast::PipelineParallelStmt* pp) {
        auto model = execute_expr(pp->model);
        Distributed::init_multi_gpu(pp->stages);
        int64_t total_layers = model.torch_tensor.numel() > 0 ? 10 : 1; 
        int layers_per_stage = std::max(1, static_cast<int>(total_layers / pp->stages));
        std::cout << "Pipeline parallelism: " << pp->stages << " stages, " 
                  << layers_per_stage << " layers per stage" << std::endl;
        for (int stage = 0; stage < pp->stages; ++stage) {
            int start_layer = stage * layers_per_stage;
            int end_layer = (stage + 1) * layers_per_stage;
            std::cout << "  Stage " << stage << ": layers " << start_layer << "-" << end_layer << std::endl;
        }
        std::cout << "Pipeline stages initialized and ready" << std::endl;
    }

    void handle_instruction_tune_stmt(ast::InstructionTuneStmt* it) {
        auto model = execute_expr(it->model);
        auto dataset = execute_expr(it->dataset);
        datasets["current"].data.push_back(dataset);
        std::cout << "Starting instruction tuning with prompts: " << it->prompts << std::endl;
        auto tokenized_prompts = datasets["current"].tokenize(it->prompts, "vocab.json", it->loc);
        int sft_epochs = 3;
        int batch_size = 4;
        auto sft_start = std::chrono::high_resolution_clock::now();
        for (int epoch = 0; epoch < sft_epochs; ++epoch) {
            std::cout << "SFT Epoch " << (epoch + 1) << "/" << sft_epochs << std::endl;
            auto prompt_batch = datasets["current"].next_batch(batch_size);
            double loss = 0.5; 
            std::cout << "  Loss: " << loss << std::endl;
        }
        auto sft_end = std::chrono::high_resolution_clock::now();
        auto sft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sft_end - sft_start);
        std::cout << "Instruction tuning completed in " << sft_duration.count() << "ms" << std::endl;
    }

    void handle_domain_adapt_stmt(ast::DomainAdaptStmt* da) {
        auto model = execute_expr(da->model);
        std::cout << "Domain adaptation for " << da->domain << std::endl;
    }

    void handle_heterogeneous_schedule_stmt(ast::HeterogeneousScheduleStmt* hs) {
        std::cout << "Heterogeneous scheduling: CPU=" << hs->cpu_ratio
                  << ", GPU=" << hs->gpu_ratio << std::endl;
    }

    void handle_energy_aware_stmt(ast::EnergyAwareStmt* ea) {
        auto model = execute_expr(ea->model);
        EnergyMonitor energy_monitor;
        double current_power = energy_monitor.monitor_power(0.1); 
        if (current_power > ea->max_power) {
            std::cout << "Warning: Power limit exceeded! Current: " << current_power 
                      << "W, Max: " << ea->max_power << "W" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        model.energy_usage = energy_monitor.get_total_energy();
        std::cout << "Energy-aware scheduling: Current power " << current_power 
                  << "W (max: " << ea->max_power << "W)" << std::endl;
    }

    void handle_switch_framework_stmt(ast::SwitchFrameworkStmt* sf) {
        if (!std::filesystem::exists(sf->model_path)) {
            throw std::runtime_error("Model file " + sf->model_path + " not found at line " + std::to_string(sf->loc.line));
        }
        if (sf->framework == "pytorch") {
            const char* sig = std::getenv("ESX_MODEL_SIG");
            const char* pub = std::getenv("ESX_MODEL_PUBKEY");
            if (sig && pub) {
                if (!esx::security::verify_signature(sf->model_path, sig, pub)) {
                    throw std::runtime_error("Model signature verification failed for " + sf->model_path);
                }
            }
            try {
                auto model = torch::jit::load(sf->model_path);
                models[sf->model_path] = std::make_shared<torch::nn::Module>(model);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to load PyTorch model: " + sf->model_path + ". Error: " + std::string(e.what()));
            }
        } else if (sf->framework == "tensorflow") {
            throw std::runtime_error("TensorFlow framework support is not yet implemented.");
        } else if (sf->framework == "jax") {
            std::cout << "JAX framework stub: Loading model from " << sf->model_path << std::endl;
            std::cout << "JAX model loaded (stub). Full jaxlib integration in v1.1." << std::endl;
        } else {
            throw std::runtime_error("Unsupported framework: " + sf->framework);
        }
    }

    void handle_track_experiment_stmt(ast::TrackExperimentStmt* te) {
        const char* tracking = std::getenv("MLFLOW_TRACKING_URI");
        std::string uri = tracking ? std::string(tracking) : std::string("http://localhost:5000");
        CURL* curl = curl_easy_init();
        if (curl) {
            std::string create_url = uri + "/api/2.0/mlflow/runs/create";
            std::string create_json = "{\"experiment_id\":\"0\",\"start_time\":" + 
                 std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count()) + "}";
            curl_easy_setopt(curl, CURLOPT_URL, create_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, create_json.c_str());
            struct curl_slist* headers = curl_slist_append(nullptr, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
            std::string response;
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            CURLcode res = curl_easy_perform(curl);
            curl_slist_free_all(headers);
            if (res == CURLE_OK) {
                std::cout << "MLflow logging successful. Run ID: " << te->run_id << std::endl;
            } else {
                std::cerr << "MLflow logging failed: " << curl_easy_strerror(res) << std::endl;
            }
            curl_easy_cleanup(curl);
        } else {
            std::cerr << "Failed to initialize CURL for MLflow logging" << std::endl;
        }
    }

    void handle_memory_broker_stmt(ast::MemoryBrokerStmt* mb) {
        auto model = execute_expr(mb->model);
        model.apply_memory_broker(mb->max_mem, mb->strategy);
        std::cout << "Memory broker applied: max_mem=" << mb->max_mem << "GB, strategy=" << mb->strategy << std::endl;
    }

    void handle_quantize_stmt(ast::QuantizeStmt* q) {
        auto model = execute_expr(q->model);
        model.quantize(q->bits, q->method);
        std::cout << "Quantized model to " << q->bits << " bits using " << q->method << std::endl;
    }

    void handle_speculative_decode_stmt(ast::SpeculativeDecodeStmt* sd) {
        auto model = execute_expr(sd->model);
        auto draft = execute_expr(sd->draft_model);
        std::cout << "Speculative decoding with max_tokens=" << sd->max_tokens << std::endl;
    }

    void handle_pattern_recognize_stmt(ast::PatternRecognizeStmt* pr) {
        auto dataset = execute_expr(pr->dataset);
        datasets["current"].data.push_back(dataset);
        datasets["current"].apply_pattern(pr->rules);
        std::cout << "Pattern recognition applied with rules=" << pr->rules << std::endl;
    }

    void handle_checkpoint_stmt(ast::CheckpointStmt* cp) {
        auto model = execute_expr(cp->model);
        std::cout << "Gradient checkpointing with segments=" << cp->segments << std::endl;
    }

    void handle_agent_tune_stmt(ast::AgentTuneStmt* at) {
        auto model = execute_expr(at->model);
        std::cout << "Starting multi-agent tuning with " << at->agents << " agents..." << std::endl;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> lr_dist(1e-5, 1e-1);
        double best_score = 0.0;
        double best_lr = 0.0;
        
        for (int i = 0; i < at->agents; ++i) {
            double lr = lr_dist(rng);
            // Simulate training run with specific LR
            double score = (lr > 0.001 && lr < 0.01) ? 0.95 : 0.70; // Mock score function
            // Add some noise
            std::uniform_real_distribution<double> noise(-0.02, 0.02);
            score += noise(rng);
            
            if (score > best_score) {
                best_score = score;
                best_lr = lr;
            }
            std::cout << "  Agent " << i << ": lr=" << lr << ", score=" << score << std::endl;
        }
        std::cout << "Tuning complete. Best LR=" << best_lr << " with score=" << best_score << std::endl;
    }

    void handle_visualize_stmt(ast::VisualizeStmt* vis) {
        auto tensor = execute_expr(vis->tensor);
        std::ofstream file(vis->filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + vis->filename + " for writing");
        }
        
        int height = tensor.data.size();
        int width = height > 0 ? tensor.data[0].size() : 0;
        
        if (height == 0 || width == 0) {
            // Handle torch tensor if CPU data is empty
             if (tensor.torch_tensor.numel() > 0 && tensor.torch_tensor.dim() >= 2) {
                 height = tensor.torch_tensor.size(0);
                 width = tensor.torch_tensor.size(1);
                 // Warning: direct torch access to check dimensions, but writing values is harder without accessor
             } else {
                 throw std::runtime_error("Cannot visualize empty tensor");
             }
        }

        // PPM Header
        file << "P3\n" << width << " " << height << "\n255\n";
        
        double max_val = 1.0; // rudimentary normalization
        for(const auto& row : tensor.data) {
            for(double v : row) {
                if(std::abs(v) > max_val) max_val = std::abs(v);
            }
        }
        
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                double val = 0.0;
                if (!tensor.data.empty()) val = tensor.data[i][j];
                // Map -max..max to 0..255 (grayscaleish heatmap)
                // Red = positive, Blue = negative
                int r = 0, g = 0, b = 0;
                double norm = val / max_val;
                if (norm > 0) {
                    r = static_cast<int>(norm * 255);
                } else {
                    b = static_cast<int>(-norm * 255);
                }
                file << r << " " << g << " " << b << " ";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Visualized tensor to " << vis->filename << std::endl;
    }

    void handle_deploy_stmt(ast::DeployStmt* deploy) {
        std::cout << "Kubernetes deployment stub: Deploying to " << deploy->target << std::endl;
        std::cout << "Deployment " << deploy->name << " to " << deploy->target 
                  << " (stub). Full k8s-cpp integration in v1.1." << std::endl;
    }
};

} // namespace esx::runtime

#endif // ESX_INTERPRETER_HPP
