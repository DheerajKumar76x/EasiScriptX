
#ifndef ESX_AST_HPP
#define ESX_AST_HPP

#include <string>
#include <vector>
#include <memory>
#include <variant>

/**
 * @file ast.hpp
 * @brief Abstract Syntax Tree (AST) definitions for EasiScriptX (ESX).
 * @details Defines AST nodes for ESX's declarative syntax, supporting tensor operations,
 * model training, distributed execution, LLM fine-tuning, and enterprise features.
 */

namespace esx::ast {

struct Location {
    int line;
    int column;
    Location(int l = 1, int c = 1) : line(l), column(c) {}
};

/**
 * @brief Base class for all AST nodes.
 */
struct Node {
    Location loc;
    virtual ~Node() = default;
    explicit Node(const Location& loc) : loc(loc) {}
    virtual void validate() const = 0;
};

/**
 * @brief Expression base class.
 */
struct Expr : Node {
    explicit Expr(const Location& loc) : Node(loc) {}
};

/**
 * @brief Identifier expression (e.g., variable name).
 */
struct IdentExpr : Expr {
    std::string name;
    IdentExpr(const Location& loc, const std::string& name) : Expr(loc), name(name) {}
    void validate() const override {
        if (name.empty()) {
            throw std::runtime_error("Empty identifier at line " + std::to_string(loc.line) + ", col " + std::to_string(loc.column));
        }
    }
};

/**
 * @brief Tensor literal expression (e.g., tensor([[1,2],[3,4]])).
 */
struct TensorLitExpr : Expr {
    std::vector<std::vector<double>> values;
    TensorLitExpr(const Location& loc, const std::vector<std::vector<double>>& values)
        : Expr(loc), values(values) {}
    void validate() const override {
        if (values.empty() || values[0].empty()) {
            throw std::runtime_error("Empty tensor at line " + std::to_string(loc.line) + ", col " + std::to_string(loc.column));
        }
        size_t cols = values[0].size();
        for (const auto& row : values) {
            if (row.size() != cols) {
                throw std::runtime_error("Inconsistent tensor dimensions at line " + std::to_string(loc.line));
            }
        }
    }
};

/**
 * @brief Matrix multiplication expression (e.g., a @ b).
 */
struct MatmulExpr : Expr {
    std::shared_ptr<Expr> left, right;
    MatmulExpr(const Location& loc, std::shared_ptr<Expr> left, std::shared_ptr<Expr> right)
        : Expr(loc), left(left), right(right) {}
    void validate() const override {
        left->validate();
        right->validate();
    }
};

/**
 * @brief Convolution expression (e.g., conv2d(input, kernel, stride=1, padding=0)).
 */
struct Conv2dExpr : Expr {
    std::shared_ptr<Expr> input, kernel;
    int stride, padding;
    Conv2dExpr(const Location& loc, std::shared_ptr<Expr> input, std::shared_ptr<Expr> kernel,
               int stride, int padding)
        : Expr(loc), input(input), kernel(kernel), stride(stride), padding(padding) {}
    void validate() const override {
        if (stride <= 0 || padding < 0) {
            throw std::runtime_error("Invalid conv2d parameters at line " + std::to_string(loc.line));
        }
        input->validate();
        kernel->validate();
    }
};

/**
 * @brief Attention expression with FlashAttention-2 support (e.g., attention(q, k, v, heads=8, dim=64, flash=true)).
 */
struct AttentionExpr : Expr {
    std::shared_ptr<Expr> q, k, v;
    int heads, dim;
    bool use_flash_attention;
    AttentionExpr(const Location& loc, std::shared_ptr<Expr> q, std::shared_ptr<Expr> k,
                  std::shared_ptr<Expr> v, int heads, int dim, bool use_flash_attention = false)
        : Expr(loc), q(q), k(k), v(v), heads(heads), dim(dim), use_flash_attention(use_flash_attention) {}
    void validate() const override {
        if (heads <= 0 || dim <= 0) {
            throw std::runtime_error("Invalid attention parameters at line " + std::to_string(loc.line));
        }
        q->validate();
        k->validate();
        v->validate();
    }
};

/**
 * @brief Load from Hugging Face Hub expression (e.g., load_hf("gpt2") or load_hf("dataset:imdb")).
 */
struct LoadHFExpr : Expr {
    std::string handle; // model id or dataset id, optional prefix "dataset:"
    LoadHFExpr(const Location& loc, const std::string& handle)
        : Expr(loc), handle(handle) {}
    void validate() const override {
        if (handle.empty()) {
            throw std::runtime_error("Empty Hugging Face handle at line " + std::to_string(loc.line));
        }
    }
};

/**
 * @brief LoRA expression for parameter-efficient fine-tuning (e.g., lora(model, rank=4)).
 */
struct LoRAExpr : Expr {
    std::shared_ptr<Expr> model;
    int rank;
    LoRAExpr(const Location& loc, std::shared_ptr<Expr> model, int rank)
        : Expr(loc), model(model), rank(rank) {}
    void validate() const override {
        if (rank <= 0) {
            throw std::runtime_error("Invalid LoRA rank at line " + std::to_string(loc.line));
        }
        model->validate();
    }
};

/**
 * @brief Mixed-precision training expression (e.g., mixed_precision(model, bf16)).
 */
struct MixedPrecisionExpr : Expr {
    std::shared_ptr<Expr> model;
    std::string precision; // "fp16", "bf16"
    MixedPrecisionExpr(const Location& loc, std::shared_ptr<Expr> model, const std::string& precision)
        : Expr(loc), model(model), precision(precision) {}
    void validate() const override {
        if (precision != "fp16" && precision != "bf16") {
            throw std::runtime_error("Invalid precision type at line " + std::to_string(loc.line));
        }
        model->validate();
    }
};

/**
 * @brief Statement base class.
 */
struct Stmt : Node {
    explicit Stmt(const Location& loc) : Node(loc) {}
};

/**
 * @brief Let statement for variable assignment (e.g., let x = tensor([[1,2]])).
 */
struct LetStmt : Stmt {
    std::string name;
    std::shared_ptr<Expr> expr;
    std::string type_annotation; // optional, e.g., "Tensor<f32,[2,3]>"
    LetStmt(const Location& loc, const std::string& name, std::shared_ptr<Expr> expr)
        : Stmt(loc), name(name), expr(expr) {}
    LetStmt(const Location& loc, const std::string& name, const std::string& type_annotation, std::shared_ptr<Expr> expr)
        : Stmt(loc), name(name), expr(expr), type_annotation(type_annotation) {}
    void validate() const override {
        if (name.empty()) {
            throw std::runtime_error("Empty variable name at line " + std::to_string(loc.line));
        }
        expr->validate();
    }
};

/**
 * @brief Training statement (e.g., train(model, dataset, loss: ce, opt: adam(lr=0.001), epochs=10, device: gpu)).
 */
struct TrainStmt : Stmt {
    std::shared_ptr<Expr> model, dataset;
    std::string loss; // "ce", "mse"
    std::string opt; // "adam", "sgd", "lomo"
    std::vector<std::pair<std::string, double>> opt_params; // e.g., {{"lr", 0.001}}
    int epochs;
    std::string device; // "cpu", "gpu"
    TrainStmt(const Location& loc, std::shared_ptr<Expr> model, std::shared_ptr<Expr> dataset,
              const std::string& loss, const std::string& opt,
              const std::vector<std::pair<std::string, double>>& opt_params,
              int epochs, const std::string& device)
        : Stmt(loc), model(model), dataset(dataset), loss(loss), opt(opt),
          opt_params(opt_params), epochs(epochs), device(device) {}
    void validate() const override {
        if (epochs <= 0) {
            throw std::runtime_error("Invalid epochs at line " + std::to_string(loc.line));
        }
        if (loss != "ce" && loss != "mse") {
            throw std::runtime_error("Invalid loss function at line " + std::to_string(loc.line));
        }
        if (opt != "adam" && opt != "sgd" && opt != "lomo") {
            throw std::runtime_error("Invalid optimizer at line " + std::to_string(loc.line));
        }
        if (device != "cpu" && device != "gpu") {
            throw std::runtime_error("Invalid device at line " + std::to_string(loc.line));
        }
        model->validate();
        dataset->validate();
    }
};

/**
 * @brief Pipeline parallelism statement (e.g., pipeline_parallel(model, stages=2)).
 */
struct PipelineParallelStmt : Stmt {
    std::shared_ptr<Expr> model;
    int stages;
    PipelineParallelStmt(const Location& loc, std::shared_ptr<Expr> model, int stages)
        : Stmt(loc), model(model), stages(stages) {}
    void validate() const override {
        if (stages <= 0) {
            throw std::runtime_error("Invalid pipeline stages at line " + std::to_string(loc.line));
        }
        model->validate();
    }
};

/**
 * @brief Instruction tuning statement (e.g., instruction_tune(model, dataset, prompts)).
 */
struct InstructionTuneStmt : Stmt {
    std::shared_ptr<Expr> model, dataset;
    std::string prompts;
    InstructionTuneStmt(const Location& loc, std::shared_ptr<Expr> model,
                       std::shared_ptr<Expr> dataset, const std::string& prompts)
        : Stmt(loc), model(model), dataset(dataset), prompts(prompts) {}
    void validate() const override {
        if (prompts.empty()) {
            throw std::runtime_error("Empty prompts at line " + std::to_string(loc.line));
        }
        model->validate();
        dataset->validate();
    }
};

/**
 * @brief Domain adaptation statement (e.g., adapt_domain(model, scientific)).
 */
struct DomainAdaptStmt : Stmt {
    std::shared_ptr<Expr> model;
    std::string domain;
    DomainAdaptStmt(const Location& loc, std::shared_ptr<Expr> model, const std::string& domain)
        : Stmt(loc), model(model), domain(domain) {}
    void validate() const override {
        if (domain.empty()) {
            throw std::runtime_error("Empty domain at line " + std::to_string(loc.line));
        }
        model->validate();
    }
};

/**
 * @brief Heterogeneous scheduling statement (e.g., schedule_heterogeneous(cpu: 0.7, gpu: 0.3)).
 */
struct HeterogeneousScheduleStmt : Stmt {
    double cpu_ratio, gpu_ratio;
    HeterogeneousScheduleStmt(const Location& loc, double cpu_ratio, double gpu_ratio)
        : Stmt(loc), cpu_ratio(cpu_ratio), gpu_ratio(gpu_ratio) {}
    void validate() const override {
        if (cpu_ratio + gpu_ratio != 1.0 || cpu_ratio < 0 || gpu_ratio < 0) {
            throw std::runtime_error("Invalid scheduling ratios at line " + std::to_string(loc.line));
        }
    }
};

/**
 * @brief Energy-aware scheduling statement (e.g., energy_aware(model, max_power=100)).
 */
struct EnergyAwareStmt : Stmt {
    std::shared_ptr<Expr> model;
    double max_power; // in watts
    EnergyAwareStmt(const Location& loc, std::shared_ptr<Expr> model, double max_power)
        : Stmt(loc), model(model), max_power(max_power) {}
    void validate() const override {
        if (max_power <= 0) {
            throw std::runtime_error("Invalid max power at line " + std::to_string(loc.line));
        }
        model->validate();
    }
};

/**
 * @brief Framework switching statement (e.g., switch_framework(pytorch, model.pt)).
 */
struct SwitchFrameworkStmt : Stmt {
    std::string framework; // "pytorch", "tensorflow", "jax"
    std::string model_path;
    SwitchFrameworkStmt(const Location& loc, const std::string& framework, const std::string& model_path)
        : Stmt(loc), framework(framework), model_path(model_path) {}
    void validate() const override {
        if (framework != "pytorch" && framework != "tensorflow" && framework != "jax") {
            throw std::runtime_error("Invalid framework at line " + std::to_string(loc.line));
        }
        if (model_path.empty()) {
            throw std::runtime_error("Empty model path at line " + std::to_string(loc.line));
        }
    }
};

/**
 * @brief Experiment tracking statement (e.g., track_experiment(mlflow, run123)).
 */
struct TrackExperimentStmt : Stmt {
    std::string tracker; // "mlflow"
    std::string run_id;
    TrackExperimentStmt(const Location& loc, const std::string& tracker, const std::string& run_id)
        : Stmt(loc), tracker(tracker), run_id(run_id) {}
    void validate() const override {
        if (tracker != "mlflow") {
            throw std::runtime_error("Invalid tracker at line " + std::to_string(loc.line));
        }
        if (run_id.empty()) {
            throw std::runtime_error("Empty run ID at line " + std::to_string(loc.line));
        }
    }
};

/**
 * @brief Memory Broker statement for GPU memory optimization (e.g., memory_broker(model, max_mem=8GB, strategy=zeRO)).
 */
struct MemoryBrokerStmt : Stmt {
    std::shared_ptr<Expr> model;
    double max_mem; // in GB
    std::string strategy; // "zeRO", "offload"
    MemoryBrokerStmt(const Location& loc, std::shared_ptr<Expr> model, double max_mem, const std::string& strategy)
        : Stmt(loc), model(model), max_mem(max_mem), strategy(strategy) {}
    void validate() const override {
        if (max_mem <= 0) throw std::runtime_error("Invalid max memory at line " + std::to_string(loc.line));
        if (strategy != "zeRO" && strategy != "offload") throw std::runtime_error("Invalid strategy at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief Quantization statement for model compression (e.g., quantize(model, bits=8, method=ptq)).
 */
struct QuantizeStmt : Stmt {
    std::shared_ptr<Expr> model;
    int bits;
    std::string method; // "ptq", "qat"
    QuantizeStmt(const Location& loc, std::shared_ptr<Expr> model, int bits, const std::string& method)
        : Stmt(loc), model(model), bits(bits), method(method) {}
    void validate() const override {
        if (bits != 8 && bits != 4) throw std::runtime_error("Invalid quantization bits at line " + std::to_string(loc.line));
        if (method != "ptq" && method != "qat") throw std::runtime_error("Invalid quantization method at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief Speculative Decoding statement for faster LLM inference (e.g., speculative_decode(model, draft_model, max_tokens=100)).
 */
struct SpeculativeDecodeStmt : Stmt {
    std::shared_ptr<Expr> model, draft_model;
    int max_tokens;
    SpeculativeDecodeStmt(const Location& loc, std::shared_ptr<Expr> model, std::shared_ptr<Expr> draft_model, int max_tokens)
        : Stmt(loc), model(model), draft_model(draft_model), max_tokens(max_tokens) {}
    void validate() const override {
        if (max_tokens <= 0) throw std::runtime_error("Invalid max tokens at line " + std::to_string(loc.line));
        model->validate();
        draft_model->validate();
    }
};

/**
 * @brief Pattern Recognition statement for ARC-AGI2-inspired reasoning (e.g., pattern_recognize(dataset, rules=geometric)).
 */
struct PatternRecognizeStmt : Stmt {
    std::shared_ptr<Expr> dataset;
    std::string rules; // "geometric", "arithmetic"
    PatternRecognizeStmt(const Location& loc, std::shared_ptr<Expr> dataset, const std::string& rules)
        : Stmt(loc), dataset(dataset), rules(rules) {}
    void validate() const override {
        if (rules != "geometric" && rules != "arithmetic") throw std::runtime_error("Invalid rules at line " + std::to_string(loc.line));
        dataset->validate();
    }
};

/**
 * @brief Gradient Checkpointing statement for memory reduction (e.g., checkpoint(model, segments=4)).
 */
struct CheckpointStmt : Stmt {
    std::shared_ptr<Expr> model;
    int segments;
    CheckpointStmt(const Location& loc, std::shared_ptr<Expr> model, int segments)
        : Stmt(loc), model(model), segments(segments) {}
    void validate() const override {
        if (segments <= 0) throw std::runtime_error("Invalid segments at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief Visualization statement (e.g., visualize(tensor, filename="output.ppm")).
 */
struct VisualizeStmt : Stmt {
    std::shared_ptr<Expr> tensor;
    std::string filename;
    VisualizeStmt(const Location& loc, std::shared_ptr<Expr> tensor, const std::string& filename)
        : Stmt(loc), tensor(tensor), filename(filename) {}
    void validate() const override {
        if (filename.empty()) throw std::runtime_error("Empty filename at line " + std::to_string(loc.line));
        tensor->validate();
    }
};

/**
 * @brief Fused Matrix Multiplication with ReLU expression for kernel fusion (e.g., fused_matmul_relu(a, b)).
 */
struct FusedMatmulReluExpr : Expr {
    std::shared_ptr<Expr> left, right;
    FusedMatmulReluExpr(const Location& loc, std::shared_ptr<Expr> left, std::shared_ptr<Expr> right)
        : Expr(loc), left(left), right(right) {}
    void validate() const override {
        left->validate();
        right->validate();
    }
};

/**
 * @brief Quantization expression (e.g., quantize(model, bits=8, method=ptq)).
 */
struct QuantizeExpr : Expr {
    std::shared_ptr<Expr> model;
    int bits;
    std::string method; // "ptq", "qat"
    QuantizeExpr(const Location& loc, std::shared_ptr<Expr> model, int bits, const std::string& method)
        : Expr(loc), model(model), bits(bits), method(method) {}
    void validate() const override {
        if (bits != 8 && bits != 4) throw std::runtime_error("Invalid quantization bits at line " + std::to_string(loc.line));
        if (method != "ptq" && method != "qat") throw std::runtime_error("Invalid quantization method at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief Pruning expression (e.g., prune(model, ratio=0.5)).
 */
struct PruneExpr : Expr {
    std::shared_ptr<Expr> model;
    double ratio;
    PruneExpr(const Location& loc, std::shared_ptr<Expr> model, double ratio)
        : Expr(loc), model(model), ratio(ratio) {}
    void validate() const override {
        if (ratio < 0.0 || ratio > 1.0) throw std::runtime_error("Invalid pruning ratio at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief RNN expression (e.g., rnn(input, hidden_size=128, layers=2)).
 */
struct RNNExpr : Expr {
    std::shared_ptr<Expr> input;
    int hidden_size, layers;
    RNNExpr(const Location& loc, std::shared_ptr<Expr> input, int hidden_size, int layers)
        : Expr(loc), input(input), hidden_size(hidden_size), layers(layers) {}
    void validate() const override {
        if (hidden_size <= 0 || layers <= 0) throw std::runtime_error("Invalid RNN parameters at line " + std::to_string(loc.line));
        input->validate();
    }
};

/**
 * @brief Transformer block expression (e.g., transformer_block(input, heads=8, dim=64)).
 */
struct TransformerExpr : Expr {
    std::shared_ptr<Expr> input;
    int heads, dim;
    TransformerExpr(const Location& loc, std::shared_ptr<Expr> input, int heads, int dim)
        : Expr(loc), input(input), heads(heads), dim(dim) {}
    void validate() const override {
        if (heads <= 0 || dim <= 0) throw std::runtime_error("Invalid transformer parameters at line " + std::to_string(loc.line));
        input->validate();
    }
};

/**
 * @brief Distributed training statement (e.g., distribute(gpus: 2) { ... }).
 */
struct DistributeStmt : Stmt {
    int gpus;
    std::vector<std::shared_ptr<Stmt>> body;
    DistributeStmt(const Location& loc, int gpus, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), gpus(gpus), body(body) {}
    void validate() const override {
        if (gpus <= 0) throw std::runtime_error("Invalid GPU count at line " + std::to_string(loc.line));
        for (const auto& stmt : body) {
            stmt->validate();
        }
    }
};

/**
 * @brief Autonomic optimization statement (e.g., with autonomic { ... }).
 */
struct AutonomicStmt : Stmt {
    std::vector<std::shared_ptr<Stmt>> body;
    AutonomicStmt(const Location& loc, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), body(body) {}
    void validate() const override {
        for (const auto& stmt : body) {
            stmt->validate();
        }
    }
};

/**
 * @brief Model definition statement (e.g., model mymodel { ... }).
 */
struct ModelStmt : Stmt {
    std::string name;
    std::vector<std::shared_ptr<Stmt>> body;
    ModelStmt(const Location& loc, const std::string& name, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), name(name), body(body) {}
    void validate() const override {
        if (name.empty()) throw std::runtime_error("Empty model name at line " + std::to_string(loc.line));
        for (const auto& stmt : body) {
            stmt->validate();
        }
    }
};

/**
 * @brief Custom loss function statement (e.g., fn custom_loss(pred, true_val) { return expr; }).
 */
struct CustomLossStmt : Stmt {
    std::string name;
    std::vector<std::string> params;
    std::shared_ptr<Expr> body;
    CustomLossStmt(const Location& loc, const std::string& name, const std::vector<std::string>& params, std::shared_ptr<Expr> body)
        : Stmt(loc), name(name), params(params), body(body) {}
    void validate() const override {
        if (name.empty()) throw std::runtime_error("Empty function name at line " + std::to_string(loc.line));
        body->validate();
    }
};

/**
 * @brief Agent-based tuning statement (e.g., agent_tune(model, agents: 4, target: "val_acc")).
 */
struct AgentTuneStmt : Stmt {
    std::shared_ptr<Expr> model;
    int agents;
    std::string target_metric;
    AgentTuneStmt(const Location& loc, std::shared_ptr<Expr> model, int agents, const std::string& target_metric)
        : Stmt(loc), model(model), agents(agents), target_metric(target_metric) {}
    void validate() const override {
        if (agents <= 0) throw std::runtime_error("Invalid agents count at line " + std::to_string(loc.line));
        if (target_metric.empty()) throw std::runtime_error("Empty target metric at line " + std::to_string(loc.line));
        model->validate();
    }
};

/**
 * @brief Program containing all statements.
 */
struct Program {
    std::vector<std::shared_ptr<Stmt>> stmts;
    void validate() const {
        for (const auto& stmt : stmts) {
            stmt->validate();
        }
    }
};

/**
 * @brief Macro definition (e.g., macro bench { ... }).
 */
struct MacroDefStmt : Stmt {
    std::string name;
    std::vector<std::shared_ptr<Stmt>> body;
    MacroDefStmt(const Location& loc, const std::string& name, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), name(name), body(body) {}
    void validate() const override {
        if (name.empty()) throw std::runtime_error("Empty macro name at line " + std::to_string(loc.line));
        for (const auto& s : body) s->validate();
    }
};

/**
 * @brief Macro invocation (e.g., @bench { ... }).
 */
struct MacroInvokeStmt : Stmt {
    std::string name;
    std::vector<std::shared_ptr<Stmt>> body;
    MacroInvokeStmt(const Location& loc, const std::string& name, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), name(name), body(body) {}
    void validate() const override {
        if (name.empty()) throw std::runtime_error("Empty macro name at line " + std::to_string(loc.line));
        for (const auto& s : body) s->validate();
    }
};

/**
 * @brief Async block (e.g., async { ... }).
 */
struct AsyncStmt : Stmt {
    std::vector<std::shared_ptr<Stmt>> body;
    AsyncStmt(const Location& loc, const std::vector<std::shared_ptr<Stmt>>& body)
        : Stmt(loc), body(body) {}
    void validate() const override {
        for (const auto& s : body) s->validate();
    }
};

/**
 * @brief Streaming dataset source (e.g., kafka("topic")).
 */
struct StreamDatasetExpr : Expr {
    std::string source; // e.g., "kafka"
    std::string spec;   // e.g., topic name
    StreamDatasetExpr(const Location& loc, const std::string& source, const std::string& spec)
        : Expr(loc), source(source), spec(spec) {}
    void validate() const override {
        if (source != "kafka") throw std::runtime_error("Unsupported stream source at line " + std::to_string(loc.line));
        if (spec.empty()) throw std::runtime_error("Empty stream spec at line " + std::to_string(loc.line));
    }
};

/**
 * @brief Kubernetes deployment statement (e.g., deploy("model-service", "gcr.io/model:latest", "production")).
 */
struct DeployStmt : Stmt {
    std::string name;      // Deployment name
    std::string image;     // Container image
    std::string target;    // Kubernetes namespace or cluster
    DeployStmt(const Location& loc, const std::string& name, const std::string& image, const std::string& target)
        : Stmt(loc), name(name), image(image), target(target) {}
    void validate() const override {
        if (name.empty()) throw std::runtime_error("Empty deployment name at line " + std::to_string(loc.line));
        if (image.empty()) throw std::runtime_error("Empty image at line " + std::to_string(loc.line));
        if (target.empty()) throw std::runtime_error("Empty target at line " + std::to_string(loc.line));
    }
};

} // namespace esx::ast

#endif // ESX_AST_HPP
