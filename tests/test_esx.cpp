
#include "parser.hpp"
#include "interpreter.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <filesystem>

/**
 * @file test_esx.cpp
 * @brief Comprehensive unit tests for EasiScriptX (ESX) v1.0.
 * @details Tests parsing and execution of ESX syntax, including new features like
 * LoRA, mixed-precision, pipeline parallelism, framework interoperability, and edge cases.
 */

void test_parser() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = x @ x
        let z = conv2d(x, x, stride:1, padding:0)
        let attn = attention(x, x, x, heads:8, dim:64, flash=true)
        let lora_x = lora(x, rank:4)
        let mp_x = mixed_precision(x, bf16)
        train(x, x, loss:ce, opt:lomo(lr=0.001), epochs:10, device:cpu)
        pipeline_parallel(x, stages:2)
        instruction_tune(x, x, "prompt")
        adapt_domain(x, scientific)
        schedule_heterogeneous(cpu:0.7, gpu:0.3)
        energy_aware(x, max_power:100)
        switch_framework(pytorch, model.pt)
        track_experiment(mlflow, run123)
    )";
    try {
        esx::ast::Program program = esx::parser::parse(source);
        assert(program.stmts.size() == 13);
        std::cout << "Parser test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Parser test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_interpreter() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = x @ x
        train(x, x, loss:ce, opt:adam(lr=0.001), epochs:1, device:cpu)
    )";
    try {
        esx::ast::Program program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Interpreter test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Interpreter test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_invalid_lora() {
    std::string source = "let x = tensor([[1,2],[3,4]]); let y = lora(x, rank:0);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid LoRA rank");
    } catch (const std::exception& e) {
        std::cout << "Invalid LoRA test passed: " << e.what() << std::endl;
    }
}

void test_empty_tensor() {
    std::string source = "let x = tensor([[]]);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on empty tensor");
    } catch (const std::exception& e) {
        std::cout << "Empty tensor test passed: " << e.what() << std::endl;
    }
}

void test_invalid_attention_heads() {
    std::string source = "let x = tensor([[1,2],[3,4]]); let y = attention(x, x, x, heads:0, dim:64);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid attention heads");
    } catch (const std::exception& e) {
        std::cout << "Invalid attention heads test passed: " << e.what() << std::endl;
    }
}

void test_invalid_conv2d_params() {
    std::string source = "let x = tensor([[1,2],[3,4]]); let y = conv2d(x, x, stride:0, padding:0);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid conv2d stride");
    } catch (const std::exception& e) {
        std::cout << "Invalid conv2d params test passed: " << e.what() << std::endl;
    }
}

void test_invalid_training_epochs() {
    std::string source = "let x = tensor([[1,2],[3,4]]); train(x, x, loss:ce, opt:adam(lr=0.001), epochs:0, device:cpu);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid epochs");
    } catch (const std::exception& e) {
        std::cout << "Invalid training epochs test passed: " << e.what() << std::endl;
    }
}

void test_invalid_pipeline_stages() {
    std::string source = "let x = tensor([[1,2],[3,4]]); pipeline_parallel(x, stages:0);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid pipeline stages");
    } catch (const std::exception& e) {
        std::cout << "Invalid pipeline stages test passed: " << e.what() << std::endl;
    }
}

void test_invalid_heterogeneous_schedule() {
    std::string source = "schedule_heterogeneous(cpu:0.5, gpu:0.3);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid schedule ratios");
    } catch (const std::exception& e) {
        std::cout << "Invalid heterogeneous schedule test passed: " << e.what() << std::endl;
    }
}

void test_invalid_energy_power() {
    std::string source = "let x = tensor([[1,2],[3,4]]); energy_aware(x, max_power:0);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid max power");
    } catch (const std::exception& e) {
        std::cout << "Invalid energy power test passed: " << e.what() << std::endl;
    }
}

void test_invalid_framework() {
    std::string source = "switch_framework(invalid_framework, model.pt);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid framework");
    } catch (const std::exception& e) {
        std::cout << "Invalid framework test passed: " << e.what() << std::endl;
    }
}

void test_invalid_tracker() {
    std::string source = "track_experiment(invalid_tracker, run123);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on invalid tracker");
    } catch (const std::exception& e) {
        std::cout << "Invalid tracker test passed: " << e.what() << std::endl;
    }
}

void test_undefined_variable() {
    std::string source = "let y = undefined_var;";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on undefined variable");
    } catch (const std::exception& e) {
        std::cout << "Undefined variable test passed: " << e.what() << std::endl;
    }
}

void test_inconsistent_tensor_dimensions() {
    std::string source = "let x = tensor([[1,2],[3,4,5]]);";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        assert(false && "Should throw on inconsistent tensor dimensions");
    } catch (const std::exception& e) {
        std::cout << "Inconsistent tensor dimensions test passed: " << e.what() << std::endl;
    }
}

void test_memory_broker() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        memory_broker(x, max_mem:8, strategy:zeRO)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Memory broker test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Memory broker test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_quantization() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        quantize(x, bits:8, method:ptq)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Quantization test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Quantization test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_speculative_decoding() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = tensor([[5,6],[7,8]])
        speculative_decode(x, y, max_tokens:100)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Speculative decoding test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Speculative decoding test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_pattern_recognition() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        pattern_recognize(x, rules:geometric)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Pattern recognition test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Pattern recognition test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_gradient_checkpointing() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        checkpoint(x, segments:4)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Gradient checkpointing test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Gradient checkpointing test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_kernel_fusion() {
    std::string source = R"(
        let x = tensor([[1,2],[3,4]])
        let y = tensor([[5,6],[7,8]])
        let z = fused_matmul_relu(x, y)
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        std::cout << "Kernel fusion test passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Kernel fusion test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_performance_profiling() {
    std::string source = R"(
        let x = tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let y = x @ x
        let attn = attention(x, x, x, heads:8, dim:4, flash=true)
    )";
    
    auto start = std::chrono::high_resolution_clock::now();
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Performance profiling test passed - Execution time: " << duration.count() << "us" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Performance profiling test failed: " << e.what() << std::endl;
        assert(false);
    }
}

void test_visualize() {
    std::string source = R"(
        let x = tensor([[1,0,0],[0,1,0],[0,0,1]])
        visualize(x, filename: "test_output.ppm")
    )";
    try {
        auto program = esx::parser::parse(source);
        esx::runtime::Interpreter interpreter;
        interpreter.run(program);
        
        if (std::filesystem::exists("test_output.ppm")) {
            std::cout << "Visualize test passed" << std::endl;
            std::filesystem::remove("test_output.ppm");
        } else {
            std::cerr << "Visualize test failed: Output file not created" << std::endl;
            assert(false);
        }
    } catch (const std::exception& e) {
        std::cerr << "Visualize test failed: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "Running comprehensive EasiScriptX tests..." << std::endl;
    
    // Basic functionality tests
    test_parser();
    test_interpreter();
    
    // Negative test cases
    test_invalid_lora();
    test_empty_tensor();
    test_invalid_attention_heads();
    test_invalid_conv2d_params();
    test_invalid_training_epochs();
    test_invalid_pipeline_stages();
    test_invalid_heterogeneous_schedule();
    test_invalid_energy_power();
    test_invalid_framework();
    test_invalid_tracker();
    test_undefined_variable();
    test_inconsistent_tensor_dimensions();
    
    // New feature tests
    test_memory_broker();
    test_quantization();
    test_speculative_decoding();
    test_pattern_recognition();
    test_gradient_checkpointing();
    test_kernel_fusion();
    test_performance_profiling();
    test_visualize();
    
    std::cout << "All comprehensive tests passed!" << std::endl;
    return 0;
}
